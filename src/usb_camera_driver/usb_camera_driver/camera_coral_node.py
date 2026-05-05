import sys
import os
# Додаємо venv щоб знайти tflite-runtime
venv_path = '/home/r1/ros2_ws/venv/lib/python3.12/site-packages'
if os.path.exists(venv_path):
    sys.path.insert(0, venv_path)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import cv2
import numpy as np
import time
import threading
from std_msgs.msg import Bool

# TFLite runtime
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from mediapipe.python._framework_bindings import tflite
    except ImportError:
        tflite = None

# ── MoveNet: 17 ключових точок тіла ──────────────────────────────────────────
MOVENET_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

MOVENET_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Мінімальна кількість впевнених точок щоб вважати що людина є в кадрі
PRESENCE_MIN_KEYPOINTS = 5
KEYPOINT_CONF_THRESHOLD = 0.30


class CameraCoralNode(Node):
    """Камера + MoveNet Lightning на Coral TPU (єдина модель, повністю в SRAM)."""

    def __init__(self):
        super().__init__('camera_coral_node')

        # ── Параметри ─────────────────────────────────────────────────────────
        self.declare_parameter('video_device', 0)
        self.declare_parameter('frame_rate', 30.0)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('pose_model_path',
            '/home/r1/ros2_ws/src/usb_camera_driver/models/'
            'movenet_single_pose_lightning_ptq_edgetpu.tflite')
        self.declare_parameter('jpeg_quality', 60)

        self.video_device    = self.get_parameter('video_device').value
        self.frame_rate      = self.get_parameter('frame_rate').value
        self.width           = self.get_parameter('width').value
        self.height          = self.get_parameter('height').value
        self.pose_model_path = self.get_parameter('pose_model_path').value
        self.jpeg_quality    = self.get_parameter('jpeg_quality').value

        # ── MoveNet Lightning на Coral TPU ────────────────────────────────────
        self.pose_interpreter = None
        try:
            if tflite is None:
                raise ImportError('TFLite runtime not found.')
            self.pose_interpreter = tflite.Interpreter(
                model_path=self.pose_model_path,
                experimental_delegates=[
                    tflite.load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1')
                ]
            )
            self.pose_interpreter.allocate_tensors()
            self.pose_in  = self.pose_interpreter.get_input_details()
            self.pose_out = self.pose_interpreter.get_output_details()
            self.get_logger().info(
                f'MoveNet Lightning on Coral TPU: {self.pose_model_path}')
        except Exception as e:
            self.get_logger().error(f'MoveNet Coral failed: {e}')
            self.pose_interpreter = None

        # ── QoS ───────────────────────────────────────────────────────────────
        video_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ── Стан присутності людини ───────────────────────────────────────
        self._voice_detected   = False
        self._person_detected  = False
        self._last_presence    = None  # Для публікації тільки при зміні стану
        self.create_subscription(Bool, 'audio/voice_detected', self._voice_cb, 10)

        # ── Топіки публікації ─────────────────────────────────────────────────
        # Анотований кадр (скелет) → Foxglove
        self.pub_compressed = self.create_publisher(
            CompressedImage, 'camera/image_processed/compressed', video_qos)

        # Чистий кадр (без накладок) → ZMQ-клієнт → PC-сервер (MiDaS)
        self.pub_raw = self.create_publisher(
            CompressedImage, 'camera/image_raw/compressed', video_qos)

        # Єдиний сигнал присутності людини (відео OR аудіо)
        self.pub_presence = self.create_publisher(Bool, 'human/presence', 10)

        # ── Камера ────────────────────────────────────────────────────────────
        self.get_logger().info(f'Opening /dev/video{self.video_device}')
        self.cap = cv2.VideoCapture(self.video_device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error(
                f'Failed to open /dev/video{self.video_device}')
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # MJPEG: камера стискає сама, CPU вільний
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # ── Потік обробки ─────────────────────────────────────────────────────
        self._stop   = threading.Event()
        self._worker = threading.Thread(target=self._pipeline_loop, daemon=True)
        self._worker.start()

        self._frame_count    = 0
        self._last_log_time  = time.time()

        self.get_logger().info(
            f'CameraCoralNode запущено: {self.width}x{self.height} '
            f'@ {self.frame_rate} FPS (тільки MoveNet Lightning)')

    # ─────────────────────────────────────────────────────────────────────────
    def _voice_cb(self, msg):
        self._voice_detected = msg.data

    # ─────────────────────────────────────────────────────────────────────────
    def _pipeline_loop(self):
        target_dt = 1.0 / self.frame_rate

        while not self._stop.is_set():
            t0 = time.time()

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            stamp = self.get_clock().now().to_msg()

            # ── 1. Чистий кадр → сервер MiDaS (ДО будь-яких накладок!) ──────
            try:
                has_raw_subs = self.pub_raw.get_subscription_count() > 0
            except Exception:
                has_raw_subs = False
            if has_raw_subs and not self._stop.is_set():
                ok_raw, raw_jpeg = cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if ok_raw:
                    raw_msg = CompressedImage()
                    raw_msg.header.stamp    = stamp
                    raw_msg.header.frame_id = 'camera_link'
                    raw_msg.format          = 'jpeg'
                    raw_msg.data            = raw_jpeg.tobytes()
                    try:
                        self.pub_raw.publish(raw_msg)
                    except Exception:
                        pass

            # ── 2. MoveNet Lightning на Coral TPU ─────────────────────────────
            if self.pose_interpreter is not None:
                self._person_detected = self._run_movenet(frame)
            else:
                self._person_detected = False

            # ── 3. Голосова активність ────────────────────────────────────────
            human_present = self._person_detected or self._voice_detected
            im_h, im_w = frame.shape[:2]

            if self._voice_detected:
                cv2.circle(frame, (30, 30), 15, (0, 200, 0), -1)
                cv2.putText(frame, 'VOICE', (50, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
                # DETECTED (якщо не намальований MoveNet)
                if not self._person_detected:
                    cv2.circle(frame, (im_w - 30, 30), 15, (0, 0, 255), -1)
                    cv2.putText(frame, 'DETECTED', (im_w - 150, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # ── 4. Публікуємо /human/presence (тільки при зміні стану) ───────
            if human_present != self._last_presence:
                self._last_presence = human_present
                try:
                    presence_msg = Bool()
                    presence_msg.data = bool(human_present)
                    self.pub_presence.publish(presence_msg)
                except Exception:
                    pass

            # ── 5. Анотований кадр → Foxglove ────────────────────────────────
            ok, jpeg_buf = cv2.imencode(
                '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok and not self._stop.is_set():
                msg = CompressedImage()
                msg.header.stamp    = stamp
                msg.header.frame_id = 'camera_link'
                msg.format          = 'jpeg'
                msg.data            = jpeg_buf.tobytes()
                try:
                    self.pub_compressed.publish(msg)
                except Exception:
                    break

            # ── Діагностика ───────────────────────────────────────────────────
            self._frame_count += 1
            now = time.time()
            if now - self._last_log_time >= 5.0:
                fps = self._frame_count / (now - self._last_log_time)
                self.get_logger().info(
                    f'Pipeline: {(now - t0)*1000:.0f}мс | '
                    f'Кадрів за {now - self._last_log_time:.0f}с: '
                    f'{self._frame_count} | FPS: {fps:.1f} | '
                    f'Людина: {"ТАК" if human_present else "ні"}')
                self._frame_count   = 0
                self._last_log_time = now

            elapsed = time.time() - t0
            sleep_t = target_dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ─────────────────────────────────────────────────────────────────────────
    def _run_movenet(self, frame) -> bool:
        """MoveNet Lightning на Coral TPU.
        Повертає True якщо знайдено людину (≥ PRESENCE_MIN_KEYPOINTS впевнених точок).
        Малює скелет на frame in-place.
        """
        pose_in = self.pose_in[0]
        ph, pw  = pose_in['shape'][1], pose_in['shape'][2]

        pose_img  = cv2.resize(frame, (pw, ph))
        pose_img  = cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB)
        pose_data = np.expand_dims(pose_img, axis=0).astype(pose_in['dtype'])

        self.pose_interpreter.set_tensor(pose_in['index'], pose_data)
        self.pose_interpreter.invoke()

        # shape: [1, 1, 17, 3]  → [ky, kx, confidence]
        keypoints = self.pose_interpreter.get_tensor(
            self.pose_out[0]['index'])[0][0]

        im_h, im_w = frame.shape[:2]
        pts = []
        confident_count = 0

        for kp in keypoints:
            ky, kx, kc = float(kp[0]), float(kp[1]), float(kp[2])
            if kc > KEYPOINT_CONF_THRESHOLD:
                px = int(kx * im_w)
                py = int(ky * im_h)
                pts.append((px, py))
                confident_count += 1
                cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)
            else:
                pts.append(None)

        person_present = confident_count >= PRESENCE_MIN_KEYPOINTS

        if person_present:
            # З'єднуємо точки скелету
            for (a, b) in MOVENET_EDGES:
                if pts[a] and pts[b]:
                    cv2.line(frame, pts[a], pts[b], (0, 255, 128), 2)

            # Індикатор DETECTED
            cv2.circle(frame, (im_w - 30, 30), 15, (0, 0, 255), -1)
            cv2.putText(frame, 'DETECTED', (im_w - 150, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f'pts:{confident_count}/17', (im_w - 150, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        return person_present

    # ─────────────────────────────────────────────────────────────────────────
    def destroy_node(self):
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=3.0)
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraCoralNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()

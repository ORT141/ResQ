import sys
import os
# Add venv to path to find tflite-runtime
venv_path = '/home/r1/ros2_ws/venv/lib/python3.12/site-packages'
if os.path.exists(venv_path):
    sys.path.insert(0, venv_path)

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image, CompressedImage, Range
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import cv2
import numpy as np
import os
import time
import threading

# Try to import tflite_runtime, fallback to mediapipe.tflite if needed
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from mediapipe.python._framework_bindings import tflite
    except ImportError:
        tflite = None


class CoralDetectionNode(Node):
    def __init__(self):
        super().__init__('coral_detection_node')

        # Parameters
        self.declare_parameter('model_path', '/home/r1/ros2_ws/src/usb_camera_driver/models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite')
        self.declare_parameter('label_path', '/home/r1/ros2_ws/src/usb_camera_driver/models/coco_labels.txt')
        self.declare_parameter('threshold', 0.5)

        self.model_path = self.get_parameter('model_path').value
        self.label_path = self.get_parameter('label_path').value
        self.threshold  = self.get_parameter('threshold').value

        self.bridge = CvBridge()

        # Load labels
        self.labels = self.load_labels(self.label_path)

        # Initialize TFLite interpreter with Edge TPU delegate
        self.get_logger().info(f'Loading model: {self.model_path}')
        try:
            if tflite is None:
                raise ImportError('TFLite runtime not found. Please install it.')

            self.interpreter = tflite.Interpreter(
                model_path=self.model_path,
                experimental_delegates=[
                    tflite.load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1')
                ]
            )
            self.interpreter.allocate_tensors()
            self.get_logger().info('Coral TPU Interpreter initialized successfully.')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize Coral TPU: {e}')
            self.get_logger().info('Falling back to CPU (this will be slow)...')
            self.interpreter = None

        self.input_details  = self.interpreter.get_input_details()  if self.interpreter else None
        self.output_details = self.interpreter.get_output_details() if self.interpreter else None

        # --- QoS: BEST_EFFORT + depth=1 ------------------------------------------------
        # Причина #1: depth=10 по умолчанию накапливает кадры пока инференс работает.
        # Со значением depth=1 старые кадры выбрасываются — всегда только свежий.
        video_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- Асинхронный инференс -------------------------------------------------------
        # Проблема причины #4: rclpy.spin() — однопоточный.
        # Пока callback выполняет инференс (100–500 мс), подписка заблокирована.
        # Решение: храним «последний непрочитанный кадр» и запускаем инференс
        # в отдельном потоке, чтобы callback немедленно освобождался.
        self._latest_msg  = None
        self._msg_lock    = threading.Lock()
        self._new_frame   = threading.Event()
        self._stop_infer  = threading.Event()
        self._last_log_time = 0.0  # Логирование раз в 5 секунд
        self._frame_count = 0
        self._infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._infer_thread.start()

        # Callback group нужен для MultiThreadedExecutor
        cb_group = MutuallyExclusiveCallbackGroup()

        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.listener_callback,
            video_qos,
            callback_group=cb_group)

        self.publisher_ = self.create_publisher(Image, 'camera/image_processed', video_qos)

        # CompressedImage: ~40 KB вместо 900 KB — именно это смотрит Foxglove
        # Причина #2: сырой Image по WebSocket → Foxglove буферизует → задержка
        self.pub_compressed_ = self.create_publisher(
            CompressedImage, 'camera/image_processed/compressed', video_qos)

        self.get_logger().info('Coral Detection Node started (async inference thread active).')

    # -----------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------
    def load_labels(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        labels = {}
        for i, line in enumerate(lines):
            label = line.strip()
            if label:
                labels[i] = label
        return labels

    # -----------------------------------------------------------------------------------
    def listener_callback(self, msg):
        """Только запоминает последний кадр и сигнализирует потоку инференса.
        Возвращается НЕМЕДЛЕННО — не блокирует executor."""
        with self._msg_lock:
            self._latest_msg = msg  # перезаписываем — старый кадр выброшен
        self._new_frame.set()

    # -----------------------------------------------------------------------------------
    def _inference_loop(self):
        """Фоновый поток: берёт последний кадр и делает инференс.
        Пока инференс идёт, callback может принимать новые кадры и снова перезаписывать _latest_msg."""
        # Принудительно привязываем поток инференса к CPU 1
        try:
            os.sched_setaffinity(0, {1})
        except Exception:
            pass  # если не получилось — не критично, taskset в скрипте сделает это
        while not self._stop_infer.is_set():
            # Ждём новый кадр (или выхода)
            triggered = self._new_frame.wait(timeout=0.5)
            if not triggered or self._stop_infer.is_set():
                continue
            self._new_frame.clear()

            # Берём последний кадр (может уже быть несколько следующих — это нормально)
            with self._msg_lock:
                msg = self._latest_msg
                self._latest_msg = None

            if msg is None:
                continue

            self._run_detection(msg)

    # -----------------------------------------------------------------------------------
    def _run_detection(self, msg):
        if self.interpreter is None:
            self.publisher_.publish(msg)
            return

        start_time = time.time()

        # Convert ROS Image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Prepare input (SSD MobileNet V2 expects 300x300)
        input_shape = self.input_details[0]['shape']
        h, w = input_shape[1], input_shape[2]
        resized_frame = cv2.resize(frame, (w, h))
        input_data = np.expand_dims(resized_frame, axis=0)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Get results: boxes, classes, scores, count
        boxes   = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores  = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        count   = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])

        im_h, im_w, _ = frame.shape
        person_detected = False

        for i in range(count):
            if scores[i] > self.threshold:
                class_id = int(classes[i])
                label = self.labels.get(class_id, str(class_id))

                if label == 'person':
                    person_detected = True
                    ymin, xmin, ymax, xmax = boxes[i]
                    left   = int(xmin * im_w)
                    right  = int(xmax * im_w)
                    top    = int(ymin * im_h)
                    bottom = int(ymax * im_h)

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, f'Person: {scores[i]:.2f}', (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw detection indicator
        if person_detected:
            # Draw a red circle and "DETECTED" text in the top-right
            cv2.circle(frame, (im_w - 30, 30), 15, (0, 0, 255), -1)
            cv2.putText(frame, 'DETECTED', (im_w - 150, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Publish annotated frame (raw — для внутреннего использования)
        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out_msg.header = msg.header
        self.publisher_.publish(out_msg)

        # Publish CompressedImage (JPEG) — для Foxglove по WebSocket
        # JPEG quality=75: хорошее качество при умеренном размере
        ok, jpeg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            comp_msg = CompressedImage()
            comp_msg.header = msg.header
            comp_msg.format = 'jpeg'
            comp_msg.data = jpeg_buf.tobytes()
            self.pub_compressed_.publish(comp_msg)

        # Диагностика — логируем раз в 5 секунд (вместо каждого кадра)
        elapsed = time.time() - start_time
        self._frame_count += 1
        now = time.time()
        if now - self._last_log_time >= 5.0:
            self.get_logger().info(
                f'Инференс: {elapsed:.3f} сек | Кадров за 5с: {self._frame_count} | '
                f'FPS: {self._frame_count / 5.0:.1f}')
            self._frame_count = 0
            self._last_log_time = now

    # -----------------------------------------------------------------------------------
    def destroy_node(self):
        self._stop_infer.set()
        self._new_frame.set()  # разбудить поток чтобы он мог завершиться
        self._infer_thread.join(timeout=3.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CoralDetectionNode()

    # MultiThreadedExecutor: позволяет callback и инференс-потоку работать параллельно
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

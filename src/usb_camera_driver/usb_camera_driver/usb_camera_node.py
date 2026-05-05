import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import cv2
import threading

class UsbCameraNode(Node):
    def __init__(self):
        super().__init__('usb_camera_node')

        # Declare and get parameters
        self.declare_parameter('video_device', 0)
        self.declare_parameter('frame_rate', 30.0)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)

        self.video_device = self.get_parameter('video_device').value
        self.frame_rate   = self.get_parameter('frame_rate').value
        self.width        = self.get_parameter('width').value
        self.height       = self.get_parameter('height').value

        # QoS профиль для видео: BEST_EFFORT + очередь 1
        # Старые кадры выбрасываются, а не накапливаются в буфере
        video_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', video_qos)

        # Открываем камеру
        self.get_logger().info(f'Opening video device /dev/video{self.video_device}')
        self.cap = cv2.VideoCapture(self.video_device, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open /dev/video{self.video_device}')
            return

        # Явно задаём разрешение — меньше кадр, меньше задержка по сети
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Ограничиваем буфер V4L2 до 1 кадра
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # --- Фоновый «захватывающий» поток ---
        # cap.grab() дешевле cap.read() — только захватывает, без декодирования.
        # Поток работает непрерывно, не давая буферу V4L2 копить старые кадры.
        # Таймер ROS вызывает cap.retrieve(), которое декодирует ПОСЛЕДНИЙ пойманный кадр.
        self._grab_lock   = threading.Lock()
        self._has_frame   = False
        self._stop_grab   = threading.Event()
        self._grab_thread = threading.Thread(target=self._grabbing_loop, daemon=True)
        self._grab_thread.start()

        self.bridge = CvBridge()

        timer_period = 1.0 / self.frame_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(
            f'USB Camera Node started: {self.width}x{self.height} @ {self.frame_rate} FPS'
        )

    def _grabbing_loop(self):
        """Фоновый поток: постоянно вызывает cap.grab() чтобы дренировать буфер V4L2.
        Благодаря этому cap.retrieve() в timer_callback всегда возвращает свежий кадр."""
        while not self._stop_grab.is_set():
            with self._grab_lock:
                grabbed = self.cap.grab()
            if grabbed:
                self._has_frame = True
            else:
                self._stop_grab.wait(timeout=0.01)  # пауза при ошибке

    def timer_callback(self):
        if not self._has_frame:
            return

        with self._grab_lock:
            ret, frame = self.cap.retrieve()

        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp   = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_link'
            self.publisher_.publish(msg)
        else:
            self.get_logger().warning('cap.retrieve() failed')

    def destroy_node(self):
        self._stop_grab.set()
        self._grab_thread.join(timeout=2.0)
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UsbCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

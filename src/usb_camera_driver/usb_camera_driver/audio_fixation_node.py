import os
import sys

# Підключаємо venv
venv_path = '/home/r1/ros2_ws/venv/lib/python3.12/site-packages'
if os.path.exists(venv_path):
    sys.path.insert(0, venv_path)

import numpy as np
import threading
import time
import alsaaudio
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

class AudioFixationNode(Node):
    """Повернено: проста детекція за рівнем RMS.
    Реагує на загальну гучність без частотної фільтрації.
    """

    def __init__(self):
        super().__init__('audio_fixation_node')

        # Параметри
        self.declare_parameter('threshold', 1800.0) # Золота середина
        self.declare_parameter('device', 'plughw:CARD=Camera,DEV=0')
        self.declare_parameter('sample_rate', 48000)
        self.declare_parameter('hangover_time', 1.5)

        self.threshold     = self.get_parameter('threshold').value
        self.device_name   = self.get_parameter('device').value
        self.sample_rate   = self.get_parameter('sample_rate').value
        self.hangover_time = self.get_parameter('hangover_time').value

        self.pub_voice = self.create_publisher(Bool, 'audio/voice_detected', 10)

        self._stop_event = threading.Event()
        self._last_voice_time = 0.0
        self._voice_active = False

        self._thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()

        self.get_logger().info(f'Simple RMS VAD restored. Threshold: {self.threshold}')

    def _audio_loop(self):
        while not self._stop_event.is_set():
            try:
                inp = alsaaudio.PCM(
                    alsaaudio.PCM_CAPTURE, 
                    alsaaudio.PCM_NORMAL, 
                    device=self.device_name,
                    channels=1,
                    rate=self.sample_rate,
                    format=alsaaudio.PCM_FORMAT_S16_LE,
                    periodsize=1024
                )

                while not self._stop_event.is_set():
                    l, data = inp.read()
                    if l > 0:
                        audio = np.frombuffer(data, dtype=np.int16).astype('float32')
                        
                        # Обчислюємо RMS
                        rms = np.sqrt(np.mean(audio**2))
                        
                        now = time.time()
                        
                        if rms > self.threshold:
                            self._last_voice_time = now
                            if not self._voice_active:
                                self._voice_active = True
                                self._publish_status(True)
                        else:
                            if self._voice_active and (now - self._last_voice_time > self.hangover_time):
                                self._voice_active = False
                                self._publish_status(False)
                                
            except Exception as e:
                self.get_logger().error(f'Audio error: {e}')
                time.sleep(2.0)

    def _publish_status(self, status):
        msg = Bool()
        msg.data = status
        self.pub_voice.publish(msg)
        self.get_logger().info(f'Voice detection: {"START" if status else "STOP"}')

    def destroy_node(self):
        self._stop_event.set()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = AudioFixationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Безпечне завершення без зайвих traceback
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()

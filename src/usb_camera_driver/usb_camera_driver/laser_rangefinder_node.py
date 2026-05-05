import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
import serial
import threading
import time

class LaserRangefinderNode(Node):
    def __init__(self):
        super().__init__('laser_rangefinder_node')
        
        # Параметры
        self.declare_parameter('port', '/dev/ttyAMA0')
        self.declare_parameter('baudrate', 9600)
        self.declare_parameter('frame_id', 'laser_rangefinder')
        
        self.port = self.get_parameter('port').value
        self.baudrate = self.get_parameter('baudrate').value
        self.frame_id = self.get_parameter('frame_id').value
        
        self.publisher_ = self.create_publisher(Range, 'laser_range', 10)
        
        self.get_logger().info(f'Starting Laser Rangefinder on {self.port} at {self.baudrate}')
        
        self.ser = None
        self.running = True
        self.thread = threading.Thread(target=self.serial_loop, daemon=True)
        self.thread.start()

    def serial_loop(self):
        last_cmd_time = 0
        buffer = bytearray()
        while self.running:
            try:
                if self.ser is None or not self.ser.is_open:
                    self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
                    self.get_logger().info(f'Connected to {self.port}')

                now = time.time()
                if now - last_cmd_time > 3.0:
                    self.ser.reset_input_buffer()
                    self.ser.write(bytes([0xAA, 0x00, 0x01, 0xBE, 0x00, 0x01, 0x00, 0x01, 0xC1])) # Laser ON
                    time.sleep(0.1)
                    self.ser.write(bytes([0xAA, 0x00, 0x00, 0x21, 0x00, 0x01, 0x00, 0x00, 0x22])) # Continuous
                    last_cmd_time = now

                if self.ser.in_waiting > 0:
                    buffer.extend(self.ser.read(self.ser.in_waiting))
                    
                    while len(buffer) >= 9:
                        if buffer[0] == 0xAA:
                            if len(buffer) >= 13 and buffer[5] == 0x04:
                                packet = buffer[:13]
                                cs = sum(packet[1:12]) & 0xFF
                                if cs == packet[12]:
                                    # Делим на 2, так как единицы измерения - 0.5 мм
                                    dist_mm = (packet[6] << 24) + (packet[7] << 16) + (packet[8] << 8) + packet[9]
                                    dist_m = (dist_mm / 2.0) / 1000.0
                                    
                                    msg = Range()
                                    msg.header.stamp = self.get_clock().now().to_msg()
                                    msg.header.frame_id = self.frame_id
                                    msg.range = float(dist_m)
                                    self.publisher_.publish(msg)
                                    last_cmd_time = now
                                buffer = buffer[13:]
                            elif len(buffer) >= 9 and buffer[5] == 0x01:
                                buffer = buffer[9:]
                            else:
                                if len(buffer) > 1 and 0xAA in buffer[1:]:
                                    buffer = buffer[buffer.find(0xAA, 1):]
                                else:
                                    buffer = bytearray()
                        else:
                            buffer.pop(0)
                
                time.sleep(0.01)
            except Exception as e:
                self.get_logger().error(f'Serial error: {e}')
                if self.ser:
                    self.ser.close()
                self.ser = None
                time.sleep(2.0)

    def destroy_node(self):
        self.running = False
        if self.ser:
            self.ser.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = LaserRangefinderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

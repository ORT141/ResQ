#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import serial
import struct
import re
import time

class VitalRadarNode(Node):
    def __init__(self):
        super().__init__('vital_radar')
        # Параметр порта (можно переопределить при запуске)
        self.declare_parameter('port', '/dev/ttyUSB1') 
        self.port = self.get_parameter('port').value

        self.pub_data = self.create_publisher(Point, '/radar/vitals', 10)
        
        self.ser = None
        self.connect_serial()
        self.timer = self.create_timer(0.05, self.read_serial)

        self.hr = 0.0
        self.br = 0.0
        self.dist = 0.0
        self.buffer = b""

    def connect_serial(self):
        try:
            if self.ser: self.ser.close()
            self.ser = serial.Serial(self.port, 115200, timeout=0.1)
            self.get_logger().info(f"Подключен к радару дыхания: {self.port}")
        except Exception as e:
            self.get_logger().error(f"Ошибка порта {self.port}: {e}")

    def read_serial(self):
        if not self.ser or not self.ser.is_open: return

        try:
            if self.ser.in_waiting:
                chunk = self.ser.read(self.ser.in_waiting)
                self.buffer += chunk
                
                # Попытка найти текст (HR/BR)
                try:
                    text = chunk.decode('utf-8', errors='ignore')
                    if "breath rate =" in text:
                        m = re.search(r"breath rate = ([\d\.]+)", text)
                        if m: self.br = float(m.group(1))
                    if "heart rate =" in text:
                        m = re.search(r"heart rate = ([\d\.]+)", text)
                        if m: self.hr = float(m.group(1))
                except: pass

                # Попытка найти бинарные данные (Dist)
                # (Упрощенный парсер для стабильности)
                self.buffer = self.buffer[-500:] # Не копим мусор
                
                # Публикуем
                msg = Point()
                msg.x = float(self.dist) # Пока 0, если не парсим бинарку
                msg.y = float(self.hr)
                msg.z = float(self.br)
                self.pub_data.publish(msg)

        except Exception as e:
            self.get_logger().warn(f"Ошибка чтения: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = VitalRadarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
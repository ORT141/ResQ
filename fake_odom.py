#!/usr/bin/env python3
"""
Fake Odometry Node
Слушает /cmd_vel и публикует /odom + TF odom->base_footprint.
Позволяет тестировать Nav2 без реальных колёс.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import math


class FakeOdometry(Node):
    def __init__(self):
        super().__init__('fake_odometry')

        # Текущая поза робота
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Текущая скорость
        self.vx = 0.0
        self.wz = 0.0

        # Издатели
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Подписчик на команды скорости
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # Таймер обновления (50 Гц)
        self.last_time = self.get_clock().now()
        self.create_timer(0.02, self.update)

        self.get_logger().info('Fake Odometry node запущен. Слушаю /cmd_vel ...')

    def cmd_vel_callback(self, msg: Twist):
        self.vx = msg.linear.x
        self.wz = msg.angular.z

    def update(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        # Интегрируем скорость в положение
        delta_x = self.vx * math.cos(self.theta) * dt
        delta_y = self.vx * math.sin(self.theta) * dt
        delta_theta = self.wz * dt

        self.x += delta_x
        self.y += delta_y
        self.theta += delta_theta

        # Нормализуем угол [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        # Кватернион из угла поворота
        qz = math.sin(self.theta / 2.0)
        qw = math.cos(self.theta / 2.0)

        # --- Публикуем TF odom -> base_footprint ---
        tf = TransformStamped()
        tf.header.stamp = now.to_msg()
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_footprint'
        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        tf.transform.translation.z = 0.0
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf)

        # --- Публикуем /odom ---
        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_footprint'

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.angular.z = self.wz

        self.odom_pub.publish(odom)


def main():
    rclpy.init()
    node = FakeOdometry()
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

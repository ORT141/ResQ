from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # USB Camera Node
        Node(
            package='usb_camera_driver',
            executable='usb_camera_node',
            name='usb_camera_node',
            parameters=[
                {'video_device': 0},
                {'frame_rate': 30.0}
            ]
        ),
        # Coral Detection Node
        Node(
            package='usb_camera_driver',
            executable='coral_detection_node',
            name='coral_detection_node',
            parameters=[
                {'threshold': 0.5}
            ]
        ),
        # Audio Fixation Node (VAD)
        Node(
            package='usb_camera_driver',
            executable='audio_fixation_node',
            name='audio_fixation_node',
            parameters=[
                {'threshold': 0.6},
                {'device': 'default'}
            ]
        )
    ])

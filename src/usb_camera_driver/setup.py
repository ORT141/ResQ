from setuptools import find_packages, setup

package_name = 'usb_camera_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/coral_camera.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='r1',
    maintainer_email='r1@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'usb_camera_node = usb_camera_driver.usb_camera_node:main',
            'coral_detection_node = usb_camera_driver.coral_detection_node:main',
            'camera_coral_node = usb_camera_driver.camera_coral_node:main',
            'laser_rangefinder_node = usb_camera_driver.laser_rangefinder_node:main',
            'audio_fixation_node = usb_camera_driver.audio_fixation_node:main',
        ],
    },
)

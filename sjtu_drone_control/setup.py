from setuptools import setup

package_name = 'sjtu_drone_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='georg.novtony@aon.at',
    description='TODO: Package description',
    license='TODO: License declaration',
    # tests_require=['pytest'],
    extras_require={'test':['pytest'],},
    entry_points={
        'console_scripts': [
            'teleop=sjtu_drone_control.teleop:main',
            'teleop_joystick=sjtu_drone_control.teleop_joystick:main',
            'fellow_path_controller=sjtu_drone_control.fellow_path:main',
        ],
    },
)

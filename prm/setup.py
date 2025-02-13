from setuptools import find_packages, setup

package_name = 'prm'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='abdo_gamal',
    maintainer_email='abdalrahmangamal1999@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    # tests_require=['pytest'],
    extras_require={'test':['pytest'],},
    entry_points={
        'console_scripts': [
        'map_publisher=prm.pointcloud_visualize:main',
        'path_publisher=prm.path:main',
        'follower=prm.follower:main',
        'prm=prm.prm:main'
        ],
    },
)

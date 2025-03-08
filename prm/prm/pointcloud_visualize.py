import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import open3d as o3d
from std_msgs.msg import Header
from ament_index_python.packages import get_package_share_directory



class PCDPublisher(Node):
    def __init__(self):
        super().__init__('pcd_publisher')
        self.publisher_ = self.create_publisher(PointCloud2, 'pointcloud', 10)
        self.timer = self.create_timer(2.0, self.timer_callback)  # Adjust the publishing rate
        # self.pcd = o3d.io.read_point_cloud("/home/intel/fiverr/md/drone_ws/src/maps/output_map.pcd")  # Adjust the path to your .pcd file
        prm_path = get_package_share_directory('sjtu_drone_bringup')
        pcd_path=os.path.join(prm_path,'map','map.pcd')
        self.pcd = o3d.io.read_point_cloud(pcd_path)

    def timer_callback(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "world"  # Frame ID
       

        points = np.asarray(self.pcd.points, dtype=np.float32)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]

        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1  # Unstructured point cloud
        cloud_msg.width = len(points)
        cloud_msg.fields = fields
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 12  # 3 floats (x, y, z) 4 bytes each
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.data = np.asarray(points, dtype=np.float32).tobytes()
        cloud_msg.is_dense = True  # No invalid (NaN, Inf) points
        self.publisher_.publish(cloud_msg)
        self.get_logger().info('Publishing Point Cloud')

def main(args=None):
    rclpy.init(args=args)
    pcd_publisher = PCDPublisher()
    rclpy.spin(pcd_publisher)
    pcd_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



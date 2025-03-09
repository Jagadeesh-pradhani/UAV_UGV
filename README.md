# UAV_UGV

```bash
ros2 topic pub /drone_goal geometry_msgs/msg/Point "{x: 14.0, y: 0.0, z: 5.0}" -1

-1.0 4.0 3.0 
-4.0 4.0 3.0
-1.0 0.0 3.0 
-4.0 0.0 3.0 
-4.0 -4.0 3.0
1.0 0.0 3.0


ros2 topic pub /drone_goal geometry_msgs/msg/Point "{x: -1.0, y: 4.0, z: 3.0}" -1
ros2 topic pub /drone_goal geometry_msgs/msg/Point "{x: -4.0, y: 4.0, z: 3.0}" -1

```
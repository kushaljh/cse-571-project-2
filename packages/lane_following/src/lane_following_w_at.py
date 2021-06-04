#!/usr/bin/env python3

import numpy as np
from scipy.stats import norm
import rospy
import tf
import math

from duckietown_msgs.msg import Twist2DStamped, Pose2DStamped
from std_msgs.msg import Int32MultiArray

from geometry_msgs.msg import Quaternion, PoseStamped, PoseArray, Pose, PoseWithCovarianceStamped
from sensor_msgs.msg import Joy

from duckietown.dtros import DTROS, NodeType, TopicType


class LaneFollowingNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(LaneFollowingNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.LOCALIZATION
        )

        # Get the vehicle name
        self.veh_name = rospy.get_namespace().strip("/") 

        # self.pose_sub = rospy.Subscriber(
        #     "/initialpose", 
        #     PoseWithCovarianceStamped, 
        #     self.clicked_pose_cb, 
        #     queue_size=1
        # )

        self.joy_pub = rospy.Publisher(
            f"/{self.veh_name}/joy", 
            Joy, 
            queue_size=1
        )

        self.sub_sensor = rospy.Subscriber(
            f"/{self.veh_name}/detected_tags",
            Int32MultiArray,
            self.at_callback,
            queue_size=1
        )

        # self.pub_pose = rospy.Publisher(
        #     "/pose",
        #     PoseStamped,
        #     queue_size=1
        # )

        self.tf_listener = tf.TransformListener()

        self.log("Initialized.")
        self.start_following()

    def start_following(self):
        msg_joy = Joy()
        msg_joy.axes = [0.0] * 8
        msg_joy.buttons = [0] * 15
        msg_joy.buttons[7] = 1
        self.joy_pub.publish(msg_joy)

    def end_following(self):
        msg_joy = Joy()
        msg_joy.axes = [0.0] * 8
        msg_joy.buttons = [0] * 15
        msg_joy.buttons[6] = 1
        self.joy_pub.publish(msg_joy)


    def at_callback(self, msg_sensor):
        for at_id in msg_sensor.data:
            if at_id == 32:
                try:
                    at_to_robo, _ = self.tf_listener.lookupTransform(f'/at_{at_id}_base_link', f'/april_tag_{at_id}', rospy.Time.now())
                    r = np.sqrt(at_to_robo[0] ** 2 + at_to_robo[1] ** 2)
                    print(f"the dist is : {r}")
                    if r < 0.3:
                        self.end_following()
                except (tf.LookupException): # Will occur if odom frame does not exist
                    continue
        
if __name__ == '__main__':
    # Initialize the node
    lane_following_node = LaneFollowingNode(node_name='lane_following_node')
    # Keep it spinning to keep the node alive
    rospy.spin()

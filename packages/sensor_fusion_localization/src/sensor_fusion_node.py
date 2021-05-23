#!/usr/bin/env python3

import numpy as np
from scipy.stats import norm
import rospy
import tf
import math

from duckietown_msgs.msg import Twist2DStamped, Pose2DStamped
from std_msgs.msg import Int32MultiArray

from geometry_msgs.msg import Quaternion, PoseStamped, PoseArray, Pose, PoseWithCovarianceStamped
# from nav_msgs.msg import OccupancyGrid

from duckietown.dtros import DTROS, NodeType, TopicType

SIGMA_R = 0.01
SIGMA_PHI = 0.001
MOTION_NOISE = 0.005
ANGLE_NOISE = 0.02
ANGLE_NOISE_2 = 0.02
N_VIZ_PARTICLES = 100

class SensorFusionNode(DTROS):
    """
    Much of this code block is lifted from the official Duckietown Github:
    https://github.com/duckietown/dt-car-interface/blob/daffy/packages/dagu_car/src/velocity_to_pose_node.py

    The goal of this node is to provide a state estimate using one of the two filtering methods we have covered in class: the Extended Kalman Filter
    and the Particle Filter. You will be fusing the estimates from a motion model with sensor readings from the cameras.
    We have left some code here from the official Duckietown repo, but you should feel free to discard it
    if you so choose to use a different approach.

    The motion model callback as listed here will provide you with motion estimates, but you will need to figure out the covariance matrix yourself.
    Additionally, for the PF, you will need to figure out how to sample (based on the covariance matrix),
    and for the EKF, you will need to figure out how to Linearize. Our expectation is that this is a group project, so we are not providing
    skeleton code for these parts.

    Likewise, you will need to implement your own sensor model and figure out how to manage the sensor readings. We have implemented a subscriber below
    that will fire the `sensor_fusion_callback` whenever a new sensor reading comes in. You will need to figure out how to unpack the sensor reading and
    what to do with them. To do this, you might use the [tf](https://docs.ros.org/en/melodic/api/tf/html/python/tf_python.html) package to get the transformation from the tf tree
    at the appropriate time. Just like in the motion model, you will need to consider the covariance matrix for the sensor model.

    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use

    Subscriber:
        ~velocity (:obj:`Twist2DStamped`): The robot velocity, typically obtained from forward kinematics

    Publisher:
        ~pose (:obj:`Pose2DStamped`): The integrated pose relative to the pose of the robot at node initialization

    """
    def __init__(self, node_name, fusion_type, n_particles):
        # Initialize the DTROS parent class
        super(SensorFusionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.LOCALIZATION
        )

        # Get the vehicle name
        self.veh_name = rospy.get_namespace().strip("/")
        self.FUSION_TYPE = "PF" 

        self.pose_sub = rospy.Subscriber(
            "/initialpose", 
            PoseWithCovarianceStamped, 
            self.clicked_pose_cb, 
            queue_size=1
        )

        self.pub_pose = rospy.Publisher(
            "/pose",
            PoseStamped,
            queue_size=1
        )
        self.particle_pub = rospy.Publisher(
            "/particles", 
            PoseArray, 
            queue_size=1
        ) 

        self.tf_listener = tf.TransformListener()

        self.log("Initialized.")
    

    def clicked_pose_cb(self, msg):
        pose = msg.pose.pose

        self.last_pose = Pose2DStamped()
        self.last_pose.x = pose.position.x
        self.last_pose.y = pose.position.y
        self.last_pose.theta = self.quaternion_to_angle(pose.orientation)
        self.last_theta_dot = 0
        self.last_v = 0

        self.mu_bar = np.array([self.last_pose.x, self.last_pose.y, self.last_pose.theta])

        self.NUM_PARTICLES = 1000 # n_particles # TODO: change to input argument
        self.sigma_r = SIGMA_R
        self.sigma_phi = SIGMA_PHI
        self.motion_noise = MOTION_NOISE
        self.angle_noise = ANGLE_NOISE
        self.angle_noise_2 = ANGLE_NOISE_2

        self.weights = np.ones(self.NUM_PARTICLES) / float(self.NUM_PARTICLES)
        self.particles = np.zeros((self.NUM_PARTICLES, 3))
        self.particles[:,0] = self.mu_bar[0] + np.random.normal(loc=0.0,scale=0.05,size=self.NUM_PARTICLES)
        self.particles[:,1] = self.mu_bar[1] + np.random.normal(loc=0.0,scale=0.05,size=self.NUM_PARTICLES)
        self.particles[:,2] = self.mu_bar[2] + np.random.normal(loc=0.0,scale=0.02,size=self.NUM_PARTICLES)
        
        # Setup the subscriber to the motion of the robot
        self.sub_velocity = rospy.Subscriber(
            f"/{self.veh_name}/kinematics_node/velocity",
            Twist2DStamped,
            self.motion_model_callback,
            queue_size=1
        )

        # Setup the subscriber for when sensor readings come in
        self.sub_sensor = rospy.Subscriber(
            f"/{self.veh_name}/detected_tags",
            Int32MultiArray,
            self.sensor_fusion_callback,
            queue_size=1
        )
    
    def quaternion_to_angle(self, q):
        x, y, z, w = q.x, q.y, q.z, q.w
        roll, pitch, yaw = tf.transformations.euler_from_quaternion((x, y, z, w))
        return yaw

    def angle_to_quaternion(self, angle):
        return Quaternion(*tf.transformations.quaternion_from_euler(0, 0, angle))

    def particle_to_pose(self, particle):
        pose = Pose()
        pose.position.x = particle[0]
        pose.position.y = particle[1]
        pose.orientation = self.angle_to_quaternion(particle[2])
        return pose

    def publish_particles(self, particles):
            pa = PoseArray()
            pa.header.stamp = rospy.Time.now()
            pa.header.frame_id = "map"
            pa.poses = [self.particle_to_pose(p) for p in particles]
            self.particle_pub.publish(pa)    

    def visualize_particles(self):
        if self.particle_pub.get_num_connections() > 0:
            if self.particles.shape[0] > N_VIZ_PARTICLES: 
                inds = np.random.choice(range(self.particles.shape[0]), N_VIZ_PARTICLES, p=self.weights)
                self.publish_particles(self.particles[inds,:])
            else:
                self.publish_particles(self.particles)

    def get_observed_features(self, msg_sensor):
        # 7.12 from book
        # m_t = []
        # for at_id in msg_sensor.data:
        #     trans, _ = self.tf_listener.lookupTransform('/map', f'/april_tag_{at_id}', rospy.Time(0))
        #     m_t.append(trans[:2])

        # z_t = [] 
        # for m_t_i in m_t:
        #     curr = np.zeros(2)
        #     x_temp = (m_t_i[0] - self.mu_bar[0])
        #     y_temp = (m_t_i[1] - self.mu_bar[1])
        #     curr[0] = np.sqrt(x_temp ** 2 + y_temp ** 2)
        #     curr[1] = np.arctan2(y_temp, x_temp) - self.mu_bar[2]
        #     z_t.append(curr)
        # return z_t, m_t

        # Self-design
        at_to_map_list =[] 
        at_to_robo_list = []
        for at_id in msg_sensor.data:
            try:
                at_to_map, _ = self.tf_listener.lookupTransform('/map', f'/april_tag_{at_id}', rospy.Time(0))
                at_to_robo, _ = self.tf_listener.lookupTransform(f'/at_{at_id}_base_link', f'/april_tag_{at_id}', rospy.Time.now())
                at_to_map_list.append(at_to_map)
                at_to_robo_list.append(at_to_robo)
            except (tf.LookupException): # Will occur if odom frame does not exist
                continue

        m_t = []
        z_t = []
        for at_to_map, at_to_robo in zip():
            m_t.append(at_to_map[:2])
            r = np.sqrt(at_to_robo[0] ** 2 + at_to_robo[1] ** 2)
            phi = np.arctan2(at_to_robo[1], at_to_robo[0])
            z_t.append(np.array([r, phi]))
        return z_t, m_t

    def sensor_fusion_callback(self, msg_sensor):
        """
        This function should handle the sensor fusion. It will fire whenever
        the robot observes an AprilTag
        """

        # The motion model callback as listed here will provide you with motion estimates, but you will need to 
        # figure out the covariance matrix yourself. Additionally, for the PF, you will need to figure out how to 
        # sample (based on the covariance matrix), and for the EKF, you will need to figure out how to Linearize. 
        # Our expectation is that this is a group project, so we are not providing skeleton code for these parts.

        # returns (x, y, s) for each landmark
        # m_t = self.get_measurement(msg_sensor)

        # returns (r_t, phi_t, s) (Eq. 7.12)
        z_t, m_t = self.get_observed_features(msg_sensor)

        # Particle_Filter in pdf page 118

        # Sample motion model
        avg = self.mu_bar
        # TODO maybe move to motion callback because this motion noise
        # noise = np.random.normal(loc=0.0, scale=self.motion_noise, size=self.particles.shape)
        cov = np.eye(3) * [self.motion_noise, self.motion_noise, self.angle_noise]
        cov2 = np.eye(3) * [self.motion_noise, self.motion_noise, self.angle_noise_2]
        mean = np.zeros(3)
        noise1 = np.random.multivariate_normal(mean, cov, size=int(self.particles.shape[0] * 0.8))
        noise2 = np.random.multivariate_normal(mean, cov2, size=int(self.particles.shape[0] * 0.2))
        noise = np.concatenate((noise1,noise2))
        particles = avg + noise

        # Measurement Model (book pdf page 278 in book)
        weights = self.measurement_model(z_t, m_t, self.particles)
        # print(weights)
        self.resample(particles, weights)

        mu_t = np.average(self.particles, axis=0, weights=self.weights)

        msg_pose = PoseStamped()
        msg_pose.header.frame_id = 'map'
        msg_pose.header.stamp = rospy.Time.now()
        msg_pose.pose.position.x = mu_t[0]
        msg_pose.pose.position.y = mu_t[1]
        msg_pose.pose.orientation = self.angle_to_quaternion(mu_t[2])
        self.pub_pose.publish(msg_pose)

        self.visualize_particles()
        
    def resample(self, particles, weights):
        step_array = (1.0/particles.shape[0]) * np.arange(particles.shape[0], dtype=np.float32)
        initval = np.random.uniform() * (1.0/particles.shape[0])
        vals    = initval + step_array
        cumwt   = np.cumsum(weights) 
        inds = np.searchsorted(cumwt, vals, side='left')
        
        self.particles[:] = particles[inds,:]
        self.weights[:] = 1.0 / particles.shape[0]

    def measurement_model(self, z_t, m_t, particles):
        weights = np.ones(len(particles))
        for i in range(len(z_t)):
            r_t_i, phi_t_i = z_t[i]
            m_t_i = m_t[i]

            x = (m_t_i[0] - particles[:, 0])
            y = (m_t_i[1] - particles[:, 1])
            r_hat = np.sqrt(x ** 2 + y ** 2)
            phi_hat = np.arctan2(y, x)

            weights *= norm.pdf(r_hat, loc=r_t_i, scale=self.sigma_r) \
                    * norm.pdf(phi_hat, loc=phi_t_i, scale=self.sigma_phi)
        weights /= np.sum(weights)
        return weights

    def motion_model_callback(self, msg_velocity):
        """
        This function will use robot velocity information to give a new state
        Performs the calclulation from velocity to pose and publishes a messsage with the result.
        Feel free to modify this however you wish. It's left more-or-less as-is
        from the official duckietown repo
        Args:
            msg_velocity (:obj:`Twist2DStamped`): the current velocity message
        """
        
        if self.last_pose.header.stamp.to_sec() > 0:
            dt = (msg_velocity.header.stamp - self.last_pose.header.stamp).to_sec()
            print(f"v: {msg_velocity.v}, omega: {msg_velocity.omega}")
            if dt > 0:  # skip first frame

                # Integrate the relative movement between the last pose and the current
                theta_delta = self.last_theta_dot * dt
                # to ensure no division by zero for radius calculation:
                if np.abs(self.last_theta_dot) < 0.000001:
                    # straight line
                    x_delta = self.last_v * dt
                    y_delta = 0
                else:
                    # arc of circle
                    radius = self.last_v / self.last_theta_dot
                    x_delta = radius * np.sin(theta_delta)
                    y_delta = radius * (1.0 - np.cos(theta_delta))

                self.particles[:,-1] += theta_delta
                self.particles[self.particles[:, -1] < -1 * math.pi, 2] += 2 * math.pi  
                self.particles[self.particles[:, -1] > math.pi, 2] -= 2 * math.pi
                cos_theta = np.cos(self.particles[:, -1])
                sin_theta = np.sin(self.particles[:, -1])
                self.particles[:, 0] += x_delta * cos_theta - y_delta * sin_theta
                self.particles[:, 1] += y_delta * cos_theta + x_delta * sin_theta

                self.mu_bar = np.average(self.particles, axis=0, weights=self.weights)

                msg_pose = PoseStamped()
                msg_pose.header = msg_velocity.header
                msg_pose.header.frame_id = 'map'#self.veh_name
                msg_pose.header.stamp = rospy.Time.now() # would the msg_velocity.header not set this?
                msg_pose.pose.position.x = self.mu_bar[0]
                msg_pose.pose.position.y = self.mu_bar[1]
                msg_pose.pose.orientation = self.angle_to_quaternion(self.mu_bar[2])
                self.pub_pose.publish(msg_pose)
                self.visualize_particles()
                
                self.last_pose.header.stamp = msg_velocity.header.stamp
                self.last_theta_dot = msg_velocity.omega
                self.last_v = msg_velocity.v
        else:
            self.last_pose.header.stamp = msg_velocity.header.stamp
            self.last_theta_dot = msg_velocity.omega
            self.last_v = msg_velocity.v
        
if __name__ == '__main__':
    # Initialize the node
    fusion_type = rospy.get_param("~fusion_type")
    n_particles = 1000 # int(rospy.get_param("~n_particles"))
    sensor_fusion_node = SensorFusionNode(node_name='sensor_fusion_node', fusion_type=fusion_type, n_particles=n_particles)
    # Keep it spinning to keep the node alive
    rospy.spin()

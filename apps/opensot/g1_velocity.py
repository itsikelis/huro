#!/usr/bin/env python3
import subprocess
from tf2_ros import TransformBroadcaster
from xbot2_interface import pyxbot2_interface as xbi
from xbot2_interface import pyxbot2_collision
from xbot2_interface import pyaffine3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from rcl_interfaces.srv import GetParameters
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, WrenchStamped, PoseStamped
from visualization_msgs.msg import InteractiveMarkerControl, InteractiveMarker, Marker
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from ament_index_python.packages import get_package_share_directory
from interactive_markers.menu_handler import MenuHandler

import time
import numpy as np
import array
from scipy.spatial.transform import Rotation as R
import tf_transformations

import pyopensot as pysot
from pyopensot.tasks.velocity import Postural, Cartesian, CoM
from pyopensot.constraints.velocity import JointLimits, VelocityLimits
from pyopensot_collision.constraints.velocity import CollisionAvoidance
from pyopensot_collision.constraints.velocity import CollisionAvoidance
from pyopensot.constraints.force import FrictionCone
from pyopensot.variables import Torque
from pyopensot.tasks import MinimizeVariable

from unitree_hg.msg import LowCmd, LowState, IMUState, MotorState
from huro_py.crc_hg import Crc

G1_NUM_MOTOR = 29

Kp = [
    600.0,  # hips
    700.0,
    500.0,
    1000.0,  # knee
    900.0,
    500.0,  # legs
    600.0,  # hips
    700.0,
    500.0,
    1000.0,  # knee
    900.0,
    500.0,  # legs
    400.0,
    400.0,
    400.0,  # waist
    100.0,
    100.0,
    50.0,
    50.0,
    20.0,
    20.0,
    20.0,  # arms
    100.0,
    100.0,
    50.0,
    50.0,
    20.0,
    20.0,
    20.0,  # arms
]

Kd = [
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,  # legs
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,  # legs
    10.0,
    10.0,
    10.0,  # waist
    2.0,
    2.0,
    2.0,
    2.0,
    1.0,
    1.0,
    1.0,  # arms
    2.0,
    2.0,
    2.0,
    2.0,
    1.0,
    1.0,
    1.0,  # arms
]

q_init = [
    -0.1,
    0.0,
    0.0,  # hips
    0.432,  # knee
    -0.317,
    0.0,  # ankles
    -0.1,
    0.0,
    0.0,  # hips
    0.432,  # knee
    -0.317,
    0.0,  # ankles
    0.0,
    0.0,
    0.0,  # waist
    0.3,
    0.25,
    0.0,
    1.0,
    0.15,
    0.0,
    0.0,  # arm
    0.3,
    -0.25,
    0.0,
    1.0,
    0.15,
    0.0,
    0.0,
]  # arm

class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints

class G1CollisionAvoidanceNode(Node):
    def __init__(self):
        super().__init__('g1_collision_avoidance_node')
        self.get_logger().info("Starting G1 Collision Avoidance Node")
        self.client = self.create_client(GetParameters, '/robot_state_publisher/get_parameters')

        self.control_dt = 0.005
        self.timer_dt_ms = int(self.control_dt * 1000)
        self.time = 0.0
        self.t = 0.0
        self.init_duration_s = 3.0

        self.mode = Mode.PR
        self.mode_machine = 0
        self.motors_on = 1

        self.right_hand_pose_ref = None
        self.left_hand_pose_ref = None

        self.imu = IMUState()
        self.motor = [MotorState() for _ in range(G1_NUM_MOTOR)]

        self.topic_name = (
            "lowstate" if self.get_parameter_or("HIGH_FREQ", False) else "lf/lowstate"
        )

        self.lowcmd_pub = self.create_publisher(LowCmd, "/lowcmd", 10)
        self.lowstate_sub = self.create_subscription(
            LowState, self.topic_name, self.low_state_handler, 10
        )

        self.start_opensot_sub = self.create_subscription(
            Bool, "/start_opensot", self.start_opensot_callback, 10
        )

        self.emergency_stop_sub = self.create_subscription(
            Bool, "/emergency_stop", self.emergency_stop_callback, 10
        )

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Service /robot_state_publisher/get_parameters not available, waiting...')

        request = GetParameters.Request()
        request.names = ['robot_description']

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        self.urdf = None
        if future.result() is not None:
            values = future.result().values
            for val in values:
                self.urdf = val.string_value
        else:
            self.get_logger().error('Failed to get robot_description from parameter server')

        self.joint_state_publisher = self.create_publisher(JointState, '/joint_states', 10)
        self.base_link_broadcaster = TransformBroadcaster(self)

        self.interactive_marker_server = InteractiveMarkerServer(self, 'teleoperation_markers')
        self.marker_pose = PoseStamped()

        self.force_publishers = {}

        self.initialize()
        self.initialize_imarkers()

        self.control_timer = self.create_timer(self.control_dt, self.control_loop)

    def initialize_imarkers(self):
        self.marker_enabled = {}
        self.menu_handler = {}  
        base_ref, _ = self.base.getReference()
        com_ref, _ = self.com.getReference()
        pose_ref  = pyaffine3.Affine3()
        pose_ref.translation = com_ref
        pose_ref.linear = base_ref.linear.copy()
        self.get_logger().info(f"Initial base pose:\n{pose_ref}")
        self.make_6dof_marker('base_marker', pose_ref, 'world')

        right_hand_ref = self.right_gripper.getReference()
        self.get_logger().info(f"Initial right hand pose:\n{right_hand_ref}")
        self.make_6dof_marker('right_hand_marker', right_hand_ref[0], 'pelvis')

        left_hand_ref = self.left_gripper.getReference()
        self.get_logger().info(f"Initial left hand pose:\n{left_hand_ref}")
        self.make_6dof_marker('left_hand_marker', left_hand_ref[0], 'pelvis')

    def control_loop(self):
        if not hasattr(self, 'start_opensot'):
            self.start_opensot = False

        low_cmd = LowCmd()
        self.time += self.control_dt

        low_cmd.mode_pr = self.mode
        low_cmd.mode_machine = self.mode_machine

        if self.time < self.init_duration_s:
            for i in range(G1_NUM_MOTOR):
                cmd = low_cmd.motor_cmd[i]
                cmd.mode = self.motors_on
                cmd.q = q_init[i]
                cmd.dq = 0.0
                cmd.tau = 0.0
                cmd.kp = Kp[i]
                cmd.kd = Kd[i]

        elif True: # elif self.start_opensot and self.motors_on or True:
            self.model.setJointPosition(self.q)
            self.model.setJointVelocity(self.dq)
            self.model.update()

            base_ref0, vel_ref0 = self.base.getReference()
            self.base_ref = base_ref0.copy()
            self.base_vel_ref = vel_ref0

            self.base_ref.translation[:] = [
                self.marker_pose.pose.position.x,
                self.marker_pose.pose.position.y,
                self.marker_pose.pose.position.z,
            ]

            quaternion = [
                self.marker_pose.pose.orientation.x,
                self.marker_pose.pose.orientation.y,
                self.marker_pose.pose.orientation.z,
                self.marker_pose.pose.orientation.w,
            ]

            # self.base_ref.translation[:] = p_fixed
            self.base_ref.linear = R.from_quat(quaternion).as_matrix()
            self.base.setReference(self.base_ref, self.base_vel_ref)
            
            com_ref, _ = self.com.getReference()

            com_ref = np.array([
                self.marker_pose.pose.position.x,
                self.marker_pose.pose.position.y,
                self.marker_pose.pose.position.z,
            ])

            self.com.setReference(com_ref)

            self.stack.update()


            dq = self.solver.solve()
            self.q = self.model.sum(self.q, dq * self.control_dt)
            self.dq = dq

            msg = JointState()
            msg.name = self.model.getJointNames()[1::]
            msg.position = self.q[7: ]
            msg.header.stamp = self.get_clock().now().to_msg()

            w_t_b = TransformStamped()
            w_t_b.header.frame_id = "world"
            w_t_b.child_frame_id = "pelvis"
            w_t_b.header.stamp = msg.header.stamp
            w_t_b.transform.translation.x = self.q[0]
            w_t_b.transform.translation.y = self.q[1]
            w_t_b.transform.translation.z = self.q[2]
            w_t_b.transform.rotation.x = self.q[3]
            w_t_b.transform.rotation.y = self.q[4]
            w_t_b.transform.rotation.z = self.q[5]
            w_t_b.transform.rotation.w = self.q[6]

            # for contact_frame in self.contact_frames:
            #     T = self.model.getPose(contact_frame)
            #     self.forces_msgs[contact_frame].header.stamp = msg.header.stamp
            #     f_local = T.linear.transpose() @ self.variables.getVariable(contact_frame).getValue(dq)
            #     self.forces_msgs[contact_frame].wrench.force.x = f_local[0]
            #     self.forces_msgs[contact_frame].wrench.force.y = f_local[1]
            #     self.forces_msgs[contact_frame].wrench.force.z = f_local[2]

            if self.forces_msgs is not None:
                for contact_frame, force_msg in self.forces_msgs.items():
                    self.force_publishers[contact_frame].publish(force_msg)

            for i in range(G1_NUM_MOTOR):
                cmd = low_cmd.motor_cmd[i]
                cmd.mode = self.motors_on
                cmd.q = self.q[i + 7]
                cmd.dq = self.dq[i + 6]
                cmd.tau = 0.0
                cmd.kp = Kp[i]
                cmd.kd = Kd[i]

        else:
            for i in range(G1_NUM_MOTOR):
                cmd = low_cmd.motor_cmd[i]
                cmd.mode = self.motors_on
                cmd.q = q_init[i]
                cmd.dq = 0.0
                cmd.tau = 0.0
                cmd.kp = Kp[i]
                cmd.kd = Kd[i]

        low_cmd.crc = Crc(low_cmd)
        self.lowcmd_pub.publish(low_cmd)

    def initialize_force_publishers(self, contact_frames):
        for contact_frame in contact_frames:
            self.force_publishers[contact_frame] = self.create_publisher(
                WrenchStamped, "wrench_" + contact_frame, 10
            )

    def initialize(self, manipulation_frame = 'world'):
        self.get_logger().warning("Initializing XBot2 Model Interface")
        self.contact_frames = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ]

        self.model = xbi.ModelInterface2(self.urdf)
        self.q = np.zeros(self.model.nq)
        self.q[2] = 0.6756
        self.q[6] = 1.0
        self.q[7: ] = q_init[0:].copy()
        self.dq = np.zeros(self.model.nv)
        self.model.setJointPosition(self.q)
        self.model.setJointVelocity(self.dq)
        self.model.update()

        variables_vec = dict()
        variables_vec['qddot'] = self.model.nv
        for contact_frame in self.contact_frames:
            variables_vec[contact_frame] = 3
        self.variables = pysot.OptvarHelper(variables_vec)

        self.com = CoM(self.model)

        w_T_b = TransformStamped()
        w_T_b.header.frame_id = "world"
        w_T_b.child_frame_id = "pelvis"

        self.forces_msgs = dict()
        for contact_frame in self.contact_frames:
            self.forces_msgs[contact_frame] = WrenchStamped()
            self.forces_msgs[contact_frame].header.frame_id = contact_frame
            self.forces_msgs[contact_frame].wrench.force.x = self.forces_msgs[
                contact_frame
            ].wrench.force.y = self.forces_msgs[contact_frame].wrench.force.z = 0.0

        self.contacts = []
        for cf in self.contact_frames:
            t = Cartesian(
                f"{cf}_task", 
                self.model, 
                cf,
                "world", 
            )
            t.setLambda(1.0)
            self.contacts.append(t)

        self.base = Cartesian(
            "base_task",
            self.model,
            "pelvis",
            manipulation_frame
        )
        self.base.setLambda(1.0) 

        self.torso = Cartesian(
            "base_task",
            self.model,
            "torso_link",
            manipulation_frame
        )
        self.torso.setLambda(1.0) 

        self.right_gripper = Cartesian(
            "right_gripper_task",
            self.model,
            "right_hand_point_contact",
            'pelvis',
        )
        self.right_gripper.setLambda(1.0)

        self.left_gripper = Cartesian(
            "left_gripper_task",
            self.model,
            "left_hand_point_contact",
            'pelvis',
        )
        self.left_gripper.setLambda(1.0)

        self.postural = Postural(self.model)
        self.postural.setLambda(1.0)
        self.W_postural = self.postural.getWeight()
        self.W_postural[0: 6] = 0.0
        self.W_postural[6: 10] = 0.0
        self.postural.setWeight(self.W_postural)

        # CONSTRAINTS

        self.qmin, self.qmax = self.model.getJointLimits()
        self.qlims = JointLimits(self.model, self.qmax, self.qmin)
        self.dqmax = self.model.getVelocityLimits()
        self.dqlims = VelocityLimits(self.model, self.dqmax, self.control_dt)

        # WE ADD THE COLLISION AVOIDANCE CONSTRAINT

        stack_contacts = self.contacts[0]
        for t in self.contacts[1:]:
            stack_contacts = stack_contacts / t

        self.stack = (
            stack_contacts
            / (self.com + self.base%[3,4,5] + self.torso%[3,4,5]) 
            / self.postural
            << (self.right_gripper + self.left_gripper)
            << self.qlims
            << self.dqlims
        )
        self.stack.update()

        self.solver = pysot.iHQP(self.stack)
        self.initialize_force_publishers(self.contact_frames)

    def start_opensot_callback(self, msg: Bool):
        if msg.data:
            self.start_opensot = True
        else:
            self.start_opensot = False

    def emergency_stop_callback(self, msg: Bool):
        if msg.data:
            self.motors_on = 0

    def low_state_handler(self, msg: LowState):
        # self.get_logger().info(str(self.motors_on))
        self.mode_machine = msg.mode_machine
        self.imu = msg.imu_state
        for i in range(G1_NUM_MOTOR):
            self.motor[i] = msg.motor_state[i]

        # ## Handle Controller Message
        # self.controller_msg = msg.wireless_remote
        # if self.controller_msg[3] == 1:
        #     self.motors_on = 0
        ## Handle Controller Message
        self.controller_msg = msg.wireless_remote

        # Start OpenSoT when L1 is pressed
        if self.controller_msg[2] == 2:
            self.start_opensot = True

        # E-Stop if A is pressed
        if self.controller_msg[3] == 1:
            self.motors_on = 0
    
    def make_6dof_marker(self, name, pose, frame_id):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = frame_id
        int_marker.name = name
        int_marker.description = '6-DOF Control'
        int_marker.scale = 0.3

        int_marker.pose.position.x = pose.translation[0]
        int_marker.pose.position.y = pose.translation[1]
        int_marker.pose.position.z = pose.translation[2]

        quat_xyzw = R.from_matrix(pose.linear).as_quat() # Format: [x, y, z, w]
        int_marker.pose.orientation.x = quat_xyzw[0]
        int_marker.pose.orientation.y = quat_xyzw[1]
        int_marker.pose.orientation.z = quat_xyzw[2]
        int_marker.pose.orientation.w = quat_xyzw[3]

        self.marker_pose.pose = int_marker.pose

        # Add a visible marker (e.g., a cube)
        cube_marker = Marker()
        cube_marker.type = Marker.CUBE
        cube_marker.scale.x = 0.05
        cube_marker.scale.y = 0.05
        cube_marker.scale.z = 0.05
        cube_marker.color.r = 0.0
        cube_marker.color.g = 1.0
        cube_marker.color.b = 0.0
        cube_marker.color.a = 1.0

        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(cube_marker)
        int_marker.controls.append(control)

        # Add 6-DOF controls
        self.add_6dof_controls(int_marker)

        self.marker_enabled[name] = True

        menu = MenuHandler()
        h_enable = menu.insert("Enable", callback=self.process_menu)
        h_reset = menu.insert("Reset", callback=self.process_menu)
        self.menu_handler[name] = menu

        self.menu_entry_ids = getattr(self, "menu_entry_ids", {})
        self.menu_entry_ids[name] = {"enable": h_enable, "reset": h_reset}

        menu_control = InteractiveMarkerControl()
        menu_control.interaction_mode = InteractiveMarkerControl.MENU
        menu_control.name = "menu"
        int_marker.controls.append(menu_control)

        self.interactive_marker_server.insert(marker=int_marker, feedback_callback=self.process_feedback)
        menu.apply(self.interactive_marker_server, name)
        self.interactive_marker_server.applyChanges()

    def process_feedback(self, feedback):
        if hasattr(feedback, "event_type"):
            pass

        name = feedback.marker_name
        if not self.marker_enabled.get(name, True):
            return 

        self.marker_pose.header = feedback.header
        self.marker_pose.pose = feedback.pose


    def process_menu(self, feedback):
        name = feedback.marker_name
        ids = self.menu_entry_ids.get(name, {})
        if feedback.menu_entry_id == ids.get("enable"):
            self.marker_enabled[name] = True
        elif feedback.menu_entry_id == ids.get("reset"):
            self.marker_enabled[name] = False
            pass


    def add_6dof_controls(self, marker):
        axes = ['x', 'y', 'z']
        for axis in axes:
            # Rotation
            control = InteractiveMarkerControl()
            control.name = f'rotate_{axis}'
            control.orientation.w = 1.0
            setattr(control.orientation, axis, 1.0)
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            marker.controls.append(control)

            # Translation
            control = InteractiveMarkerControl()
            control.name = f'move_{axis}'
            control.orientation.w = 1.0
            setattr(control.orientation, axis, 1.0)
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            marker.controls.append(control)



def main(args=None):
    rclpy.init(args=args)
    node = G1CollisionAvoidanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
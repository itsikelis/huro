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

import time
import numpy as np
import array
from scipy.spatial.transform import Rotation as R
import tf_transformations

import pyopensot as pysot
from pyopensot.tasks.velocity import Postural, Cartesian
from pyopensot.constraints.velocity import JointLimits, VelocityLimits
from pyopensot_collision.constraints.velocity import CollisionAvoidance
from pyopensot.tasks.acceleration import Cartesian, CoM, DynamicFeasibility, Postural
from pyopensot.constraints.acceleration import JointLimits, VelocityLimits, TorqueLimits
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

        self.urdg = None
        if future.result() is not None:
            values = future.result().values
            for value in values:
                self.urdf = value.string_value
            self.get_logger().info("URDF successfully retrieved")
        else:
            self.get_logger().error("Failed to retrieve URDF")

        self.imarker_server = InteractiveMarkerServer(self, 'teleoperation_marker')
        self.marker_pose = PoseStamped()

        self.collision_distances_publisher = self.create_publisher(Marker, 'collision_distances', 10)

        self.force_publishers = {}

        ##### 
        self.model = xbi.ModelInterface2(self.urdf)
        qmin, qmax = self.model.getJointLimits()
        dqmax = self.model.getVelocityLimits()
        torque_limits = self.model.getEffortLimits()

        self.q = np.zeros(self.model.nq)
        self.q[2] = 0.6756
        self.q[3] = 1.0
        self.q[7: ] = q_init[0:].copy()
        self.dq = np.zeros(self.model.nv)
        self.model.setJointPosition(self.q)
        self.model.setJointVelocity(self.dq)
        self.model.update()

        self.contact_frames = [
            "left_foot_lower_left",
            "left_foot_lower_right",
            "left_foot_upper_left",
            "left_foot_upper_right",
            "right_foot_lower_left",
            "right_foot_lower_right",
            "right_foot_upper_left",
            "right_foot_upper_right",
        ]
        variables_vec = dict()
        variables_vec['qddot'] = self.model.nv
        for contact_frame in self.contact_frames:
            variables_vec[contact_frame] = 3
        self.variables = pysot.OptvarHelper(variables_vec)

        self.com = CoM(self.model, self.variables.getVariable('qddot'))
        self.com.setLambda(1.2)
        self.com_ref, vel_ref, acc_ref = self.com.getReference()
        self.com0 = self.com_ref.copy()

        base = Cartesian(
            'base',
            self.model,
            'world',
            'pelvis',
            self.variables.getVariable('qddot'),
        )
        base.setLambda(1.2)

        #####
        contact_task = list()
        for contact_frame in self.contact_frames:
            contact_task.append(
                Cartesian(
                    contact_frame + '_kin',
                    self.model,
                    'world',
                    contact_frame,
                    self.variables.getVariable('qddot'),
                )
            )
        self.posture = Postural(
            self.model,
            self.variables.getVariable('qddot'),
        )
        # idx = [18 + i for i in range(self.model.nv - 18)]

        self.stack = 1.0 * self.com + 0.0001 * self.posture

        #####
        force_variables = list()
        for force in range(len(self.contact_frames)):
            self.stack = self.stack + 5.0 * (contact_task[force] % [0, 1, 2])
            force_variables.append(self.variables.getVariable(self.contact_frames[force]))

        self.torques = Torque(
            self.model,
            self.variables.getVariable('qddot'),
            self.contact_frames,
            force_variables,
        )

        self.stack = self.stack + 1e-8 * MinimizeVariable('min_torque', self.torques)
        self.stack = self.stack + 1e-3 * MinimizeVariable('min_qddot', self.variables.getVariable('qddot'))

        #####
        self.right_hand = Cartesian(
            'right_hand_point_contact',
            self.model,
            'right_hand_point_contact',
            'world',
            self.variables.getVariable('qddot'),
        )

        self.left_hand = Cartesian(
            'left_hand_point_contact',
            self.model,
            'left_hand_point_contact',
            'world',
            self.variables.getVariable('qddot'),
        )

        self.stack = self.stack + 0.01 * (self.right_hand + self.left_hand)

        #####
        self.stack = pysot.AutoStack(self.stack) << DynamicFeasibility(
            'floating_base_dynamic',
            self.model,
            self.variables.getVariable('qddot'),
            force_variables,
            self.contact_frames,
        )
        self.stack = self.stack << JointLimits(
            self.model,
            self.variables.getVariable('qddot'),
            qmax,
            qmin,
            10.0 * dqmax,
            self.control_dt,
        )
        self.stack = self.stack << VelocityLimits(
            self.model,
            self.variables.getVariable('qddot'),
            dqmax,
            self.control_dt,
        )
        self.stack = self.stack << TorqueLimits(
            self.model,
            self.variables.getVariable('qddot'),
            force_variables,
            self.contact_frames,
            torque_limits,
        )

        for i in range(len(self.contact_frames)):
            T = self.model.getPose(self.contact_frames[i])
            mu = (T.linear, 0.8)
            self.stack = self.stack << FrictionCone(
                self.contact_frames[i],
                self.variables.getVariable(self.contact_frames[i]),
                self.model,
                mu,
            )

        self.solver = pysot.iHQP(self.stack)

        self.initialize_force_publishers(self.contact_frames)

        msg = JointState()
        msg.name = self.model.getJointNames()[1::]

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

        self.alpha = 0.2
        self.a = 0.0

        self.first = True
        self.start_opensot = False

        self.timer = self.create_timer(self.control_dt, self.control_loop)

    def low_state_handler(self, msg: LowState):
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

    def start_opensot_callback(self, msg: Bool):
        if msg.data:
            self.start_opensot = True
        else:
            self.start_opensot = False

    def emergency_stop_callback(self, msg: Bool):
        if msg.data:
            self.motors_on = 0

    def setup_tasks_and_constraints(self):
        pass

    def initialize_force_publishers(self, contact_frames):
        for contact_frame in contact_frames:
            self.force_publishers[contact_frame] = self.create_publisher(
                WrenchStamped, "wrench_" + contact_frame, 10
            )

    def create_6dof_marker(self, name, pose, frame_id):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = frame_id
        int_marker.name = name
        int_marker.description = "6-DOF Control"
        int_marker.scale = 0.3

        int_marker.pose.position.x = pose.translation[0]
        int_marker.pose.position.y = pose.translation[1]
        int_marker.pose.position.z = pose.translation[2]

        quat_xyzw = R.from_matrix(pose.linear).as_quat()
        int_marker.pose.orientation.x = quat_xyzw[0]
        int_marker.pose.orientation.y = quat_xyzw[1]
        int_marker.pose.orientation.z = quat_xyzw[2]
        int_marker.pose.orientation.w = quat_xyzw[3]

        self.marker_pose.pose = int_marker.pose

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

        self.add_6dof_controls(int_marker)
        self.imarker_server.insert(marker = int_marker, feedback_callback = self.marker_feedback)
        self.imarker_server.applyChanges()

    def process_feedback(self, feedback):
        self.marker_pose.header = feedback.header
        self.marker_pose.pose = feedback.pose

    def add_6dof_controls(self, marker):
        axes = ['x', 'y', 'z']
        for axis in axes:
            control = InteractiveMarkerControl()
            control.name = f'rotate_{axis}'
            control.orientation.w = 1.0
            setattr(control.orientation, axis, 1.0)
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.name = f'move_{axis}'
            control.orientation.w = 1.0
            setattr(control.orientation, axis, 1.0)
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            marker.controls.append(control)

    def clamp(self, value, low, high):
        if value < low:
            return low
        if value > high:
            return high
        return value
    
    def get_inverse_dynamics(self, x, ddq):
        # Update joint position
        self.model.setJointAcceleration(ddq)
        tau = self.model.computeInverseDynamics()
        for i in range(len(self.contact_frames)):
            Jc = self.model.getJacobian(self.contact_frames[i])
            contact_force = self.variables.getVariable(self.contact_frames[i]).getValue(
                x
            )

            tau = tau - Jc[:3, :].T @ np.array(contact_force)
        return tau

    def control_loop(self):
        low_cmd = LowCmd()
        self.time += self.control_dt

        low_cmd.mode_pr = self.mode
        low_cmd.mode_machine = self.mode_machine

        if self.time < self.init_duration_s:
            for i in range(G1_NUM_MOTOR):
                ratio = self.clamp(self.time / self.init_duration_s, 0.0, 1.0)
                cmd = low_cmd.motor_cmd[i]
                cmd.mode = self.motors_on
                # cmd.q = (1.0 - ratio) * self.motor[i].q + ratio * q_init[i]
                cmd.q = q_init[i]
                cmd.dq = 0.0
                cmd.tau = 0.0
                cmd.kp = Kp[i]
                cmd.kd = Kd[i]

        elif self.start_opensot:
            self.t =+ self.control_dt
            dt = self.control_dt
            self.model.setJointPosition(self.q)
            self.model.setJointVelocity(self.dq)
            self.model.update()

            self.com_ref[2] = self.com0[2]
            
            self.com.setReference(self.com_ref)

            if self.right_hand_pose_ref is not None:
                right_hand_pose_affine = self.right_hand.getReference()
                right_hand_pose_affine[0].translation = (
                    self.right_hand_pose_ref.pose.position.x,
                    self.right_hand_pose_ref.pose.position.y,
                    self.right_hand_pose_ref.pose.position.z,
                )
                self.get_logger().info(f"Right Hand Ref: {right_hand_pose_affine[0].translation}")
                
                R = tf_transformations.quaternion_matrix(
                    [
                        self.right_hand_pose_ref.pose.orientation.x,
                        self.right_hand_pose_ref.pose.orientation.y,
                        self.right_hand_pose_ref.pose.orientation.z,
                        self.right_hand_pose_ref.pose.orientation.w,
                    ]
                )[:3, :3]

                right_hand_pose_affine[0].linear = R

                self.right_hand.setReference(
                    right_hand_pose_affine[0], np.zeros(6), np.zeros(6)
                )
            
            if self.left_hand_pose_ref is not None:
                pass

            self.stack.update()

            x = self.solver.solve()
            ddq = self.variables.getVariable('qddot').getValue(x)
            self.q = self.model.sum(
                self.q, self.dq * dt + 0.5 * ddq * dt * dt
            )

            msg = JointState()
            msg.name = self.model.getJointNames()[1::]
            msg.position = self.q[7:]
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

            self.dq += ddq * dt

            for contact_frame in self.contact_frames:
                T = self.model.getPose(contact_frame)
                self.forces_msgs[contact_frame].header.stamp = msg.header.stamp
                f_local = T.linear.transpose() @ self.variables.getVariable(contact_frame).getValue(x)
                self.forces_msgs[contact_frame].wrench.force.x = f_local[0]
                self.forces_msgs[contact_frame].wrench.force.y = f_local[1]
                self.forces_msgs[contact_frame].wrench.force.z = f_local[2]

            if self.forces_msgs is not None:
                for contact_frame, force_msg in self.forces_msgs.items():
                    self.force_publishers[contact_frame].publish(force_msg)
            
            tau = self.get_inverse_dynamics(x, ddq)

            for i in range(G1_NUM_MOTOR):
                cmd = low_cmd.motor_cmd[i]
                cmd.mode = self.motors_on
                cmd.q = self.q[i + 7]
                cmd.dq = self.dq[i + 6]
                cmd.tau = tau[i + 6]
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



def main(args=None):
    rclpy.init(args=args)
    node = G1CollisionAvoidanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
import math

from unitree_hg.msg import LowCmd, LowState, IMUState, MotorState

from huro_py.crc_hg import Crc

from rcl_interfaces.srv import GetParameters
from ament_index_python.packages import get_package_share_directory
from xbot2_interface import pyxbot2_interface as xbi
from pyopensot.tasks.acceleration import Cartesian, CoM, DynamicFeasibility, Postural
from pyopensot.constraints.acceleration import JointLimits, VelocityLimits, TorqueLimits
from pyopensot.constraints.force import FrictionCone
from pyopensot.variables import Torque
from pyopensot.tasks import MinimizeVariable
import pyopensot as pysot
import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, WrenchStamped, PoseStamped
from tf2_ros import TransformBroadcaster
import tf_transformations
import subprocess
import time

from std_msgs.msg import Bool

# Custom PRorAB enum or constant
# from hucebot_g1_ros.msg import MotorMode  # Assuming PRorAB is defined here

# from hucebot_g1_ros.config import G1_NUM_MOTOR, Kp, Kd
# from hucebot_g1_ros.motor_crc_hg import get_crc  # Assuming Python version available

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

# q_init = [0.0 for _ in range(G1_NUM_MOTOR)]
# q_init[0] = -0.6
# q_init[3] = 1.2
# q_init[4] = -0.6
# q_init[6] = -0.6
# q_init[9] = 1.2
# q_init[10] = -0.6

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



# q_init = [-3.42817396e-01, 2.31913421e-02,  9.70253372e-04, #hips
#            4.31958795e-01, #knee
#           -3.17440391e-01, -7.41453618e-02, # ankles
#           -2.02988297e-01, -3.61669660e-02,  2.02979296e-02, #hips
#            4.15024102e-01, #knee
#           -1.28817156e-01, -1.57903448e-01, #ankles
#            1.16415322e-10, -1.45519152e-11, -2.91038305e-11, # waist
#            3.00000012e-01,  2.50000000e-01, -9.31322575e-10, 9.70000029e-01,  1.50000006e-01,  3.72529030e-09, -9.31322575e-10, # arm
#            3.00000012e-01, -2.50000000e-01,  1.86264515e-09, 9.70000029e-01, -1.50000006e-01,  3.72529030e-09,  1.86264515e-09] # arm

# q_init = [-0.1, 0.,  0., #hips
#            0.432, #knee
#           -0.317, 0., # ankles
#           -0.1, 0.,  0., #hips
#            0.432, #knee
#           -0.317, 0., #ankles
#            0., 0., 0., # waist
#            0.3,  0.25, 0., 1.,  0.15,  0., 0., # arm
#            0.3, -0.25,  0., 1., 0.15,  0.,  0.] # arm


class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints


class MoveExample(Node):
    def __init__(self):
        super().__init__("huromove_py_example")

        self.control_dt = 0.005  # 10ms
        self.timer_dt_ms = int(self.control_dt * 1000)
        self.time = 0.0
        self.t = 0.0
        self.init_duration_s = 3.0

        self.mode_ = Mode.PR
        self.mode_machine = 0

        self.motors_on = 1

        self.imu = IMUState()
        self.motor = [MotorState() for _ in range(G1_NUM_MOTOR)]

        # Rosbag
        # self.bag_motor = [MotorState() for _ in range(G1_NUM_MOTOR)]

        self.topic_name = (
            "lowstate" if self.get_parameter_or("HIGH_FREQ", False) else "lf/lowstate"
        )

        self.lowcmd_pub = self.create_publisher(LowCmd, "/lowcmd", 10)
        self.lowstate_sub = self.create_subscription(
            LowState, self.topic_name, self.low_state_handler, 10
        )

        # Teleop sub
        self.teleop_sub = self.create_subscription(
            PoseStamped, "right_hand_goal", self.teleop_callback, 10
        )

         ###### SUBSSCRIBERS

        self.start_opensot_sub = self.create_subscription(
            Bool, "/start_opensot", self.start_opensot_callback, 10
        )

        self.emergency_stop_sub = self.create_subscription(
            Bool, "/emergency_stop", self.emergency_stop_callback, 10
        )

        self.right_hand_pose_ref = None

        # For bag playback
        # self.bag_lowstate_sub = self.create_subscription(LowCmd, "/bag/lowcmd", self.bag_lowcmd_callback, 10)

        self.timer = self.create_timer(self.control_dt, self.control)

        ###################################

        self.get_logger().info("whole_body_g1 node has been started.")
        self.client = self.create_client(
            GetParameters, "/robot_state_publisher/get_parameters"
        )

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for parameter service...")

        request = GetParameters.Request()
        request.names = ["robot_description"]

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        self.urdf = None
        if future.result() is not None:
            values = future.result().values
            for val in values:
                self.urdf = val.string_value
        else:
            self.get_logger().error("Failed to call service")

        # self.joint_state_publisher = self.create_publisher(
        #     JointState, "joint_states", 10
        # )

        self.force_publishers = {}

        # self.base_link_broadcaster = TransformBroadcaster(self)

        ###################
        self.model = xbi.ModelInterface2(self.urdf)
        qmin, qmax = self.model.getJointLimits()
        dqmax = self.model.getVelocityLimits()
        torque_limits = self.model.getEffortLimits()

        self.q = np.zeros(self.model.nq)
        self.q[2] = 0.6756
        self.q[6] = 1
        self.q[7:] = q_init[0:].copy()
        self.dq = np.zeros(self.model.nv)
        self.model.setJointPosition(self.q)
        self.model.setJointVelocity(self.dq)
        self.model.update()

        # Instantiate Variables: qddot and contact forces (3 per contact)
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
        variables_vec["qddot"] = self.model.nv
        for contact_frame in self.contact_frames:
            variables_vec[contact_frame] = 3
        self.variables = pysot.OptvarHelper(variables_vec)

        # Creates tasks cand constraints
        self.com = CoM(self.model, self.variables.getVariable("qddot"))
        self.com.setLambda(1.2)
        self.com_ref, vel_ref, acc_ref = self.com.getReference()
        self.com0 = self.com_ref.copy()

        base = Cartesian(
            "base", self.model, "world", "pelvis", self.variables.getVariable("qddot")
        )
        base.setLambda(1.1)

        contact_tasks = list()
        for contact_frame in self.contact_frames:
            contact_tasks.append(
                Cartesian(
                    contact_frame + "_kin",
                    self.model,
                    "world",
                    contact_frame,
                    self.variables.getVariable("qddot"),
                )
            )

        self.posture = Postural(self.model, self.variables.getVariable("qddot"))
        # self.posture.setLambda(1.2)
        idx = [18 + i for i in range(self.model.nv - 18)]
        # print("idx", idx)
        self.stack = 1.0 * self.com + 0.0001 * self.posture
        # % idx
        # + 0.1 * (base % [3, 4, 5])

        # idx = [6 + i for i in range(self.model.nv - 6)]
        # print("idx", idx)

        # self.stack = 0.1 * self.posture  % idx

        force_variables = list()
        for i in range(len(self.contact_frames)):
            self.stack = self.stack + 5.0 * (contact_tasks[i] % [0, 1, 2])
            force_variables.append(self.variables.getVariable(self.contact_frames[i]))

        self.torques = Torque(
            model=self.model,
            qddot_var=self.variables.getVariable("qddot"),
            contact_links=self.contact_frames,
            force_vars=force_variables,
        )
        self.stack = self.stack + 1e-8 * MinimizeVariable("min_torques", self.torques)
        self.stack = self.stack + 1e-3 * MinimizeVariable(
            "min_qddot", self.variables.getVariable("qddot")
        )

        self.right_hand = Cartesian(
            "right_hand_point_contact",
            self.model,
            "right_hand_point_contact",
            "world",
            self.variables.getVariable("qddot"),
        )

        self.stack = self.stack + 0.01 * self.right_hand

        # Creates the stack.
        # Notice:  we do not need to keep track of the DynamicFeasibility constraint so it is created when added into the stack.
        # The same can be done with other constraints such as Joint Limits and Velocity Limits
        self.stack = pysot.AutoStack(self.stack) << DynamicFeasibility(
            "floating_base_dynamics",
            self.model,
            self.variables.getVariable("qddot"),
            force_variables,
            self.contact_frames,
        )
        self.stack = self.stack << JointLimits(
            self.model,
            self.variables.getVariable("qddot"),
            qmax,
            qmin,
            10.0 * dqmax,
            self.control_dt,
        )
        self.stack = self.stack << VelocityLimits(
            self.model, self.variables.getVariable("qddot"), dqmax, self.control_dt
        )
        self.stack = self.stack << TorqueLimits(
            self.model,
            self.variables.getVariable("qddot"),
            force_variables,
            self.contact_frames,
            torque_limits,
        )
        for i in range(len(self.contact_frames)):
            T = self.model.getPose(self.contact_frames[i])
            mu = (T.linear, 0.8)  # rotation is world to contact
            self.stack = self.stack << FrictionCone(
                self.contact_frames[i],
                self.variables.getVariable(self.contact_frames[i]),
                self.model,
                mu,
            )

        # Creates the solver
        self.solver = pysot.iHQP(self.stack)

        # Initialize the node
        self.initialize_force_publishers(self.contact_frames)

        msg = JointState()
        msg.name = self.model.getJointNames()[1::]

        w_T_b = TransformStamped()
        w_T_b.header.frame_id = "world"
        w_T_b.child_frame_id = "pelvis"

        self.force_msgs = {}
        for contact_frame in self.contact_frames:
            self.force_msgs[contact_frame] = WrenchStamped()
            self.force_msgs[contact_frame].header.frame_id = contact_frame
            self.force_msgs[contact_frame].wrench.torque.x = self.force_msgs[
                contact_frame
            ].wrench.torque.y = self.force_msgs[contact_frame].wrench.torque.z = 0.0
        #
        self.alpha = 0.2
        self.a = 0.0

        self.first = True
        self.start_opensot = False

        #######################

    def start_opensot_callback(self, msg: Bool):
        if msg.data:
            self.start_opensot = True
        else:
            self.start_opensot = False
            # restart q_init

    def emergency_stop_callback(self, msg: Bool):
        if msg.data:
            self.motors_on = 0

    def initialize_force_publishers(self, contact_frames):
        for contact_frame in contact_frames:
            self.force_publishers[contact_frame] = self.create_publisher(
                WrenchStamped, "wrench_" + contact_frame, 10
            )

    def publish(self, joint_state_msg, transform_msg, force_msgs=None):
        # self.joint_state_publisher.publish(joint_state_msg)
        # self.base_link_broadcaster.sendTransform(transform_msg)

        if force_msgs is not None:
            for contact_frame, force_msg in force_msgs.items():
                self.force_publishers[contact_frame].publish(force_msg)

    def control(self):
        low_cmd = LowCmd()
        self.time += self.control_dt

        low_cmd.mode_pr = self.mode_
        low_cmd.mode_machine = self.mode_machine

        if self.time < self.init_duration_s:
            for i in range(G1_NUM_MOTOR):
                ratio = self.clamp(self.time / self.init_duration_s, 0.0, 1.0)
                cmd = low_cmd.motor_cmd[i]
                cmd.mode = self.motors_on
                # This produces an initial shock that makes the simulation fall
                # cmd.q = (1.0 - ratio) * self.motor[i].q + ratio * q_init[i]
                cmd.q = q_init[i]
                cmd.dq = 0.0
                cmd.tau = 0.0
                cmd.kp = Kp[i]
                cmd.kd = Kd[i]
        elif self.start_opensot:
            self.t += self.control_dt
            dt = self.control_dt
            self.model.setJointPosition(self.q)
            self.model.setJointVelocity(self.dq)
            self.model.update()

            # # Linear interpolation of the CoM 5cm above
            # if self.first:
            #     self.x0 = self.com_ref.copy()
            #     self.x1 = self.x0.copy()
            #     self.x1[2] += 0.05
            #     self.first = False
            # if self.a < 1.0:
            #     self.a += 0.5 * dt
            #     self.com_ref = (1.0 - self.a) * self.x0 + self.a * self.x1

            # Compute new reference for CoM task
            self.com_ref[2] = self.com0[2]
            # - 0.05 + self.alpha * np.sin(3.1415 * self.t)
            # com_ref[1] = com0[1] + alpha * np.cos(3.1415 * t)

            self.com.setReference(self.com_ref)

            # Teleop reference
            if self.right_hand_pose_ref is not None:
                right_hand_pose_affine = self.right_hand.getReference()

                right_hand_pose_affine[0].translation = (
                    self.right_hand_pose_ref.position.x,
                    self.right_hand_pose_ref.position.y,
                    self.right_hand_pose_ref.position.z,
                )
                print("right_hand_pose_affine", right_hand_pose_affine[0].translation)
                R = tf_transformations.quaternion_matrix(
                    [
                        self.right_hand_pose_ref.orientation.x,
                        self.right_hand_pose_ref.orientation.y,
                        self.right_hand_pose_ref.orientation.z,
                        self.right_hand_pose_ref.orientation.w,
                    ]
                )
                R = R[:3, :3]  # Extract rotation part
                right_hand_pose_affine[0].linear = R

                # print("right_hand_pose_affine", right_hand_pose_affine)
                self.right_hand.setReference(
                    right_hand_pose_affine[0], np.zeros(6), np.zeros(6)
                )

            # self.q_ref = np.zeros(self.model.nq)
            # self.q_ref[0:7] = self.q[0:7].copy()  # Use the current base state
            # self.q_ref[7:] = self.bag_motor.copy()  # Use the bag motor state for reference
            # # self.get_logger().info(self.q_ref)
            # self.posture.setReference(self.q_ref)

            # Update Stack
            self.stack.update()
            #
            # Solve
            x = self.solver.solve()
            ddq = self.variables.getVariable("qddot").getValue(
                x
            )  # from variables vector we retrieve the joint accelerations
            self.q = self.model.sum(
                self.q, self.dq * dt + 0.5 * ddq * dt * dt
            )  # we use the model sum to account for the floating-base

            # Publsh joint states
            msg = JointState()
            msg.name = self.model.getJointNames()[1::]
            msg.position = self.q[7::]
            msg.header.stamp = self.get_clock().now().to_msg()
            # self.joint_state_publisher.publish(msg)

            # Publish base link transform
            w_T_b = TransformStamped()
            w_T_b.header.frame_id = "world"
            w_T_b.child_frame_id = "pelvis"
            w_T_b.header.stamp = msg.header.stamp
            w_T_b.transform.translation.x = self.q[0]
            w_T_b.transform.translation.y = self.q[1]
            w_T_b.transform.translation.z = self.q[2]
            w_T_b.transform.rotation.x = self.q[3]
            w_T_b.transform.rotation.y = self.q[4]
            w_T_b.transform.rotation.z = self.q[5]
            w_T_b.transform.rotation.w = self.q[6]
            # self.base_link_broadcaster.sendTransform(w_T_b)

            self.dq += ddq * dt
            #
            # # Publish joint states
            # msg.position = q[7::]
            # msg.header.stamp = node.get_clock().now().to_msg()
            # #
            # w_T_b.header.stamp = msg.header.stamp
            # w_T_b.transform.translation.x = q[0]
            # w_T_b.transform.translation.y = q[1]
            # w_T_b.transform.translation.z = q[2]
            # w_T_b.transform.rotation.x = q[3]
            # w_T_b.transform.rotation.y = q[4]
            # w_T_b.transform.rotation.z = q[5]
            # w_T_b.transform.rotation.w = q[6]
            # ##################
            for contact_frame in self.contact_frames:
                T = self.model.getPose(contact_frame)
                self.force_msgs[contact_frame].header.stamp = msg.header.stamp
                f_local = T.linear.transpose() @ self.variables.getVariable(
                    contact_frame
                ).getValue(
                    x
                )  # here we compute the value of the contact forces in local frame from world frame
                self.force_msgs[contact_frame].wrench.force.x = f_local[0]
                self.force_msgs[contact_frame].wrench.force.y = f_local[1]
                self.force_msgs[contact_frame].wrench.force.z = f_local[2]

            if self.force_msgs is not None:
                for contact_frame, force_msg in self.force_msgs.items():
                    self.force_publishers[contact_frame].publish(force_msg)

            tau = self.get_inverse_dynamics(x, ddq)

            for i in range(G1_NUM_MOTOR):
                cmd = low_cmd.motor_cmd[i]
                cmd.mode = self.motors_on
                cmd.q = self.q[i + 7]
                cmd.dq = self.dq[i + 6]
                # cmd.dq = 0.0
                cmd.tau = tau[6 + i]
                # cmd.tau = 0.0
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
        # self.get_logger().info("Published low_cmd")

    # def bag_lowcmd_callback(self, cmd_msg: LowCmd):
    #     for i in range(G1_NUM_MOTOR):
    #         self.bag_motor[i] = cmd_msg.motor_cmd[i].q

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

    def teleop_callback(self, msg):
        """
        Callback for teleoperation.
        """
        self.right_hand_pose_ref = msg.pose

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

    def clamp(self, value, low, high):
        if value < low:
            return low
        if value > high:
            return high
        return value


def main(args=None):
    rclpy.init(args=args)
    node = MoveExample()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

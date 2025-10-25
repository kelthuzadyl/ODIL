"""
Robot interface for Interbotix XS manipulator.
"""

from interbotix_common_modules.common_robot.robot import robot_startup, robot_shutdown
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

class Robot:
    """
    Encapsulates InterbotixManipulatorXS initialization and basic control methods.
    """

    def __init__(
        self,
        robot_model: str = 'vx300s',
        robot_name: str = 'vx300s',
        group_name: str = 'arm',
        gripper_name: str = 'gripper',
        moving_time: float = 3.0,
        accel_time: float = 1.5,
    ):
        """
        Initialize the robot and move to default position.

        Args:
            robot_model: Interbotix robot model.
            robot_name: Robot name.
            group_name: Arm group name.
            gripper_name: Gripper name.
            moving_time: Default move duration.
            accel_time: Default acceleration duration.
            default_joint_positions: Initial joint positions.
        """

        # Create the robot interface
        self.bot = InterbotixManipulatorXS(
            robot_model=robot_model,
            robot_name=robot_name,
            group_name=group_name,
            gripper_name=gripper_name,
            moving_time=moving_time,
            accel_time=accel_time
        )
        
        # Start robot
        robot_startup()


    def move_to_default_pose(self):
        """Move robot arm to default joint positions."""
        default_joint_positions = [0, -0.72, 0.59, 0, 1.02, 0]
        self.set_joint_positions(default_joint_positions)

    # ──────────────────────────────
    # Wrapper methods for robot control
    # ──────────────────────────────
    def set_joint_positions(self, positions: list, moving_time: float = 3.0, blocking: bool = True):
        """Move robot arm to specified joint positions."""
        self.bot.arm.set_joint_positions(positions, moving_time=moving_time, blocking=blocking)

    def set_ee_cartesian_trajectory(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        moving_time: float = 0.2
    ):
        """Move the robot end-effector along a Cartesian trajectory."""
        self.bot.arm.set_ee_cartesian_trajectory(
            x=x, y=y, z=z,
            roll=roll, pitch=pitch, yaw=yaw,
            moving_time=moving_time
        )

    def get_ee_pose(self):
        """Return the current end-effector pose as a 4x4 homogeneous matrix."""
        return self.bot.arm.get_ee_pose()
    
    def shutdown(self):
        """Shutdown the robot safely."""
        robot_shutdown()
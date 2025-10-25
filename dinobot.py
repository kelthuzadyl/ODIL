"""
3D Visual Servoing Node with DINO Correspondences (DINObot)
----------------------------------------------------
"""

import numpy as np
from collections import deque
from PIL import Image

import rclpy
from rclpy.node import Node

from match_servoers.dino_servoer import DINOVisualServoer
from config.config import HANDEYE_TRANSFORM, K
from config.robot import Robot
from vs_utils import transform_to_state, solve_transform_3d


class DINOBot(DINOVisualServoer, Node):
    """
    3D Visual servoing node using DINO correspondences.
    """

    def __init__(
        self,
        DIR: str,
        bot: Robot,
        use_4dof_control: bool = False
    ):
        Node.__init__(self, 'visual_servoing_3d_node')
        self.bot = bot
        self.DIR = DIR

        # Load reference images
        rgb_ref = np.array(Image.open(f"{DIR}/ref_rgb_wrist.png"))
        seg_ref = np.array(Image.open(f"{DIR}/ref_mask_wrist.png")).astype(bool)
        self.depth_ref = np.load(f"{DIR}/ref_depth_wrist.npy")

        # Initialize DINO visual servoer
        DINOVisualServoer.__init__(
            self,
            DIR=DIR,
            rgb_ref=rgb_ref,
            seg_ref=seg_ref,
            use_depth=True,
            silent=True,
            visualize_matches=False
        )

        # Servoing parameters
        self.max_translation_step = 0.002 # increase if too slow
        self.max_rotation_step = np.deg2rad(2) # increase if too slow
        self.gains = [0.1] * 6
        self.terminate_threshold = (0.05, 5)  # meters, degrees
        self.error_window = deque(maxlen=3)  # For both error tracking and termination
        self.use_4dof_control = use_4dof_control

        # State tracking
        self.num_iteration = 0
        self.is_complete = False

    # ──────────────────────────────
    # Control computations
    # ──────────────────────────────
    def compute_goal_state(self, T_delta_cam: np.ndarray):
        T_current_eef_world = self.bot.get_ee_pose()
        T_delta_eef = HANDEYE_TRANSFORM @ T_delta_cam @ np.linalg.inv(HANDEYE_TRANSFORM)
        goal_state = transform_to_state(T_current_eef_world @ T_delta_eef)
        current_state = transform_to_state(T_current_eef_world)
        return goal_state, current_state

    def compute_control_input(self, goal_state: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        translation_error = goal_state[:3] - current_state[:3]
        rotation_error = goal_state[3:] - current_state[3:]

        translation = np.clip(
            np.array(self.gains[:3]) * translation_error,
            -self.max_translation_step,
            self.max_translation_step
        )
        rotation = np.clip(
            np.array(self.gains[3:]) * rotation_error,
            -self.max_rotation_step,
            self.max_rotation_step
        )
        return np.concatenate([translation, rotation])

    # ──────────────────────────────
    # Main servoing loop
    # ──────────────────────────────
    def run(self):
        while not self.is_complete:
            # Acquire measurement
            mkpts_0, mkpts_1, depth_cur = self.match_dino(filter_seg_ref=True)
            if mkpts_0 is None or len(mkpts_0) <= 3:
                self.get_logger().info("Not enough keypoints, skipping iteration.")
                continue

            # Compute transformation
            T_delta_cam = solve_transform_3d(
                mkpts_0,
                mkpts_1,
                self.depth_ref,
                depth_cur,
                K
            )
            if T_delta_cam is None:
                self.get_logger().info("Failed to compute transformation, skipping iteration.")
                continue

            # Compute errors
            T_delta_cam_inv = np.linalg.inv(T_delta_cam)
            translation_error = np.linalg.norm(T_delta_cam_inv[:3, 3])
            rotation_error = np.rad2deg(np.arccos((np.trace(T_delta_cam_inv[:3, :3]) - 1) / 2))
            self.get_logger().info(
                f"Translation Error: {translation_error:.6f}, Rotation Error: {rotation_error:.2f} deg"
            )

            # Check termination and update error history
            self.error_window.append((translation_error, rotation_error))
            if len(self.error_window) == self.error_window.maxlen:
                if all(t < self.terminate_threshold[0] and r < self.terminate_threshold[1]
                       for t, r in self.error_window):
                    self.get_logger().info("Alignment achieved, terminating servoing loop.")
                    self.is_complete = True

            # Compute goal, current state and control input
            goal_state, current_state = self.compute_goal_state(T_delta_cam)
            control_input = self.compute_control_input(goal_state, current_state)

            # Send command
            self.bot.set_ee_cartesian_trajectory(
                x=control_input[0],
                y=control_input[1],
                z=control_input[2],
                roll=control_input[3] if not self.use_4dof_control else 0,
                pitch=control_input[4] if not self.use_4dof_control else 0,
                yaw=control_input[5],
                moving_time=0.2
            )

            self.num_iteration += 1


# ──────────────────────────────
# Entry point
# ──────────────────────────────
def main():
    rclpy.init()
    DIR = "example_tasks/pan"
    bot = Robot()
    bot.move_to_default_pose()

    node = DINOBot(DIR, bot, use_4dof_control=False)
    try:
        node.run()
    finally:
        node.destroy_node()
        bot.shutdown()


if __name__ == "__main__":
    main()
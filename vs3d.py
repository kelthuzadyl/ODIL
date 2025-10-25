"""
3D Visual Servoing Node with optional UKF integration
----------------------------------------------------
"""

import numpy as np
from collections import deque
from PIL import Image

import rclpy
from rclpy.node import Node

from match_servoers.lightglue_servoer import LightGlueVisualServoer
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter as UKF
from config.config import HANDEYE_TRANSFORM, K
from config.robot import Robot
from vs_utils import transform_to_state, weighted_solve_transform_3d


class VisualServoing3D(LightGlueVisualServoer, Node):
    """
    3D Visual servoing node using LightGlue with optional UKF filtering.
    """

    def __init__(
        self,
        DIR: str,
        bot: Robot,
        use_4dof_control: bool = False,
        use_ukf: bool = True,
        prior_state: np.ndarray = None,
        prior_covariance: np.ndarray = None
    ):
        Node.__init__(self, 'visual_servoing_3d_node')
        self.bot = bot
        self.DIR = DIR
        self.use_ukf = use_ukf

        # Load reference images
        rgb_ref = np.array(Image.open(f"{DIR}/ref_rgb_wrist.png"))
        seg_ref = np.array(Image.open(f"{DIR}/ref_mask_wrist.png")).astype(bool)
        self.depth_ref = np.load(f"{DIR}/ref_depth_wrist.npy")

        # Initialize LightGlue visual servoer
        LightGlueVisualServoer.__init__(
            self,
            rgb_ref=rgb_ref,
            seg_ref=seg_ref,
            use_depth=True,
            features='superpoint',
            silent=True
        )

        # Servoing parameters
        self.max_translation_step = 0.002 # increase if too slow
        self.max_rotation_step = np.deg2rad(2) # increase if too slow
        self.gains = [0.1] * 6
        self.terminate_threshold = (0.05, 5)  # meters, degrees
        self.error_window = deque(maxlen=3)  # For both error tracking and termination
        self.use_4dof_control = use_4dof_control

        # Optional UKF
        self.ukf = None
        if self.use_ukf:
            self.ukf = self.initialize_ukf(prior_state, prior_covariance)

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
    # Optional UKF functions
    # ──────────────────────────────
    def fx(self, x, dt):
        # Process model for goal state: assume goal state changes slowly
        # This models the fact that the true goal doesn't change rapidly
        return x  # Identity dynamics - goal state is relatively stable

    def hx(self, x):
        # Measurement model: we directly observe goal state from visual servoing
        # The visual transformation gives us a direct measurement of the goal state
        return x  # Direct measurement of goal state

    def initialize_ukf(self, prior_state=None, prior_covariance=None):
        state_dim = 6
        sigma_points = MerweScaledSigmaPoints(n=state_dim, alpha=0.1, beta=2.0, kappa=3-state_dim)
        ukf = UKF(dim_x=state_dim, dim_z=state_dim, fx=self.fx, hx=self.hx, dt=1.0, points=sigma_points)

        # Initialize with an initial goal state estimate
        if prior_state is None:
            # Start with current robot pose as initial goal estimate
            current_robot_pose = self.bot.get_ee_pose()
            ukf.x = transform_to_state(current_robot_pose)
            ukf.P = np.eye(state_dim) * 0.1  # Moderate initial uncertainty
        else:
            ukf.x = transform_to_state(prior_state)
            ukf.P = prior_covariance if prior_covariance is not None else np.eye(state_dim) * 0.1

        # Measurement noise (uncertainty in goal state estimates from vision)
        ukf.R = np.eye(state_dim) * 0.01  # Vision-based goal estimation noise
        # Process noise (how much the true goal state can change between measurements)
        ukf.Q = np.eye(state_dim) * 0.001  # Goal state changes slowly
        return ukf

    def estimate_measurement_noise(self):
        """Estimate measurement noise based on recent error magnitudes."""
        if len(self.error_window) < 2:
            return None
        
        # Extract translation and rotation errors from recent measurements
        trans_errors = [err[0] for err in self.error_window]
        rot_errors = [err[1] for err in self.error_window]
        
        # Convert to variance estimates (higher errors = higher noise)
        trans_variance = np.var(trans_errors) if len(trans_errors) > 1 else 0.01
        rot_variance = np.var(rot_errors) if len(rot_errors) > 1 else 0.01
        
        # Create diagonal covariance matrix
        # Scale rotation variance appropriately (convert from degrees to radians)
        rot_variance_rad = np.deg2rad(rot_variance)
        
        cov_matrix = np.diag([trans_variance, trans_variance, trans_variance, 
                             rot_variance_rad, rot_variance_rad, rot_variance_rad])
        return cov_matrix

    def is_measurement_valid(self, cov_matrix):
        """Check if the measurement noise estimate is reasonable."""
        if cov_matrix is None:
            return False
        # Check if any variance is too large (indicating unstable estimates)
        max_variance = np.max(np.diag(cov_matrix))
        return max_variance < 1.0  # Reasonable threshold

    # ──────────────────────────────
    # Main servoing loop
    # ──────────────────────────────
    def run(self):
        while not self.is_complete:
            # Acquire measurement
            mkpts_scores_0, mkpts_scores_1, depth_cur = self.match_lightglue(filter_seg_ref=True)
            if mkpts_scores_0 is None or len(mkpts_scores_0) <= 3:
                self.get_logger().info("Not enough keypoints, skipping iteration.")
                continue

            # Compute transformation
            _, _, T_delta_cam = weighted_solve_transform_3d(
                mkpts_scores_0,
                mkpts_scores_1,
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

            # Compute goal & current state
            goal_state, current_state = self.compute_goal_state(T_delta_cam)

            if self.use_ukf:
                # Predict the next goal state estimate
                self.ukf.predict()
                
                # Estimate measurement noise based on recent error patterns
                cov_matrix = self.estimate_measurement_noise()
                
                if self.is_measurement_valid(cov_matrix):
                    # Adapt measurement noise based on recent error variability
                    base_noise = np.eye(6) * 0.01
                    adaptive_noise = base_noise + cov_matrix * 0.1
                    self.ukf.R = adaptive_noise
                
                # Update filter with the noisy goal state measurement from vision
                self.ukf.update(goal_state)
                
                # Use filtered (smoothed) goal state for control
                filtered_goal_state = self.ukf.x
                control_input = self.compute_control_input(filtered_goal_state, current_state)
            else:
                # Use raw goal state without filtering
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

    node = VisualServoing3D(DIR, bot, use_4dof_control=False, use_ukf=True)
    try:
        node.run()
    finally:
        node.destroy_node()
        bot.shutdown()


if __name__ == "__main__":
    main()

"""
2.5D Visual Servoing Node
----------------------------------------------------
Homography-based visual servoing using LightGlue and PID control.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
from PIL import Image
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node

from match_servoers.lightglue_servoer import LightGlueVisualServoer
from config.config import K
from config.robot import Robot
from vs_utils import normalize_mkpts, weighted_solve_transform_3d


# ──────────────────────────────
# Data Structures
# ──────────────────────────────
TaskFunctionResult = namedtuple('TaskFunctionResult', [
    'delta_t', 'delta_r', 'error', 'u_star', 'v_star', 'u', 'v'
])

def create_empty_result() -> TaskFunctionResult:
    """Create an empty/failed TaskFunctionResult."""
    return TaskFunctionResult(None, None, None, None, None, None, None)


# ──────────────────────────────
# PID Controller
# ──────────────────────────────
class PIDController:
    """Simple PID controller for 1D signals."""

    def __init__(self, Kp: float, Ki: float, Kd: float) -> None:
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error: float = 0.0
        self.integral: float = 0.0

    def update(self, error: float) -> float:
        """Update controller state and return control output."""
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

    def reset(self) -> None:
        """Reset the controller state."""
        self.prev_error = 0.0
        self.integral = 0.0


# ──────────────────────────────
# Visual Servoing Node
# ──────────────────────────────
class VisualServoing25D(LightGlueVisualServoer, Node):
    """
    2.5D Visual Servoing using homography-based control and LightGlue keypoints.
    """

    def __init__(
        self, 
        DIR: str, 
        bot: Robot, 
        use_4dof_control: bool = False,
        enable_visualization: bool = True
    ) -> None:
        Node.__init__(self, 'visual_servoing_25d_node')
        self.bot = bot
        self.DIR = DIR
        self.use_4dof_control = use_4dof_control
        self.enable_visualization = enable_visualization

        # Load reference data
        rgb_ref = np.array(Image.open(f"{DIR}/ref_rgb_wrist.png"))
        seg_ref = np.array(Image.open(f"{DIR}/ref_mask_wrist.png")).astype(bool)
        self.depth_ref = np.load(f"{DIR}/ref_depth_wrist.npy")

        # Initialize LightGlue (extractor + matcher)
        LightGlueVisualServoer.__init__(
            self,
            rgb_ref=rgb_ref,
            seg_ref=seg_ref,
            use_depth=True,
            features='superpoint',
            silent=True,
        )

        # PID controllers for each DoF
        self.pid_x = PIDController(0.05, 0.0, 0.01)
        self.pid_y = PIDController(0.05, 0.0, 0.01)
        self.pid_z = PIDController(0.05, 0.0, 0.01)
        self.pid_rx = PIDController(0.05, 0.0, 0.01)
        self.pid_ry = PIDController(0.05, 0.0, 0.01)
        self.pid_rz = PIDController(0.05, 0.0, 0.01)

        # Servoing state
        self.max_translation_step = 0.001 # increase if too slow
        self.max_rotation_step = np.deg2rad(1) # increase if too slow
        self.error_window = deque(maxlen=3)
        self.num_iteration = 0
        self.is_complete = False

        # Thresholds
        self.segmentation_threshold = 25.0  # apply reference mask to current frame if error < 25 pixels
        self.convergence_pixel_threshold = 5.0
        self.convergence_rotation_threshold = 2.0
        self.convergence_depth_threshold = 0.01
        self.divergence_pixel_threshold = 100.0
        self.divergence_rotation_threshold = 20.0
        self.critical_divergence_threshold = 150.0

        # Reference point persistence
        self.reference_file = f"{DIR}/reference_point_for_25dvs.npz"
        self.m_star: Optional[np.ndarray] = None
        self.Z_star: Optional[float] = None
        self.load_reference_point()

        # Visualization setup (only if enabled)
        self.fig = None
        self.ax = None
        if self.enable_visualization:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            plt.ion()

        # Internal state
        self.previous_estimate_of_R: Optional[np.ndarray] = None  # Used for homography decomposition continuity
        self._initial_transform_computed: bool = False 
    # ──────────────────────────────
    # Reference Point Persistence
    # ──────────────────────────────
    def save_reference_point(self) -> None:
        """Save the reference pixel (m_star) and depth (Z_star)."""
        if self.m_star is not None and self.Z_star is not None:
            np.savez(self.reference_file, m_star=self.m_star, Z_star=self.Z_star)
            self.get_logger().info(f"Reference point saved to {self.reference_file}")
        else:
            self.get_logger().info("No reference point to save.")

    def load_reference_point(self) -> bool:
        """Load saved reference pixel and depth if available."""
        if os.path.exists(self.reference_file):
            try:
                data = np.load(self.reference_file)
                self.m_star = data['m_star']
                self.Z_star = float(data['Z_star'])
                self.get_logger().info(f"Loaded reference: m*={self.m_star}, Z*={self.Z_star}")
                return True
            except Exception as e:
                self.get_logger().error(f"Failed to load reference file: {e}")
                self.m_star, self.Z_star = None, None
                return False
        else:
            self.get_logger().info(f"No reference file at {self.reference_file}")
            return False

    def reset_reference_point(self) -> None:
        """Reset and delete the reference point file."""
        self.m_star, self.Z_star = None, None
        if os.path.exists(self.reference_file):
            os.remove(self.reference_file)
            self.get_logger().info(f"Reference file removed: {self.reference_file}")

    def has_saved_reference_point(self) -> bool:
        """Return True if a saved reference point exists."""
        return os.path.exists(self.reference_file)

    # ──────────────────────────────
    # Visualization
    # ──────────────────────────────
    def visualize_tracking(self, current_rgb: np.ndarray, u_star: float, v_star: float, u: float, v: float, error: float) -> None:
        """Overlay current vs desired pixels on the current frame."""
        if not self.enable_visualization or self.ax is None:
            return
            
        self.ax.clear()
        self.ax.imshow(current_rgb)

        self.ax.plot(u_star, v_star, 'go', markersize=12, markeredgewidth=2, markerfacecolor='none', label='Desired')
        self.ax.plot(u, v, 'ro', markersize=12, markeredgewidth=2, markerfacecolor='none', label='Current')
        self.ax.plot([u_star, u], [v_star, v], 'b--', alpha=0.7)

        self.ax.text(10, 30, f'Pixel Error: {error:.2f}', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
        self.ax.text(10, 60, f'Iteration: {self.num_iteration + 1}', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

        self.ax.set_xlim(0, current_rgb.shape[1])
        self.ax.set_ylim(current_rgb.shape[0], 0)
        self.ax.set_title('Homography-based Visual Servoing')
        self.ax.legend()
        self.ax.grid(alpha=0.3)
        plt.draw()
        plt.pause(0.01)

    # ──────────────────────────────
    # Core Logic
    # ──────────────────────────────
    def update_reference_point(self, mkpts_0: np.ndarray, mkpts_1: np.ndarray, K_mat: np.ndarray, thresh: float = 0.5) -> None:
        """
        Compute and save the most confident reference point with non-zero depth.

        mkpts_0, mkpts_1: Nx2 pixel arrays (reference, current)
        """
        try:
            ransac_thr = thresh / np.mean([K_mat[0, 0], K_mat[1, 1]])
            mkpts_0_n = normalize_mkpts(mkpts_0, K_mat)
            mkpts_1_n = normalize_mkpts(mkpts_1, K_mat)

            H_norm, inliers = cv2.findHomography(
                mkpts_0_n, mkpts_1_n, cv2.USAC_MAGSAC, ransac_thr, confidence=0.99999
            )
            if H_norm is None or inliers is None:
                raise ValueError("Homography estimation failed")

            mask = inliers.ravel() == 1
            if np.count_nonzero(mask) == 0:
                raise ValueError("No inliers found by homography")

            inlier_orig = mkpts_0[mask]
            depths = self.depth_ref[inlier_orig[:, 1].astype(int), inlier_orig[:, 0].astype(int)]
            valid = depths > 0
            if not np.any(valid):
                raise ValueError("No inliers with non-zero depth found")

            # Compute reprojection errors in normalized coordinates (only valid inliers)
            valid_mkpts0_n = mkpts_0_n[mask][valid]
            valid_mkpts1_n = mkpts_1_n[mask][valid]
            reproj_h = (H_norm @ np.hstack([valid_mkpts0_n, np.ones((valid_mkpts0_n.shape[0], 1))]).T).T
            reproj = reproj_h[:, :2] / reproj_h[:, 2:3]
            errors = np.linalg.norm(reproj - valid_mkpts1_n[:, :2], axis=1)
            best_idx = np.argmin(errors)

            self.m_star = inlier_orig[valid][best_idx].astype(float)
            self.Z_star = float(depths[valid][best_idx]) / 1000.0 # assume raw depth in mm (Realsense)
            self.save_reference_point()
            self.get_logger().info(f"Reference point computed: m*={self.m_star}, Z*={self.Z_star}")

        except Exception as e:
            self.get_logger().error(f"update_reference_point failed: {e}")
            self.m_star, self.Z_star = None, None

    def decompose_homography(self, H_norm: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Decompose homography and select the best solution.

        Returns (R, t) or (None, None) if failed.
        """
        try:
            n_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H_norm, np.eye(3))
        except Exception as e:
            self.get_logger().warning(f"Homography decomposition failed: {e}")
            return None, None

        best_idx = None
        best_score = float('inf')

        for i in range(n_solutions):
            R_i = rotations[i]
            t_i = translations[i].flatten()
            # prefer positive z (forward motion)
            if t_i[2] <= 0:
                continue

            if self.previous_estimate_of_R is None:
                score = np.linalg.norm(t_i)  # fallback metric
            else:
                # angle between previous R and candidate R
                # clip for numerical safety
                trace = np.clip((np.trace(self.previous_estimate_of_R.T @ R_i) - 1) / 2, -1.0, 1.0)
                score = np.arccos(trace)

            if score < best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            return None, None

        selected_R = rotations[best_idx]
        selected_t = translations[best_idx].flatten()
        self.previous_estimate_of_R = selected_R
        return selected_R, selected_t

    def _safe_depth_at(self, depth_img: np.ndarray, u: int, v: int) -> float:
        """Return depth in meters at integer pixel (u,v), or 0.0 if invalid/out-of-bounds."""
        h, w = depth_img.shape[:2]
        if u < 0 or u >= w or v < 0 or v >= h:
            return 0.0
        val = float(depth_img[v, u])
        return val / 1000.0 if val != 0 else 0.0

    def get_task_function_homography(
        self,
        mkpts_0: np.ndarray,
        mkpts_1: np.ndarray,
        depth_cur: np.ndarray,
        current_rgb: np.ndarray,
        K_mat: np.ndarray
    ) -> TaskFunctionResult:
        """
        Compute task function (delta translation, delta rotation, pixel error, and pixel coords)
        using homography decomposition and the stored reference point (m_star, Z_star).

        Returns:
            TaskFunctionResult with fields (delta_t, delta_r, error, u_star, v_star, u, v)
            or empty result on failure.
        """
        if self.m_star is None or self.Z_star is None:
            self.get_logger().warning("Reference point not initialized (m_star/Z_star).")
            return create_empty_result()

        try:
            mkpts0_n = normalize_mkpts(mkpts_0, K_mat)
            mkpts1_n = normalize_mkpts(mkpts_1, K_mat)

            H_norm, inliers = cv2.findHomography(mkpts0_n, mkpts1_n, cv2.USAC_MAGSAC, 0.001, confidence=0.99999)
            if H_norm is None:
                self.get_logger().warning("Homography estimation failed in get_task_function_homography.")
                return create_empty_result()

            selected_R, selected_t = self.decompose_homography(H_norm)
            if selected_R is None:
                self.get_logger().warning("No valid homography decomposition found.")
                return create_empty_result()

            r_theta = R.from_matrix(selected_R).as_euler('xyz')

            # Prepare m_star in homogeneous coordinates
            if self.m_star is None:
                return create_empty_result()
            if len(self.m_star) == 2:
                m_star_h = np.array([self.m_star[0], self.m_star[1], 1.0], dtype=float)
            elif len(self.m_star) == 3:
                m_star_h = np.array(self.m_star, dtype=float).reshape(3,)
            else:
                self.get_logger().warning("Unexpected m_star shape.")
                return create_empty_result()

            # Reproject m_star through H_norm
            reproj_h = H_norm @ m_star_h
            if reproj_h[2] == 0:
                self.get_logger().warning("Degenerate reprojection (w=0).")
                return create_empty_result()
            m = reproj_h / reproj_h[2]

            # Pixel coordinates (u,v) for the projected point
            pixel_coords = K_mat @ m
            u = float(pixel_coords[0])
            v = float(pixel_coords[1])

            # Pixel coords for the reference
            pixel_coords_ref = K_mat @ m_star_h
            u_star = float(pixel_coords_ref[0])
            v_star = float(pixel_coords_ref[1])

            # Get depth at (u,v), with a small neighborhood fallback
            u_i, v_i = int(round(u)), int(round(v))
            Z = self._safe_depth_at(depth_cur, u_i, v_i)
            if Z == 0.0:
                # 3x3 neighborhood average of non-zero depths
                u_min, u_max = max(0, u_i - 1), min(depth_cur.shape[1] - 1, u_i + 1)
                v_min, v_max = max(0, v_i - 1), min(depth_cur.shape[0] - 1, v_i + 1)
                window = depth_cur[v_min:v_max+1, u_min:u_max+1]
                nz = window[window > 0]
                if nz.size > 0:
                    Z = float(np.mean(nz)) / 1000.0
                else:
                    self.get_logger().warning("No valid depth in local window.")
                    return create_empty_result()

            # Pixel error (L1)
            error = float(np.linalg.norm(np.array([u_star, v_star]) - np.array([u, v]), ord=1))
            self.get_logger().info(f"Pixel coords u={u:.1f}, v={v:.1f}, Z={Z:.3f}m, error={error:.3f}")

            # Compute task deltas (as in original algorithm)
            if self.Z_star == 0:
                self.get_logger().warning("Z_star is zero, cannot compute deltas.")
                return create_empty_result()
            pho = Z / self.Z_star
            delta_x = (self.m_star[0] - pho * m[0]) * self.Z_star / K_mat[0, 0]
            delta_y = (self.m_star[1] - pho * m[1]) * self.Z_star / K_mat[1, 1]
            delta_z = self.Z_star * (1.0 - pho)

            delta_t = np.array([delta_x, delta_y, delta_z], dtype=float)
            delta_r = np.array(r_theta, dtype=float)

            return TaskFunctionResult(delta_t, delta_r, error, u_star, v_star, u, v)

        except Exception as e:
            self.get_logger().error(f"get_task_function_homography error: {e}")
            return create_empty_result()

    # ──────────────────────────────
    # Main Servoing Loop
    # ──────────────────────────────
    def run(self) -> None:
        """Run the servoing loop until convergence or failure."""
        error_val = 1e6
        while not self.is_complete:
            mkpts0, mkpts1, depth_cur = self.match_lightglue(
                filter_seg_ref=True,
                # Use configurable threshold for segmentation
                apply_seg_ref_to_current_rgb=(error_val < self.segmentation_threshold)
            )
            if mkpts0 is None or len(mkpts0) <= 3:
                continue

            current_rgb, _ = self.observe()
            if current_rgb is None:
                continue

            # Sort keypoints by descending score
            idx_sort = np.argsort(-mkpts0[:, -1])
            mkpts0, mkpts1 = mkpts0[idx_sort], mkpts1[idx_sort]

            # First T_delta estimate to initialize rotation tracking
            if not self._initial_transform_computed:
                res = weighted_solve_transform_3d(mkpts0, mkpts1, self.depth_ref, depth_cur, K)
                if res is None:
                    continue
                try:
                    T_delta_cam = res[2]
                    T_delta_cam_inv = np.linalg.inv(T_delta_cam)
                    self.previous_estimate_of_R = T_delta_cam_inv[:3, :3]
                    self._initial_transform_computed = True
                except Exception:
                    self.get_logger().warning("Failed to invert initial T_delta_cam, skipping iteration.")
                    continue

            # Ensure reference point exists
            if self.m_star is None:
                self.get_logger().info("No reference point found, computing new reference point...")
                self.update_reference_point(mkpts0[:, :2], mkpts1[:, :2], K)
            else:
                self.get_logger().info(f"Using loaded reference point: m*={self.m_star}, Z*={self.Z_star}")

            result = self.get_task_function_homography(mkpts0[:, :2], mkpts1[:, :2], depth_cur, current_rgb, K)
            if result.delta_t is None:
                continue

            # Visualize (now handles optional visualization internally)
            self.visualize_tracking(current_rgb, result.u_star, result.v_star, result.u, result.v, result.error)

            self.get_logger().info(f"[Step {self.num_iteration + 1}] Pixel error: {result.error:.3f}")

            # Termination/divergence checks using configurable thresholds
            rotation_error = np.rad2deg(np.linalg.norm(result.delta_r, ord=1))
            self.error_window.append((result.error, rotation_error, abs(result.delta_t[2])))

            if len(self.error_window) == self.error_window.maxlen:
                recent_pixels = [e[0] for e in self.error_window]
                recent_rot = [e[1] for e in self.error_window]
                recent_depth = [e[2] for e in self.error_window]

                if (all(p < self.convergence_pixel_threshold for p in recent_pixels) and
                        all(r < self.convergence_rotation_threshold for r in recent_rot) and
                        all(d < self.convergence_depth_threshold for d in recent_depth)):
                    self.get_logger().info("Convergence achieved! Stopping.")
                    self.is_complete = True
                    break

                if (any(p > self.divergence_pixel_threshold for p in recent_pixels) or 
                    any(r > self.divergence_rotation_threshold for r in recent_rot)):
                    self.get_logger().warning("Divergence detected; resetting PID controllers.")
                    self.pid_x.reset(); self.pid_y.reset(); self.pid_z.reset()
                    self.pid_rx.reset(); self.pid_ry.reset(); self.pid_rz.reset()
                    if result.error > self.critical_divergence_threshold:
                        self.get_logger().error("Critical divergence; stopping.")
                        self.is_complete = True
                        break

            # PID control and send command
            control_x = np.clip(self.pid_x.update(result.delta_t[0]), -self.max_translation_step, self.max_translation_step)
            control_y = np.clip(self.pid_y.update(result.delta_t[1]), -self.max_translation_step, self.max_translation_step)
            control_z = np.clip(self.pid_z.update(result.delta_t[2]), -self.max_translation_step, self.max_translation_step)
            control_rx = np.clip(self.pid_rx.update(result.delta_r[0]), -self.max_rotation_step, self.max_rotation_step)
            control_ry = np.clip(self.pid_ry.update(result.delta_r[1]), -self.max_rotation_step, self.max_rotation_step)
            control_rz = np.clip(self.pid_rz.update(result.delta_r[2]), -self.max_rotation_step, self.max_rotation_step)

            # You may need to modify signs for your specific robot configuration
            self.bot.set_ee_cartesian_trajectory(
                x=-control_x,
                y=control_y,
                z=control_z,
                roll=-control_rx if not self.use_4dof_control else 0.0,
                pitch=-control_ry if not self.use_4dof_control else 0.0,
                yaw=-control_rz,
                moving_time=0.2,
            )

            self.num_iteration += 1

    def cleanup_visualization(self) -> None:
        """Close visualization resources."""
        if self.enable_visualization and self.fig is not None:
            plt.ioff()
            plt.close(self.fig)


# ──────────────────────────────
# Entry Point
# ──────────────────────────────
def main() -> None:
    rclpy.init()
    DIR = "example_tasks/pan"
    bot = Robot()
    
    # Create visual servoing node with optional configuration
    node = VisualServoing25D(
        DIR, 
        bot, 
        use_4dof_control=True,
        enable_visualization=True,  # Set to False to disable visualization
    )

    try:
        node.run()
    finally:
        node.cleanup_visualization()
        node.destroy_node()
        bot.shutdown()


if __name__ == "__main__":
    main()

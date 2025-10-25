"""
Base visual servoing class for ROS1.
Provides RGB and optional depth image subscriptions with thread-safe observation.
"""

import abc
import threading
from typing import Optional, Tuple

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from config.config import RGB_TOPIC_NAME, DEPTH_TOPIC_NAME  

class BaseVisualServoer(abc.ABC):
    """Abstract base class for visual servoing using ROS1."""

    def __init__(self, use_depth: bool = False, silent: bool = False) -> None:
        """
        Initialize subscribers and image buffers.

        Args:
            use_depth: Whether to subscribe to the depth topic.
            silent: Suppress info logs if True.
        """
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

        self.silent = silent
        self.use_depth = use_depth

        # Buffers for images
        self.images = {
            "rgb": None,
            "depth": None,
        }

        # Subscriptions
        self.rgb_sub = rospy.Subscriber(RGB_TOPIC_NAME, Image, self.rgb_image_callback)
        if self.use_depth:
            self.depth_sub = rospy.Subscriber(DEPTH_TOPIC_NAME, Image, self.depth_image_callback)

    # ──────────────────────────────
    # Logging helpers
    # ──────────────────────────────

    def log_info(self, message: str) -> None:
        """Log info messages if not silent."""
        if not self.silent:
            rospy.loginfo(message)

    def log_warn(self, message: str) -> None:
        """Log warning messages."""
        rospy.logwarn(message)

    def log_error(self, message: str) -> None:
        """Log error messages."""
        rospy.logerr(message)

    # ──────────────────────────────
    # ROS Callbacks
    # ──────────────────────────────

    def rgb_image_callback(self, msg: Image) -> None:
        """Handle incoming RGB image."""
        with self.lock:
            try:
                self.images["rgb"] = self.bridge.imgmsg_to_cv2(msg, "rgb8")
                self.condition.notify_all()
            except Exception as e:
                self.log_error(f"Error in rgb_image_callback: {e}")

    def depth_image_callback(self, msg: Image) -> None:
        """Handle incoming depth image."""
        with self.lock:
            try:
                self.images["depth"] = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                self.condition.notify_all()
            except Exception as e:
                self.log_error(f"Error in depth_image_callback: {e}")

    # ──────────────────────────────
    # Core observation logic
    # ──────────────────────────────

    def observe(self, timeout: float = 1.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Wait for new RGB (and optionally depth) images with timeout.

        Args:
            timeout: Max time to wait in seconds.

        Returns:
            (rgb_image, depth_image) or (None, None) on timeout.
        """
        with self.lock:
            # Reset buffers before waiting
            self.images["rgb"] = None
            if self.use_depth:
                self.images["depth"] = None

            success = self.condition.wait_for(
                lambda: self.images["rgb"] is not None and
                        (not self.use_depth or self.images["depth"] is not None),
                timeout=timeout
            )

            if not success:
                self.log_warn(f"Timeout after {timeout:.1f}s while waiting for images.")
                return None, None

            rgb_copy = self.images["rgb"].copy() if self.images["rgb"] is not None else None
            depth_copy = (
                self.images["depth"].copy()
                if self.use_depth and self.images["depth"] is not None
                else None
            )

            # Clear buffers after reading
            self.images["rgb"] = None
            if self.use_depth:
                self.images["depth"] = None

        return rgb_copy, depth_copy

    # ──────────────────────────────
    # Abstract methods
    # ──────────────────────────────

    @abc.abstractmethod
    def run(self) -> None:
        """Run the visual servoing loop (to be implemented by subclasses)."""
        pass


def main() -> None:
    """Simple test loop for image observation."""
    rospy.init_node("base_visual_servoer_test", anonymous=True)

    class TestServoer(BaseVisualServoer):
        def run(self):
            rospy.loginfo("Test servoer running...")

    node = TestServoer(use_depth=True, silent=False)

    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        rgb, depth = node.observe(timeout=1.0)
        if rgb is not None:
            rospy.loginfo(f"RGB shape: {rgb.shape}")
        if depth is not None:
            rospy.loginfo(f"Depth shape: {depth.shape}")
        rate.sleep()


if __name__ == "__main__":
    main()

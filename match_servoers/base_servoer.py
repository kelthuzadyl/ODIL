import abc
import threading
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from config.config import RGB_TOPIC_NAME, DEPTH_TOPIC_NAME


class VisualServoer(Node, abc.ABC):
    """Abstract base class for visual servoing using ROS2."""

    def __init__(self, use_depth: bool = True, silent: bool = False) -> None:
        """
        Initialize the base visual servoing node.

        Args:
            use_depth: Whether to subscribe to the depth topic.
            silent: If True, suppresses info logs.
        """
        super().__init__('visual_servoer')

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.silent = silent
        self.use_depth = use_depth

        # Buffers for latest images
        self.images = {
            "rgb": None,
            "depth": None,
        }

        # Flags and timestamps to track freshness
        self.last_rgb_stamp = None
        self.last_depth_stamp = None
        self.new_rgb_received = False
        self.new_depth_received = False

        # Subscriptions
        self.rgb_subscriber = self.create_subscription(
            Image, RGB_TOPIC_NAME, self.rgb_image_callback, 10
        )

        if self.use_depth:
            self.depth_subscriber = self.create_subscription(
                Image, DEPTH_TOPIC_NAME, self.depth_image_callback, 10
            )

    # ──────────────────────────────
    # Logging helpers
    # ──────────────────────────────

    def log_info(self, message: str) -> None:
        """Log info messages only if not in silent mode."""
        if not self.silent:
            self.get_logger().info(message)

    def log_warn(self, message: str) -> None:
        """Log warning messages."""
        self.get_logger().warning(message)

    def log_error(self, message: str) -> None:
        """Log error messages."""
        self.get_logger().error(message)

    # ──────────────────────────────
    # ROS Callbacks
    # ──────────────────────────────

    def rgb_image_callback(self, msg: Image) -> None:
        """Handle incoming RGB images."""
        with self.lock:
            try:
                self.images["rgb"] = self.bridge.imgmsg_to_cv2(msg, "rgb8")
                self.last_rgb_stamp = msg.header.stamp
                self.new_rgb_received = True
                self.log_info("RGB image received.")
            except Exception as e:
                self.log_error(f"Error in rgb_image_callback: {e}")

    def depth_image_callback(self, msg: Image) -> None:
        """Handle incoming depth images."""
        with self.lock:
            try:
                self.images["depth"] = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                self.last_depth_stamp = msg.header.stamp
                self.new_depth_received = True
                self.log_info("Depth image received.")
            except Exception as e:
                self.log_error(f"Error in depth_image_callback: {e}")

    # ──────────────────────────────
    # Core observation logic
    # ──────────────────────────────

    def observe(self, timeout: float = 5.0) -> Tuple[Optional['np.ndarray'], Optional['np.ndarray']]:
        """
        Block until new RGB (and optionally depth) images are received, or timeout.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            (rgb_image, depth_image): Copies of the latest images.
            Returns (None, None) on timeout or failure.
        """
        self.log_info("Waiting for new images...")

        # Reset freshness flags
        with self.lock:
            self.new_rgb_received = False
            if self.use_depth:
                self.new_depth_received = False

        # Track start time
        start_time = self.get_clock().now()
        timeout_duration = Duration(seconds=timeout)

        # Wait for fresh images
        while rclpy.ok():
            elapsed = self.get_clock().now() - start_time
            if elapsed > timeout_duration:
                self.log_warn(f"Timeout after {timeout:.1f}s while waiting for images.")
                return None, None

            rclpy.spin_once(self, timeout_sec=0.1)

            with self.lock:
                rgb_ready = self.new_rgb_received
                depth_ready = (not self.use_depth) or self.new_depth_received

                if rgb_ready and depth_ready:
                    rgb_copy = self.images["rgb"].copy() if self.images["rgb"] is not None else None
                    depth_copy = (
                        self.images["depth"].copy()
                        if self.use_depth and self.images["depth"] is not None
                        else None
                    )
                    self.log_info("New images received.")
                    return rgb_copy, depth_copy

        self.log_error("ROS context is no longer valid.")
        return None, None

    # ──────────────────────────────
    # Abstract methods
    # ──────────────────────────────

    @abc.abstractmethod
    def run(self) -> None:
        """Run the main visual servoing loop (to be implemented by subclasses)."""
        pass

    # ──────────────────────────────
    # Cleanup
    # ──────────────────────────────

    def destroy_node(self) -> None:
        """Ensure clean shutdown."""
        super().destroy_node()


def main() -> None:
    """Simple test loop for image observation."""
    rclpy.init()

    class TestServoer(VisualServoer):
        def run(self):
            pass  # no-op

    node = TestServoer(use_depth=True, silent=False)

    while rclpy.ok():
        node.observe(timeout=5.0)
        rclpy.spin_once(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

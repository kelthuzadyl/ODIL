import numpy as np
import rclpy

from config.robot import Robot
from vs3d import VisualServoing3D
from vs25d import VisualServoing25D


def run_servoing(node_cls, dir_path: str, bot: Robot, **node_kwargs) -> None:
    """
    Helper to run a visual servoing node with proper setup/teardown.
    """
    node = node_cls(dir_path, bot, **node_kwargs)
    try:
        node.run()
    finally:
        node.destroy_node()


def main(use_25d: bool = False) -> None:
    """
    Run 3D visual servoing and optionally 2.5D refinement.
    After servoing, computes T_delta_robot between current and reference poses.
    """
    rclpy.init()

    dir_path = "example_tasks/pan"
    bot = Robot()
    bot.move_to_default_pose()

    # 3D servoing (with optional UKF)
    run_servoing(
        VisualServoing3D,
        dir_path,
        bot,
        use_4dof_control=False,
        use_ukf=True,
    )

    # Optional 2.5D servoing refinement
    if use_25d:
        run_servoing(
            VisualServoing25D,
            dir_path,
            bot,
            use_4dof_control=True,
        )

    # Compute relative object pose change in robot base frame
    reference_pose = np.load(f"{dir_path}/ref_ee_pose.npy")
    current_pose = bot.bot.arm.get_ee_pose()
    T_delta_robot = current_pose @ np.linalg.inv(reference_pose)

    # ── Apply T_delta_robot to demo waypoints (pseudo-code) ──
    # live_waypoints = [T_delta_robot @ demo_wp for demo_wp in demo_waypoints]
    # bot.execute_waypoints(live_waypoints)

if __name__ == "__main__":
    main(use_25d=True)

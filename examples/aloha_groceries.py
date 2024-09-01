"""An example of using BiGym with pixels for the ALOHA Robot."""
import numpy as np
import time

from bigym.action_modes import AlohaPositionActionMode
from bigym.envs.groceries import GroceriesStoreLower, GroceriesStoreUpper
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.robots.configs.aloha import AlohaRobot  # Import the AlohaRobot class

print("Running 1000 steps with visualization...")
env = GroceriesStoreUpper(
    action_mode=AlohaPositionActionMode(floating_base=False, absolute=False, control_all_joints=True),
    observation_config=ObservationConfig(
        cameras=[
            CameraConfig(name="wrist_cam_left", rgb=True, depth=False, resolution=(128, 128)),
            CameraConfig(name="wrist_cam_right", rgb=True, depth=False, resolution=(128, 128)),
            CameraConfig(name="overhead_cam", rgb=True, depth=False, resolution=(1280, 720)),
        ],
    ),
    render_mode="human",
    robot_cls=AlohaRobot
)

print("Initial robot position:", env.unwrapped._robot._body.get_position())

print("Observation Space:")
print(env.observation_space)
print("Action Space:")
print(env.action_space)

action = env.action_space.sample()
print(f"action: {action}")

env.reset()
for i in range(1000):
    # Create an action array for just the gripper positions
    obs, reward, terminated, truncated, info = env.step(action)
    
    print("Current joint positions:", env.unwrapped._robot._body.get_position())

    # Render the current state
    env.render()
    
    # Add a small delay to make the visualization more visible (optional)
    time.sleep(0.01)
    
    if terminated or truncated:
        env.reset()

env.close()
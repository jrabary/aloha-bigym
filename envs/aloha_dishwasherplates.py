"""An example of using BiGym with pixels for the ALOHA Robot."""
from bigym.action_modes import AlohaPositionActionMode
from bigym.envs.dishwasher_plates import DishwasherUnloadPlates, DishwasherLoadPlates, DishwasherUnloadPlatesLong
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.robots.configs.aloha import AlohaRobot  

print("Running 1000 steps with visualization...")
env = DishwasherUnloadPlatesLong(
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
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        env.reset()

env.close()
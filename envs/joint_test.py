"""An example of using BiGym with pixels for the ALOHA Robot."""
from bigym.action_modes import AlohaPositionActionMode
from bigym.envs.pick_and_place import StoreBox, PickBox
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.robots.configs.aloha import AlohaRobot

def print_joint_positions(env):
    robot = env.unwrapped._robot
    mojo = env.unwrapped._mojo
    print("\nJoint Positions:")
    for i, actuator in enumerate(robot.limb_actuators):
        joint_name = actuator.joint
        joint_pos = mojo.physics.bind(actuator).get_joint_position()
        print(f"{joint_name}: {joint_pos}")

print("Running 1000 steps with visualization...")
env = PickBox(
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
print_joint_positions(env)

print("\nObservation Space:")
print(env.observation_space)
print("\nAction Space:")
print(env.action_space)

action = env.action_space.sample()
print(f"\nSample action: {action}")

print("\nResetting environment...")
env.reset()
print_joint_positions(env)

print("\nRunning simulation...")
for i in range(1000):
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if i % 100 == 0:  # Print joint positions every 100 steps
        print(f"\nStep {i}:")
        print_joint_positions(env)
    
    if terminated or truncated:
        print("\nResetting environment...")
        env.reset()
        print_joint_positions(env)

env.close()
import numpy as np
from bigym.action_modes import AlohaPositionActionMode
from bigym.envs.pick_and_place import PickBox
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.robots.configs.aloha import AlohaRobot
import pygame

def forward_kinematics(joint_positions, joint_angles):
    current_pos = np.array([0., 0., 0.])
    positions = [current_pos]
    
    for i in range(len(joint_angles)):
        rotation = np.array([
            [np.cos(joint_angles[i]), -np.sin(joint_angles[i]), 0],
            [np.sin(joint_angles[i]), np.cos(joint_angles[i]), 0],
            [0, 0, 1]
        ])
        current_pos = current_pos + rotation @ joint_positions[i]
        positions.append(current_pos)
    
    return np.array(positions)

def ccd_ik(target, joint_positions, joint_angles, max_iterations=100, tolerance=1e-3):
    num_joints = len(joint_angles)
    
    for _ in range(max_iterations):
        positions = forward_kinematics(joint_positions, joint_angles)
        end_effector = positions[-1]
        
        if np.linalg.norm(end_effector - target) < tolerance:
            break
        
        for i in reversed(range(num_joints)):
            current_pos = positions[i]
            to_end = end_effector - current_pos
            to_target = target - current_pos
            
            rotation_axis = np.cross(to_end, to_target)
            if np.linalg.norm(rotation_axis) < 1e-5:
                continue
            
            rotation_axis /= np.linalg.norm(rotation_axis)
            angle = np.arctan2(np.linalg.norm(np.cross(to_end, to_target)), np.dot(to_end, to_target))
            
            rotation = np.array([
                [np.cos(angle) + rotation_axis[0]**2 * (1 - np.cos(angle)),
                 rotation_axis[0] * rotation_axis[1] * (1 - np.cos(angle)) - rotation_axis[2] * np.sin(angle),
                 rotation_axis[0] * rotation_axis[2] * (1 - np.cos(angle)) + rotation_axis[1] * np.sin(angle)],
                [rotation_axis[1] * rotation_axis[0] * (1 - np.cos(angle)) + rotation_axis[2] * np.sin(angle),
                 np.cos(angle) + rotation_axis[1]**2 * (1 - np.cos(angle)),
                 rotation_axis[1] * rotation_axis[2] * (1 - np.cos(angle)) - rotation_axis[0] * np.sin(angle)],
                [rotation_axis[2] * rotation_axis[0] * (1 - np.cos(angle)) - rotation_axis[1] * np.sin(angle),
                 rotation_axis[2] * rotation_axis[1] * (1 - np.cos(angle)) + rotation_axis[0] * np.sin(angle),
                 np.cos(angle) + rotation_axis[2]**2 * (1 - np.cos(angle))]
            ])
            
            for j in range(i+1, num_joints):
                joint_positions[j] = rotation @ joint_positions[j]
            
            joint_angles[i] += angle
    
    return joint_angles

def get_user_input():
    keys = pygame.key.get_pressed()
    left_arm_movement = np.zeros(3)
    right_arm_movement = np.zeros(3)
    left_gripper = 0
    right_gripper = 0
    
    if keys[pygame.K_q]: left_arm_movement[0] += 1
    if keys[pygame.K_a]: left_arm_movement[0] -= 1
    if keys[pygame.K_w]: left_arm_movement[1] += 1
    if keys[pygame.K_s]: left_arm_movement[1] -= 1
    if keys[pygame.K_e]: left_arm_movement[2] += 1
    if keys[pygame.K_d]: left_arm_movement[2] -= 1
    if keys[pygame.K_r]: left_gripper += 1
    if keys[pygame.K_f]: left_gripper -= 1
    if keys[pygame.K_u]: right_arm_movement[0] += 1
    if keys[pygame.K_j]: right_arm_movement[0] -= 1
    if keys[pygame.K_i]: right_arm_movement[1] += 1
    if keys[pygame.K_k]: right_arm_movement[1] -= 1
    if keys[pygame.K_o]: right_arm_movement[2] += 1
    if keys[pygame.K_l]: right_arm_movement[2] -= 1
    if keys[pygame.K_p]: right_gripper += 1
    if keys[pygame.K_SEMICOLON]: right_gripper -= 1
    
    return left_arm_movement, right_arm_movement, left_gripper, right_gripper

def main():
    pygame.init()
    screen = pygame.display.set_mode((1, 1))
    
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
    
    obs = env.reset()
    
    # Initialize joint positions and angles for both arms
    left_joint_positions = [np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([0, 0, 1])]
    right_joint_positions = [np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([0, 0, 1])]
    left_joint_angles = [0, 0, 0]
    right_joint_angles = [0, 0, 0]
    
    movement_scale = 0.01  # Scale factor for movement
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        left_movement, right_movement, left_gripper, right_gripper = get_user_input()
        
        left_target = forward_kinematics(left_joint_positions, left_joint_angles)[-1] + left_movement * movement_scale
        right_target = forward_kinematics(right_joint_positions, right_joint_angles)[-1] + right_movement * movement_scale
        
        left_joint_angles = ccd_ik(left_target, left_joint_positions, left_joint_angles)
        right_joint_angles = ccd_ik(right_target, right_joint_positions, right_joint_angles)
        
        action = np.concatenate([left_joint_angles, right_joint_angles, [left_gripper, right_gripper], [0, 0, 0, 0, 0.03, 0.03]])
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            obs = env.reset()
    
    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()
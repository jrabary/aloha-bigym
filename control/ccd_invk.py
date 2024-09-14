import numpy as np

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

# joint_pos = [np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([0, 0, 1])]
# initial_angles = [0, 0, 0]
# target = np.array([1, 1, 2])

# final_angles = ccd_ik(target, joint_pos, initial_angles)
# final_position = forward_kinematics(joint_pos, final_angles)[-1]

# print("Target position:", target)
# print("Achieved position:", final_position)
# print("Joint angles:", final_angles)
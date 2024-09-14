import numpy as np

#counterclockwise rot
def rotation_matrix(axis, theta):

    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def forward_kinematics(theta, robot_info, arm='left'):
    T = np.eye(4)
    
    base_pos = robot_info['links'][f'{arm}_base_link']['pos']
    T[:3, 3] = base_pos
    
    links = [f'{arm}_shoulder_link', f'{arm}_upper_arm_link', f'{arm}_upper_forearm_link', 
             f'{arm}_lower_forearm_link', f'{arm}_wrist_link', f'{arm}_gripper_link']
    joints = [f'{arm}_waist', f'{arm}_shoulder', f'{arm}_elbow', 
              f'{arm}_forearm_roll', f'{arm}_wrist_angle', f'{arm}_wrist_rotate']
    
    for i, (link, joint) in enumerate(zip(links, joints)):
        link_pos = robot_info['links'][link]['pos']
        joint_axis = robot_info['joints'][joint]['axis']
        
        R = rotation_matrix(joint_axis, theta[i])
        T_rot = np.eye(4)
        T_rot[:3, :3] = R
        T = T @ T_rot
        T_trans = np.eye(4)
        T_trans[:3, 3] = link_pos
        T = T @ T_trans
    
    return T[:3, 3] 

def jacobian(theta, robot_info, arm='left'):

    epsilon = 1e-6
    J = np.zeros((3, 6))
    
    for i in range(6):
        theta_plus = theta.copy()
        theta_plus[i] += epsilon
        theta_minus = theta.copy()
        theta_minus[i] -= epsilon
        
        pos_plus = forward_kinematics(theta_plus, robot_info, arm)
        pos_minus = forward_kinematics(theta_minus, robot_info, arm)
        
        J[:, i] = (pos_plus - pos_minus) / (2 * epsilon)
    
    return J

def inverse_kinematics(target_pos, current_angles, robot_info, arm='left', max_iterations=100, tolerance=1e-3):

    theta = np.array(current_angles)
    
    for _ in range(max_iterations):
        current_pos = forward_kinematics(theta, robot_info, arm)
        error = target_pos - current_pos
        
        if np.linalg.norm(error) < tolerance:
            break
        
        J = jacobian(theta, robot_info, arm)
        J_inv = np.linalg.pinv(J)
        
        delta_theta = J_inv @ error
        theta += delta_theta
        
        for i, joint_name in enumerate([f'{arm}_waist', f'{arm}_shoulder', f'{arm}_elbow', 
                                        f'{arm}_forearm_roll', f'{arm}_wrist_angle', f'{arm}_wrist_rotate']):
            joint_info = robot_info['joints'][joint_name]
            theta[i] = np.clip(theta[i], joint_info['range'][0], joint_info['range'][1])
    
    return theta

# robot_info = {'joints': {None: {'type': 'hinge', 'axis': np.array([1., 0., 0.]), 'range': [-3.14158, 3.14158]}, 'left_waist': {'type': 'hinge', 'axis': np.array([0., 0., 1.]), 'range': [-3.14159, 3.14159]}, 'left_shoulder': {'type': 'hinge', 'axis': np.array([0., 0., 1.]), 'range': [-3.14159, 3.14159]}, 'left_elbow': {'type': 'hinge', 'axis': np.array([0., 0., 1.]), 'range': [-3.14159, 3.14159]}, 'left_forearm_roll': {'type': 'hinge', 'axis': np.array([0., 0., 1.]), 'range': [-3.14159, 3.14159]}, 'left_wrist_angle': {'type': 'hinge', 'axis': np.array([0., 0., 1.]), 'range': [-3.14159, 3.14159]}, 'left_wrist_rotate': {'type': 'hinge', 'axis': np.array([0., 0., 1.]), 'range': [-3.14159, 3.14159]}, 'right_waist': {'type': 'hinge', 'axis': np.array([0., 0., 1.]), 'range': [-3.14159, 3.14159]}, 'right_shoulder': {'type': 'hinge', 'axis': np.array([0., 0., 1.]), 'range': [-3.14159, 3.14159]}, 'right_elbow': {'type': 'hinge', 'axis': np.array([0., 0., 1.]), 'range': [-3.14159, 3.14159]}, 'right_forearm_roll': {'type': 'hinge', 'axis': np.array([0., 0., 1.]), 'range': [-3.14159, 3.14159]}, 'right_wrist_angle': {'type': 'hinge', 'axis': np.array([0., 0., 1.]), 'range': [-3.14159, 3.14159]}, 'right_wrist_rotate': {'type': 'hinge', 'axis': np.array([0., 0., 1.]), 'range': [-3.14159, 3.14159]}}, 'links': {'left_base_link': {'pos': np.array([-0.469, -0.019,  0.75 ])}, 'left_shoulder_link': {'pos': np.array([0.   , 0.   , 0.079])}, 'left_upper_arm_link': {'pos': np.array([0.     , 0.     , 0.04805])}, 'left_upper_forearm_link': {'pos': np.array([0.05955, 0.     , 0.3    ])}, 'left_lower_forearm_link': {'pos': np.array([0.2, 0. , 0. ])}, 'left_wrist_link': {'pos': np.array([0.1, 0. , 0. ])}, 'left_gripper_link': {'pos': np.array([0.069744, 0.      , 0.      ])}, 'right_base_link': {'pos': np.array([ 0.469, -0.019,  0.75 ])}, 'right_shoulder_link': {'pos': np.array([0.   , 0.   , 0.079])}, 'right_upper_arm_link': {'pos': np.array([0.     , 0.     , 0.04805])}, 'right_upper_forearm_link': {'pos': np.array([0.05955, 0.     , 0.3    ])}, 'right_lower_forearm_link': {'pos': np.array([0.2, 0. , 0. ])}, 'right_wrist_link': {'pos': np.array([0.1, 0. , 0. ])}, 'right_gripper_link': {'pos': np.array([0.069744, 0.      , 0.      ])}, 'center': {'pos': np.array([0.  , 0.  , 0.75])}}}
# target_position = np.array([0.5, 0.3, 1.2])  
# current_angles = np.zeros(6)  
# new_angles = inverse_kinematics(target_position, current_angles, robot_info, arm='left')
# print("Target position:", target_position)
# print("Calculated joint angles:", new_angles)
# print("Achieved end-effector position:", forward_kinematics(new_angles, robot_info, arm='left'))
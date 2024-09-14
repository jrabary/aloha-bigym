# import numpy as np
# import mujoco
# import mujoco.viewer
# import mink
# from bigym.envs.pick_and_place import PickBox
# from bigym.action_modes import AlohaPositionActionMode
# from bigym.utils.observation_config import ObservationConfig, CameraConfig
# from bigym.robots.configs.aloha import AlohaRobot
# from bigym.envs.manipulation import StackBlocks

# from reduced_configuration import ReducedConfiguration


# from pathlib import Path
# from loop_rate_limiters import RateLimiter

# _JOINT_NAMES = [
#     "waist",
#     "shoulder",
#     "elbow",
#     "forearm_roll",
#     "wrist_angle",
#     "wrist_rotate",
# ]

# # Single arm velocity limits, taken from:
# # https://github.com/Interbotix/interbotix_ros_manipulators/blob/main/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/vx300s.urdf.xacro
# _VELOCITY_LIMITS = {k: np.pi for k in _JOINT_NAMES}

# class AlohaMocapControl:
#     def __init__(self):
#         self.env = PickBox(
#             action_mode=AlohaPositionActionMode(floating_base=False, absolute=False, control_all_joints=True),
#             observation_config=ObservationConfig(
#                 cameras=[
#                     CameraConfig(name="overhead_cam", rgb=True, depth=False, resolution=(1280, 720)),
#                 ],
#             ),
#             render_mode="human",
#             robot_cls=AlohaRobot
#         )
#         self.env.reset()
        
#         self.model = self.env.unwrapped._mojo.model
#         self.data = self.env.unwrapped._mojo.data
        
#     def run(self):
#         model = self.model
#         data = self.data

#         joint_names: list[str] = []
#         velocity_limits: dict[str, float] = {}
#         for prefix in ["aloha_scene/left", "aloha_scene/right"]:
#             for n in _JOINT_NAMES:
#                 name = f"{prefix}_{n}"
#                 joint_names.append(name)
#                 velocity_limits[name] = _VELOCITY_LIMITS[n]

#         dof_ids = np.array([model.joint(name).id for name in joint_names])
#         actuator_ids = np.array([model.actuator(name).id for name in joint_names])

#         # Configuration
#         # configuration = mink.Configuration(model)
#         relevant_qpos_indices = np.array([model.jnt_qposadr[model.joint(name).id] for name in joint_names])
#         relevant_qvel_indices = np.array([model.jnt_dofadr[model.joint(name).id] for name in joint_names])
#         print(f"Relevant qpos indices: {relevant_qpos_indices}")
#         print(f"Relevant qvel indices: {relevant_qvel_indices}")

#         configuration = ReducedConfiguration(model, relevant_qpos_indices, relevant_qvel_indices)
        
#         l_ee_task = mink.FrameTask(
#                 frame_name="aloha_scene/left_gripper",
#                 frame_type="site",
#                 position_cost=1.0,
#                 orientation_cost=1.0,
#                 lm_damping=1.0,
#             )
        
#         r_ee_task = mink.FrameTask(
#                 frame_name="aloha_scene/right_gripper",
#                 frame_type="site",
#                 position_cost=1.0,
#                 orientation_cost=1.0,
#                 lm_damping=1.0,
#             )

#         tasks = [
#             l_ee_task,
#             r_ee_task
#         ]

#         l_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("aloha_scene/left_wrist_link").id)
#         r_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("aloha_scene/right_wrist_link").id)
#         frame_geoms = mink.get_body_geom_ids(model, model.body("aloha_scene/metal_frame").id)
#         collision_pairs = [
#             (l_wrist_geoms, r_wrist_geoms),
#             (l_wrist_geoms + r_wrist_geoms, frame_geoms + ["aloha_scene/table"]),
#         ]

#         collision_avoidance_limit = mink.CollisionAvoidanceLimit(
#             model=model,
#             geom_pairs=collision_pairs,  # type: ignore
#             minimum_distance_from_collisions=0.05,
#             collision_detection_distance=0.1,
#         )

#         limits = [
#             # mink.ConfigurationLimit(model=model),
#             mink.VelocityLimit(model, velocity_limits),
#             # collision_avoidance_limit,
#         ]

#         # solver = "quadprog"
#         solver = "osqp"
#         pos_threshold = 0.01
#         ori_threshold = 0.01
#         max_iters = 20

#         with mujoco.viewer.launch_passive(
#             model=model, data=data, show_left_ui=False, show_right_ui=False
#         ) as viewer:
#             mujoco.mjv_defaultFreeCamera(model, viewer.cam)

#             self.add_target_sites()
#             mujoco.mj_forward(model, data)

#             target_l, target_r = self.generate_random_targets()

#             l_ee_task.set_target(mink.SE3.from_translation(target_l))
#             r_ee_task.set_target(mink.SE3.from_translation(target_r))

#             self.update_target_sites(target_l, target_r)
            
#             rate = RateLimiter(frequency=200.0)
#             while viewer.is_running():
#                 jacobian = configuration.get_frame_jacobian("aloha_scene/left_gripper", "site")
#                 print("Jacobian Matrix:\n", jacobian)
#                 condition_number = np.linalg.cond(jacobian)
#                 print("Jacobian Condition Number:", condition_number)

#                 # Compute velocity and integrate into the next configuration.
#                 for i in range(max_iters):
#                     # print(f"Left gripper position: {data.site_xpos[model.site('aloha_scene/left_gripper').id]}")
#                     # print(f"Right gripper position: {data.site_xpos[model.site('aloha_scene/right_gripper').id]}")
#                     # print(f"Target position: {self.data.site_xpos[self.target_site_id]}")
#                     # configuration.q[:] = data.qpos

#                     vel = mink.solve_ik(
#                         configuration,
#                         tasks,
#                         rate.dt,
#                         solver,
#                         limits=limits,
#                         damping=1e-3,
#                     )

#                     print(f"Velocity: {vel}")
#                     configuration.integrate_inplace(vel, rate.dt)

#                     print(f"self qpos: {self.env.unwrapped._mojo.data.qpos}")

#                     # l_err = l_ee_task.compute_error(configuration)
#                     # # print(f"Left error: {l_err}")
#                     # l_pos_achieved = np.linalg.norm(l_err[:3]) < pos_threshold
#                     # l_ori_achieved = np.linalg.norm(l_err[3:]) < ori_threshold

#                     # r_err = r_ee_task.compute_error(configuration)  
#                     # # print(f"Right error: {r_err}")
#                     # r_pos_achieved = np.linalg.norm(r_err[:3]) < pos_threshold
#                     # r_ori_achieved = np.linalg.norm(r_err[3:]) < ori_threshold
#                     left_gripper_pos = data.site_xpos[model.site('aloha_scene/left_gripper').id]
#                     right_gripper_pos = data.site_xpos[model.site('aloha_scene/right_gripper').id]
#                     target_pos = self.data.site_xpos[self.target_site_id]

#                     left_gripper_rot = data.site_xmat[model.site('aloha_scene/left_gripper').id]
#                     right_gripper_rot = data.site_xmat[model.site('aloha_scene/right_gripper').id]
#                     target_rot = self.data.site_xmat[self.target_site_id]

#                     l_pos_achieved = np.linalg.norm(left_gripper_pos - target_pos) < pos_threshold
#                     r_pos_achieved = np.linalg.norm(right_gripper_pos - target_pos) < pos_threshold

#                     l_ori_achieved = np.linalg.norm(left_gripper_rot - target_rot) < ori_threshold
#                     r_ori_achieved = np.linalg.norm(right_gripper_rot - target_rot) < ori_threshold

#                     print(f"norm of left dist: {np.linalg.norm(left_gripper_pos - target_pos)}")

#                     if (l_pos_achieved and l_ori_achieved and r_pos_achieved and r_ori_achieved):
#                         print(f"Target reached after {i} iterations.")
#                         break
                    
#                 data.ctrl[actuator_ids] = configuration.q[dof_ids]
               
#                 # Step the simulation
#                 mujoco.mj_step(model, data)
#                 print(f"Control inputs (data.ctrl): {data.ctrl[actuator_ids]}")
#                 print(f"Actuator forces (data.actuator_force): {data.actuator_force[actuator_ids]}")

#                 # print(f"control choice: {data.ctrl[actuator_ids]}")
#                 print(f"model: {model}")
#                 mujoco.mj_step(model, data)

#                 # Visualize at fixed FPS.
#                 viewer.sync()
#                 rate.sleep()

#     def generate_random_targets(self):
#         # Define the workspace limits for the ALOHA arms
#         x_range = (-0.4, -0.1)
#         y_range = (-0.2, 0.2)
#         z_range = (1, 1.2)  #height

#         x = np.random.uniform(*x_range)
#         y = np.random.uniform(*y_range)
#         z = np.random.uniform(*z_range)

#         x_range = (0.1, 0.4)
#         y_range = (-0.2, 0.2)
#         z_range = (1, 1.2)  #height

#         x2 = np.random.uniform(*x_range)
#         y2 = np.random.uniform(*y_range)
#         z2 = np.random.uniform(*z_range)

#         return np.array([x, y, z]), np.array([x2, y2, z2])
    
#     def add_target_sites(self):
#         target_l, target_r = self.generate_random_targets()
#         self.target_site_id = self.model.site('aloha_scene/target').id
#         self.target_site_id2 = self.model.site('aloha_scene/target2').id
#         self.update_target_sites(target_l, target_r)

#     def update_target_sites(self, target_l, target_r):
#         self.data.site_xpos[self.target_site_id] = target_l
#         self.model.site_pos[self.target_site_id] = target_l
#         self.data.site_xpos[self.target_site_id2] = target_r
#         self.model.site_pos[self.target_site_id2] = target_r

# if __name__ == "__main__":
#     controller = AlohaMocapControl()
#     controller.run()

import numpy as np
import mujoco
import mujoco.viewer
import mink
from bigym.envs.pick_and_place import PickBox
from bigym.action_modes import AlohaPositionActionMode
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.robots.configs.aloha import AlohaRobot
from mink import SO3

from reduced_configuration import ReducedConfiguration
from loop_rate_limiters import RateLimiter

# Joint names for both arms
_JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]

# Velocity limits for the joints
_VELOCITY_LIMITS = {k: np.pi for k in _JOINT_NAMES}

class AlohaMocapControl:
    def __init__(self):
        self.env = PickBox(
            action_mode=AlohaPositionActionMode(floating_base=False, absolute=False, control_all_joints=True),
            observation_config=ObservationConfig(
                cameras=[
                    CameraConfig(name="overhead_cam", rgb=True, depth=False, resolution=(1280, 720)),
                ],
            ),
            render_mode="human",
            robot_cls=AlohaRobot
        )
        self.env.reset()
        
        self.model = self.env.unwrapped._mojo.model
        self.data = self.env.unwrapped._mojo.data
        
    def run(self):
        model = self.model
        data = self.data

        # Collect joint names and velocity limits for both arms
        joint_names: list[str] = []
        velocity_limits: dict[str, float] = {}
        for prefix in ["aloha_scene/left", "aloha_scene/right"]:
            for n in _JOINT_NAMES:
                name = f"{prefix}_{n}"
                joint_names.append(name)
                velocity_limits[name] = _VELOCITY_LIMITS[n]

        dof_ids = np.array([model.joint(name).id for name in joint_names])
        actuator_ids = np.array([model.actuator(name).id for name in joint_names])

        # Configuration
        relevant_qpos_indices = np.array([model.jnt_qposadr[model.joint(name).id] for name in joint_names])
        relevant_qvel_indices = np.array([model.jnt_dofadr[model.joint(name).id] for name in joint_names])
        print(f"Relevant qpos indices: {relevant_qpos_indices}")
        print(f"Relevant qvel indices: {relevant_qvel_indices}")

        configuration = ReducedConfiguration(model, relevant_qpos_indices, relevant_qvel_indices)
        
        # Define tasks for both arms
        l_ee_task = mink.FrameTask(
                frame_name="aloha_scene/left_gripper",
                frame_type="site",
                position_cost=1.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            )
        
        r_ee_task = mink.FrameTask(
                frame_name="aloha_scene/right_gripper",
                frame_type="site",
                position_cost=1.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            )

        tasks = [
            l_ee_task,
            r_ee_task
        ]

        # Define limits (you can adjust or uncomment configuration limit as needed)
        limits = [
            # mink.ConfigurationLimit(model=model),
            mink.VelocityLimit(model, velocity_limits),
            # collision_avoidance_limit,
        ]

        solver = "osqp"
        pos_threshold = 0.01
        ori_threshold = 0.01
        max_iters = 20

        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            # Add target sites and generate random targets (positions and rotations)
            self.add_target_sites()
            mujoco.mj_forward(model, data)

            target_l, target_r, rot_l, rot_r = self.generate_random_targets()

            # Set target positions and orientations for the tasks
            l_target_pose = mink.SE3.from_rotation_and_translation(rot_l, target_l)
            r_target_pose = mink.SE3.from_rotation_and_translation(rot_r, target_r)

            l_ee_task.set_target(l_target_pose)
            r_ee_task.set_target(r_target_pose)

            # Update target sites for visualization
            self.update_target_sites(target_l, target_r, rot_l, rot_r)

            # Get current positions and orientations
            left_gripper_pos = data.site_xpos[model.site('aloha_scene/left_gripper').id]
            right_gripper_pos = data.site_xpos[model.site('aloha_scene/right_gripper').id]
            left_gripper_rot = data.site_xmat[model.site('aloha_scene/left_gripper').id].reshape(3, 3)
            right_gripper_rot = data.site_xmat[model.site('aloha_scene/right_gripper').id].reshape(3, 3)

            # Compute position and orientation errors
            l_pos_error = np.linalg.norm(left_gripper_pos - target_l)
            r_pos_error = np.linalg.norm(right_gripper_pos - target_r)

            l_ori_error = np.linalg.norm(left_gripper_rot - rot_l.as_matrix())
            r_ori_error = np.linalg.norm(right_gripper_rot - rot_r.as_matrix())

            print(f"Left position error: {l_pos_error}")
            print(f"Right position error: {r_pos_error}")
            print(f"Left orientation error: {l_ori_error}")
            print(f"Right orientation error: {r_ori_error}")
            print(f"Left gripper position: {left_gripper_pos}")
            print(f"Right gripper position: {right_gripper_pos}")
            print(f"Left gripper orientation: {left_gripper_rot}")
            print(f"Right gripper orientation: {right_gripper_rot}")
            print(f"Target l position: {target_l}")
            print(f"Target l orientation: {rot_l.as_matrix()}")
            print(f"Target r position: {target_r}")
            print(f"Target r orientation: {rot_r.as_matrix()}")

            rate = RateLimiter(frequency=200.0)
            while viewer.is_running():
                jacobian = configuration.get_frame_jacobian("aloha_scene/left_gripper", "site")
                print("Jacobian Matrix:\n", jacobian)
                condition_number = np.linalg.cond(jacobian)
                print("Jacobian Condition Number:", condition_number)

                # Compute velocity and integrate into the next configuration.
                for i in range(max_iters):
                    vel = mink.solve_ik(
                        configuration,
                        tasks,
                        rate.dt,
                        solver,
                        limits=None,
                        damping=1e-3,
                    )

                    print(f"Velocity: {vel}")
                    configuration.integrate_inplace(vel, rate.dt)

                    # Get current positions and orientations
                    left_gripper_pos = data.site_xpos[model.site('aloha_scene/left_gripper').id]
                    right_gripper_pos = data.site_xpos[model.site('aloha_scene/right_gripper').id]
                    left_gripper_rot = data.site_xmat[model.site('aloha_scene/left_gripper').id].reshape(3, 3)
                    right_gripper_rot = data.site_xmat[model.site('aloha_scene/right_gripper').id].reshape(3, 3)

                    # Compute position and orientation errors
                    l_pos_error = np.linalg.norm(left_gripper_pos - target_l)
                    r_pos_error = np.linalg.norm(right_gripper_pos - target_r)

                    l_ori_error = np.linalg.norm(left_gripper_rot - rot_l.as_matrix())
                    r_ori_error = np.linalg.norm(right_gripper_rot - rot_r.as_matrix())

                    print(f"Left position error: {l_pos_error}")
                    print(f"Right position error: {r_pos_error}")
                    print(f"Left orientation error: {l_ori_error}")
                    print(f"Right orientation error: {r_ori_error}")
                    print(f"Left gripper position: {left_gripper_pos}")
                    print(f"Right gripper position: {right_gripper_pos}")
                    print(f"Left gripper orientation: {left_gripper_rot}")
                    print(f"Right gripper orientation: {right_gripper_rot}")
                    print(f"Target l position: {target_l}")
                    print(f"Target l orientation: {rot_l.as_matrix()}")
                    print(f"Target r position: {target_r}")
                    print(f"Target r orientation: {rot_r.as_matrix()}")

                    l_pos_achieved = l_pos_error < pos_threshold
                    r_pos_achieved = r_pos_error < pos_threshold

                    l_ori_achieved = l_ori_error < ori_threshold
                    r_ori_achieved = r_ori_error < ori_threshold

                    if (l_pos_achieved and l_ori_achieved and r_pos_achieved and r_ori_achieved):
                        print(f"Target reached after {i} iterations.")
                        break
                        
                # Apply the computed joint positions to the actuators
                data.ctrl[actuator_ids] = configuration.q[dof_ids]
               
                # Step the simulation
                mujoco.mj_step(model, data)
                print(f"Control inputs (data.ctrl): {data.ctrl[actuator_ids]}")
                print(f"Actuator forces (data.actuator_force): {data.actuator_force[actuator_ids]}")

                # Visualize at fixed FPS.
                viewer.sync()
                rate.sleep()

    def generate_random_targets(self):
        # Define the workspace limits for the left arm
        x_range_left = (-0.4, -0.1)
        y_range_left = (-0.2, 0.2)
        z_range_left = (1.0, 1.2)

        # Random position for left target
        x_l = np.random.uniform(*x_range_left)
        y_l = np.random.uniform(*y_range_left)
        z_l = np.random.uniform(*z_range_left)
        target_l = np.array([x_l, y_l, z_l])

        # Define the workspace limits for the right arm
        x_range_right = (0.1, 0.4)
        y_range_right = (-0.2, 0.2)
        z_range_right = (1.0, 1.2)

        # Random position for right target
        x_r = np.random.uniform(*x_range_right)
        y_r = np.random.uniform(*y_range_right)
        z_r = np.random.uniform(*z_range_right)
        target_r = np.array([x_r, y_r, z_r])

        # rot_l = np.array([np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi)])
        # rot_r = np.array([np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi)])

#         Left gripper orientation: [[ 0.98594576  0.16076923 -0.04543352]
#  [ 0.16076923 -0.83907153  0.51972322]
#  [ 0.04543352 -0.51972322 -0.85312577]]
# Right gripper orientation: [[-0.69831662  0.47513026  0.53535514]
#  [ 0.24867168 -0.54030231  0.80388794]
#  [ 0.6712051   0.69449597  0.25915064]]

        # rot_l = SO3.sample_uniform()
        # rot_r = SO3.sample_uniform()

        # rot_l = SO3.from_matrix(np.array([[ 0.98287616,  0.00586364,  0.18417404], [ 0.00929491, -0.99979885, -0.01777276], [ 0.18403278,  0.01918031, -0.98273295]]))
        # rot_r = SO3.from_matrix(np.array([[-0.87198589, -0.48937584, -0.01232488], [ 0.0654842,  -0.14155857,  0.98776161], [-0.48513136,  0.86050709,  0.15548347]]))

        rot_l = SO3.from_matrix(np.array([[ 0.98594576,  0.16076923, -0.04543352], [ 0.16076923, -0.83907153,  0.51972322], [ 0.04543352, -0.51972322, -0.85312577]]))
        rot_r = SO3.from_matrix(np.array([[-0.69831662,  0.47513026,  0.53535514], [ 0.24867168, -0.54030231,  0.80388794], [ 0.6712051,   0.69449597,  0.25915064]]))

        return target_l, target_r, rot_l, rot_r
        
    def add_target_sites(self):
        target_l, target_r, rot_l, rot_r = self.generate_random_targets()
        self.target_site_id_l = self.model.site('aloha_scene/target').id
        self.target_site_id_r = self.model.site('aloha_scene/target2').id
        self.update_target_sites(target_l, target_r, rot_l, rot_r)

    def update_target_sites(self, target_l, target_r, rot_l, rot_r):
        # Update positions
        self.data.site_xpos[self.target_site_id_l] = target_l
        self.model.site_pos[self.target_site_id_l] = target_l
        self.data.site_xpos[self.target_site_id_r] = target_r
        self.model.site_pos[self.target_site_id_r] = target_r

        # Update orientations
        # self.data.site_xmat[self.target_site_id_l] = rot_l
        # self.model.site_quat[self.target_site_id_l] = rot_l
        # self.data.site_xmat[self.target_site_id_r] = rot_r
        # self.model.site_quat[self.target_site_id_r] = rot_r

        rot_l_matrix_flat = rot_l.as_matrix().flatten()
        rot_r_matrix_flat = rot_r.as_matrix().flatten()

        self.data.site_xmat[self.target_site_id_l] = rot_l_matrix_flat
        # self.model.site_mat[self.target_site_id_l] = rot_l_matrix_flat
        self.data.site_xmat[self.target_site_id_r] = rot_r_matrix_flat
        # self.model.site_mat[self.target_site_id_r] = rot_r_matrix_flat

if __name__ == "__main__":
    controller = AlohaMocapControl()
    controller.run()

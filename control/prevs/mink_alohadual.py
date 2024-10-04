import numpy as np
import mujoco
import mujoco.viewer
import mink
from bigym.envs.pick_and_place import PickBox, TakeCups
from bigym.envs.manipulation import StackBlocks
from bigym.action_modes import AlohaPositionActionMode
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.robots.configs.aloha import AlohaRobot
from mink import SO3

from reduced_configuration import ReducedConfiguration, ReducedConfigurationLimit, ReducedVelocityLimit
from loop_rate_limiters import RateLimiter
import time

_JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]

_VELOCITY_LIMITS = {k: np.pi for k in _JOINT_NAMES}

class AlohaMocapControl:
    def __init__(self):
        self.env = TakeCups(
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

        self.left_gripper_actuator_id = self.model.actuator("aloha_scene/aloha_gripper_left/gripper_actuator").id
        self.right_gripper_actuator_id = self.model.actuator("aloha_scene/aloha_gripper_right/gripper_actuator").id

    def control_gripper(self, left_gripper_position, right_gripper_position):
        # Args: gripper_position (float): A value between 0.002 (closed) and 0.037 (open).
        left_gripper_position = np.clip(left_gripper_position, 0.002, 0.037)
        right_gripper_position = np.clip(right_gripper_position, 0.002, 0.037)
        self.data.ctrl[self.left_gripper_actuator_id] = left_gripper_position
        self.data.ctrl[self.right_gripper_actuator_id] = right_gripper_position
        
    def run(self):
        model = self.model
        data = self.data

        self.control_gripper(0.037, 0.037)

        left_joint_names = []
        right_joint_names = []
        velocity_limits = {}

        for n in _JOINT_NAMES:
            name_left = f"aloha_scene/left_{n}"
            name_right = f"aloha_scene/right_{n}"
            left_joint_names.append(name_left)
            right_joint_names.append(name_right)
            velocity_limits[name_left] = _VELOCITY_LIMITS[n]
            velocity_limits[name_right] = _VELOCITY_LIMITS[n]

        left_dof_ids = np.array([model.joint(name).id for name in left_joint_names])
        left_actuator_ids = np.array([model.actuator(name).id for name in left_joint_names])
        left_relevant_qpos_indices = np.array([model.jnt_qposadr[model.joint(name).id] for name in left_joint_names])
        left_relevant_qvel_indices = np.array([model.jnt_dofadr[model.joint(name).id] for name in left_joint_names])
        left_configuration = ReducedConfiguration(model, data, left_relevant_qpos_indices, left_relevant_qvel_indices)

        right_dof_ids = np.array([model.joint(name).id for name in right_joint_names])
        right_actuator_ids = np.array([model.actuator(name).id for name in right_joint_names])
        right_relevant_qpos_indices = np.array([model.jnt_qposadr[model.joint(name).id] for name in right_joint_names])
        right_relevant_qvel_indices = np.array([model.jnt_dofadr[model.joint(name).id] for name in right_joint_names])
        right_configuration = ReducedConfiguration(model, data, right_relevant_qpos_indices, right_relevant_qvel_indices)

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

        l_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("aloha_scene/left_wrist_link").id)
        r_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("aloha_scene/right_wrist_link").id)
        frame_geoms = mink.get_body_geom_ids(model, model.body("aloha_scene/metal_frame").id)
        collision_pairs = [
            (l_wrist_geoms, r_wrist_geoms),
            (l_wrist_geoms + r_wrist_geoms, frame_geoms + ["aloha_scene/table"]),
        ]

        collision_avoidance_limit = mink.CollisionAvoidanceLimit(
            model=model,
            geom_pairs=collision_pairs,  
            minimum_distance_from_collisions=0.1,
            collision_detection_distance=0.1,
        )

        left_limit = [
            # ReducedConfigurationLimit(model, left_relevant_qpos_indices),
            # ReducedVelocityLimit(model, left_relevant_qvel_indices, velocity_limits),
            collision_avoidance_limit,
        ]

        right_limit = [
            # ReducedConfigurationLimit(model, right_relevant_qpos_indices),
            # ReducedVelocityLimit(model, right_relevant_qpos_indices, velocity_limits),
            collision_avoidance_limit,
        ]

        solver = "osqp"
        pos_threshold = 0.1
        ori_threshold = 0.1
        max_iters = 20

        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            self.add_target_sites()
            mujoco.mj_forward(model, data)

            target_l, target_r, rot_l, rot_r = self.generate_random_targets()

            l_target_pose = mink.SE3.from_rotation_and_translation(rot_l, target_l)
            r_target_pose = mink.SE3.from_rotation_and_translation(rot_r, target_r)

            l_ee_task.set_target(l_target_pose)
            r_ee_task.set_target(r_target_pose)

            self.update_target_sites(target_l, target_r, rot_l, rot_r)

            left_gripper_pos = data.site_xpos[model.site('aloha_scene/left_gripper').id]
            right_gripper_pos = data.site_xpos[model.site('aloha_scene/right_gripper').id]
            left_gripper_rot = data.site_xmat[model.site('aloha_scene/left_gripper').id].reshape(3, 3)
            right_gripper_rot = data.site_xmat[model.site('aloha_scene/right_gripper').id].reshape(3, 3)

            print(f"Left gripper position: {left_gripper_pos}")
            print(f"Target l position: {target_l}")
            print(f"Right gripper position: {right_gripper_pos}")
            print(f"Target r position: {target_r}")
            print(f"Left gripper orientation: {left_gripper_rot}")
            print(f"Target l orientation: {rot_l.as_matrix()}")
            print(f"Right gripper orientation: {right_gripper_rot}")
            print(f"Target r orientation: {rot_r.as_matrix()}")

            rate = RateLimiter(frequency=200.0)
            while viewer.is_running():
                left_jacobian = left_configuration.get_frame_jacobian("aloha_scene/left_gripper", "site")
                left_jacobian_rank = np.linalg.matrix_rank(left_jacobian)
                print("Left Jacobian Rank:", left_jacobian_rank)
                print("left condition number:", np.linalg.cond(left_jacobian))
                print("left jacobian matrix:\n", left_jacobian)

                right_jacobian = right_configuration.get_frame_jacobian("aloha_scene/right_gripper", "site")
                right_jacobian_rank = np.linalg.matrix_rank(right_jacobian)
                print("Right Jacobian Rank:", right_jacobian_rank)
                print("right condition number:", np.linalg.cond(right_jacobian))
                print("right jacobian matrix:\n", right_jacobian)

                for i in range(max_iters):
                    left_vel = mink.solve_ik(
                        left_configuration,
                        [l_ee_task],
                        rate.dt,
                        solver,
                        limits=left_limit,
                        damping=1e-3,
                    )

                    right_vel = mink.solve_ik(
                        right_configuration,
                        [r_ee_task],
                        rate.dt,
                        solver,
                        limits=right_limit,
                        damping=1e-3,
                    )

                    left_configuration.integrate_inplace(left_vel, rate.dt)
                    right_configuration.integrate_inplace(right_vel, rate.dt)

                    data.qpos[left_relevant_qpos_indices] = left_configuration.q
                    data.qpos[right_relevant_qpos_indices] = right_configuration.q

                    data.qvel[left_relevant_qvel_indices] = left_configuration.dq
                    data.qvel[right_relevant_qvel_indices] = right_configuration.dq

                    left_gripper_pos = data.site_xpos[model.site('aloha_scene/left_gripper').id]
                    right_gripper_pos = data.site_xpos[model.site('aloha_scene/right_gripper').id]
                    left_gripper_rot = data.site_xmat[model.site('aloha_scene/left_gripper').id].reshape(3, 3)
                    right_gripper_rot = data.site_xmat[model.site('aloha_scene/right_gripper').id].reshape(3, 3)

                    l_pos_error = np.linalg.norm(left_gripper_pos - target_l)
                    r_pos_error = np.linalg.norm(right_gripper_pos - target_r)

                    l_ori_error = np.linalg.norm(left_gripper_rot - rot_l.as_matrix())
                    r_ori_error = np.linalg.norm(right_gripper_rot - rot_r.as_matrix())

                    print(f"Left position error: {l_pos_error}")
                    print(f"Right position error: {r_pos_error}")
                    print(f"Left orientation error: {l_ori_error}")
                    print(f"Right orientation error: {r_ori_error}")

                    l_pos_achieved = l_pos_error < pos_threshold
                    r_pos_achieved = r_pos_error < pos_threshold

                    l_ori_achieved = True
                    r_ori_achieved = True

                    if (l_pos_achieved and l_ori_achieved and r_pos_achieved and r_ori_achieved):
                        print(f"Target reached after {i} iterations.")
                        target_l, target_r, rot_l, rot_r = self.generate_random_targets()

                        l_target_pose = mink.SE3.from_rotation_and_translation(rot_l, target_l)
                        r_target_pose = mink.SE3.from_rotation_and_translation(rot_r, target_r)

                        l_ee_task.set_target(l_target_pose)
                        r_ee_task.set_target(r_target_pose)

                        self.update_target_sites(target_l, target_r, rot_l, rot_r)

                        print("")
                        break

                    data.ctrl[left_actuator_ids] = left_configuration.q
                    data.ctrl[right_actuator_ids] = right_configuration.q
                
                    mujoco.mj_step(model, data)

                    viewer.sync()
                    rate.sleep()

                data.ctrl[left_actuator_ids] = left_configuration.q
                data.ctrl[right_actuator_ids] = right_configuration.q
               
                mujoco.mj_step(model, data)

                viewer.sync()
                rate.sleep()

    def generate_random_targets(self):
        x_range_left = (-0.4, -0.1)
        y_range_left = (-0.2, 0.2)
        z_range_left = (1.0, 1.2)

        x_l = np.random.uniform(*x_range_left)
        y_l = np.random.uniform(*y_range_left)
        z_l = np.random.uniform(*z_range_left)
        target_l = np.array([x_l, y_l, z_l])

        x_range_right = (0.1, 0.4)
        y_range_right = (-0.2, 0.2)
        z_range_right = (1.0, 1.2)

        x_r = np.random.uniform(*x_range_right)
        y_r = np.random.uniform(*y_range_right)
        z_r = np.random.uniform(*z_range_right)
        target_r = np.array([x_r, y_r, z_r])

        rot_l = SO3.sample_uniform()
        rot_r = SO3.sample_uniform()

        # manual rotation
        # rot_l = SO3.from_matrix(np.array([[ 0.98594576,  0.16076923, -0.04543352], [ 0.16076923, -0.83907153,  0.51972322], [ 0.04543352, -0.51972322, -0.85312577]]))
        # rot_r = SO3.from_matrix(np.array([[-0.69831662,  0.47513026,  0.53535514], [ 0.24867168, -0.54030231,  0.80388794], [ 0.6712051,   0.69449597,  0.25915064]]))

        return target_l, target_r, rot_l, rot_r
        
    def add_target_sites(self):
        target_l, target_r, rot_l, rot_r = self.generate_random_targets()
        self.target_site_id_l = self.model.site('aloha_scene/target').id
        self.target_site_id_r = self.model.site('aloha_scene/target2').id
        self.update_target_sites(target_l, target_r, rot_l, rot_r)

    def update_target_sites(self, target_l, target_r, rot_l, rot_r):
        self.data.site_xpos[self.target_site_id_l] = target_l
        self.model.site_pos[self.target_site_id_l] = target_l
        self.data.site_xpos[self.target_site_id_r] = target_r
        self.model.site_pos[self.target_site_id_r] = target_r

        rot_l_matrix_flat = rot_l.as_matrix().flatten()
        rot_r_matrix_flat = rot_r.as_matrix().flatten()

        self.data.site_xmat[self.target_site_id_l] = rot_l_matrix_flat
        self.data.site_xmat[self.target_site_id_r] = rot_r_matrix_flat

if __name__ == "__main__":
    controller = AlohaMocapControl()
    controller.run()

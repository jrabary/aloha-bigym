import numpy as np
import mujoco
import mujoco.viewer
import mink
from bigym.envs.pick_and_place import PickBox
from bigym.action_modes import AlohaPositionActionMode
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.robots.configs.aloha import AlohaRobot
from bigym.envs.manipulation import StackBlocks

from reduced_configuration import ReducedConfiguration


from pathlib import Path
from loop_rate_limiters import RateLimiter

_JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]

# Single arm velocity limits, taken from:
# https://github.com/Interbotix/interbotix_ros_manipulators/blob/main/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/vx300s.urdf.xacro
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
        # configuration = mink.Configuration(model)
        relevant_qpos_indices = np.array([model.jnt_qposadr[model.joint(name).id] for name in joint_names])
        relevant_qvel_indices = np.array([model.jnt_dofadr[model.joint(name).id] for name in joint_names])
        print(f"Relevant qpos indices: {relevant_qpos_indices}")
        print(f"Relevant qvel indices: {relevant_qvel_indices}")

        configuration = ReducedConfiguration(model, relevant_qpos_indices, relevant_qvel_indices)
        
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
        
        print(f"l_ee_task: {l_ee_task.frame_name}")

        # l_ee_task.set_target(mink.SE3.from_translation(np.array([0.5, 1, 0.5])))
        # r_ee_task.set_target(mink.SE3.from_translation(np.array([-1, 1, 0.5])))
        
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
            geom_pairs=collision_pairs,  # type: ignore
            minimum_distance_from_collisions=0.05,
            collision_detection_distance=0.1,
        )

        limits = [
            mink.ConfigurationLimit(model=model),
            mink.VelocityLimit(model, velocity_limits),
            # collision_avoidance_limit,
        ]

        solver = "quadprog"
        # solver = "osqp"
        pos_threshold = 0.01
        ori_threshold = 0.01
        max_iters = 20

        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            self.add_target_site()
            mujoco.mj_forward(model, data)

            target = self.generate_random_target()
            # r_target = self.generate_random_target()

            l_ee_task.set_target(mink.SE3.from_translation(target))
            r_ee_task.set_target(mink.SE3.from_translation(target))
            self.update_target_site(target)
            
            rate = RateLimiter(frequency=200.0)
            while viewer.is_running():
                jacobian = configuration.get_frame_jacobian("aloha_scene/left_gripper", "site")
                print("Jacobian Matrix:\n", jacobian)
                condition_number = np.linalg.cond(jacobian)
                print("Jacobian Condition Number:", condition_number)

                # Compute velocity and integrate into the next configuration.
                for i in range(max_iters):
                    # print(f"Left gripper position: {data.site_xpos[model.site('aloha_scene/left_gripper').id]}")
                    # print(f"Right gripper position: {data.site_xpos[model.site('aloha_scene/right_gripper').id]}")
                    # print(f"Target position: {self.data.site_xpos[self.target_site_id]}")
                    # configuration.q[:] = data.qpos

                    vel = mink.solve_ik(
                        configuration,
                        tasks,
                        rate.dt,
                        solver,
                        limits=[],
                        damping=1e-3,
                    )

                    print(f"Velocity: {vel}")
                    configuration.integrate_inplace(vel, rate.dt)

                    print(f"self qpos: {self.env.unwrapped._mojo.data.qpos}")

                    # l_err = l_ee_task.compute_error(configuration)
                    # # print(f"Left error: {l_err}")
                    # l_pos_achieved = np.linalg.norm(l_err[:3]) < pos_threshold
                    # l_ori_achieved = np.linalg.norm(l_err[3:]) < ori_threshold

                    # r_err = r_ee_task.compute_error(configuration)  
                    # # print(f"Right error: {r_err}")
                    # r_pos_achieved = np.linalg.norm(r_err[:3]) < pos_threshold
                    # r_ori_achieved = np.linalg.norm(r_err[3:]) < ori_threshold
                    left_gripper_pos = data.site_xpos[model.site('aloha_scene/left_gripper').id]
                    right_gripper_pos = data.site_xpos[model.site('aloha_scene/right_gripper').id]
                    target_pos = self.data.site_xpos[self.target_site_id]

                    left_gripper_rot = data.site_xmat[model.site('aloha_scene/left_gripper').id]
                    right_gripper_rot = data.site_xmat[model.site('aloha_scene/right_gripper').id]
                    target_rot = self.data.site_xmat[self.target_site_id]

                    print(f"target rot: {target_rot}")
                    print(f"left gripper rot: {left_gripper_rot}")
                    print(f"right gripper rot: {right_gripper_rot}")

                    l_pos_achieved = np.linalg.norm(left_gripper_pos - target_pos) < pos_threshold
                    r_pos_achieved = np.linalg.norm(right_gripper_pos - target_pos) < pos_threshold

                    l_ori_achieved = np.linalg.norm(left_gripper_rot - target_rot) < ori_threshold
                    r_ori_achieved = np.linalg.norm(right_gripper_rot - target_rot) < ori_threshold

                    print(f"norm of left dist: {np.linalg.norm(left_gripper_pos - target_pos)}")

                    if (l_pos_achieved and l_ori_achieved and r_pos_achieved and r_ori_achieved):
                        print(f"Target reached after {i} iterations.")
                        break
                    
                data.ctrl[actuator_ids] = configuration.q[dof_ids]
               
                # Step the simulation
                mujoco.mj_step(model, data)
                print(f"Control inputs (data.ctrl): {data.ctrl[actuator_ids]}")
                print(f"Actuator forces (data.actuator_force): {data.actuator_force[actuator_ids]}")

                # print(f"control choice: {data.ctrl[actuator_ids]}")
                print(f"model: {model}")
                mujoco.mj_step(model, data)

                # Visualize at fixed FPS.
                viewer.sync()
                rate.sleep()

    def generate_random_target(self):
        # Define the workspace limits for the ALOHA arms
        x_range = (-0.2, 0.2)
        y_range = (-0.2, 0.2)
        z_range = (1, 1.2)  #height

        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        z = np.random.uniform(*z_range)

        return np.array([x, y, z])
    
    def add_target_site(self):
        target = self.generate_random_target()
        self.target_site_id = self.model.site('aloha_scene/target').id
        self.update_target_site(target)

    def update_target_site(self, target):
        self.data.site_xpos[self.target_site_id] = target
        self.model.site_pos[self.target_site_id] = target
        

if __name__ == "__main__":
    controller = AlohaMocapControl()
    controller.run()
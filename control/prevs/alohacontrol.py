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

from reduced_configuration import ReducedConfiguration
from loop_rate_limiters import RateLimiter
import threading
from pynput import mouse  
import glfw

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
        self.env = StackBlocks(
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

        self.target_l = np.array([-0.25, 0.0, 1.1])
        self.target_r = np.array([0.25, 0.0, 1.1])
        self.rot_l = SO3.identity()
        self.rot_r = SO3.identity()
        self.targets_updated = False  

        self.x_min, self.x_max = -0.4, 0.4
        self.y_min, self.y_max = -0.4, 0.4
        self.z_min, self.z_max = 0.8, 1.4

        self.delta = 0.04

        self.mouse_listener = mouse.Listener(on_scroll=self.on_scroll, on_click=self.on_click)
        self.mouse_listener.start()

        self.left_gripper_actuator_id = self.model.actuator("aloha_scene/aloha_gripper_left/gripper_actuator").id
        self.right_gripper_actuator_id = self.model.actuator("aloha_scene/aloha_gripper_right/gripper_actuator").id
        self.left_gripper_pos = 0.037
        self.right_gripper_pos = 0.037

    def so3_to_matrix(self, so3_rotation: SO3) -> np.ndarray:
        return so3_rotation.as_matrix()

    def matrix_to_so3(self, rotation_matrix: np.ndarray) -> SO3:
        return SO3.from_matrix(rotation_matrix)

    def apply_rotation(self, current_rotation: SO3, rotation_change: np.ndarray) -> SO3:
        rotation_matrix = self.so3_to_matrix(current_rotation)
        change_matrix = SO3.exp(rotation_change).as_matrix()
        new_rotation_matrix = change_matrix @ rotation_matrix
        return self.matrix_to_so3(new_rotation_matrix)

    def update_rotation(self, axis: str, angle: float):
        rotation_change = np.zeros(3)
        if axis == 'x':
            rotation_change[0] = angle
        elif axis == 'y':
            rotation_change[1] = angle
        elif axis == 'z':
            rotation_change[2] = angle
        
        self.rot_l = self.apply_rotation(self.rot_l, rotation_change)
        self.rot_r = self.apply_rotation(self.rot_r, rotation_change)
        self.targets_updated = True

    def keyboard_callback(self, keycode):
        rotation_angle = 0.1 
        keycode = chr(keycode)
       
        if keycode == 'W':
            self.target_l[1] += self.delta
            print(f" Moving left target forward")
        elif keycode == 'S':
            print(f"self.target_l[1]: {self.target_l[1]}")
            self.target_l[1] -= self.delta
            print(f" Moving left target backward")
            print(f"new self.target_l[1]: {self.target_l[1]}")
        elif keycode == 'A':
            self.target_l[0] -= self.delta
            print(f" Moving left target left")
        elif keycode == 'D':
            self.target_l[0] += self.delta
            print(f" Moving left target right")
        elif keycode == 'Q':
            self.target_l[2] += self.delta
            print(f" Moving left target up")
        elif keycode == 'E':
            self.target_l[2] -= self.delta
            print(f" Moving left target down")

        elif keycode == 'I':
            self.target_r[1] += self.delta
            print(f" Moving right target forward")
        elif keycode == 'K':
            self.target_r[1] -= self.delta
            print(f" Moving right target backward")
        elif keycode == 'J':
            self.target_r[0] -= self.delta
            print(f" Moving right target left")
        elif keycode == 'L':
            self.target_r[0] += self.delta
            print(f" Moving right target right")
        elif keycode == 'U':
            self.target_r[2] += self.delta
            print(f" Moving right target up")
        elif keycode == 'O':
            self.target_r[2] -= self.delta
            print(f" Moving right target down")
        
        elif keycode == 'X':
            # print(f"pre gripper positions: {self.left_gripper_pos}, {self.right_gripper_pos}")
            # self.left_gripper_pos += self.delta / 50
            # self.right_gripper_pos += self.delta / 50
            # print(f"new gripper positions: {self.left_gripper_pos}, {self.right_gripper_pos}")
            self.left_gripper_pos = 0.037
            self.right_gripper_pos = 0.037
            print(f" Opening gripper")
        elif keycode == 'Z':    
            # print(f"pre gripper positions: {self.left_gripper_pos}, {self.right_gripper_pos}")
            # self.left_gripper_pos -= self.delta / 50
            # self.right_gripper_pos -= self.delta / 50
            # print(f"new gripper positions: {self.left_gripper_pos}, {self.right_gripper_pos}")
            self.left_gripper_pos = 0.002
            self.right_gripper_pos = 0.002
            print(f" Closing gripper")

        #TODO: change rotations in each dimension
        
        elif keycode == 'R':
            self.update_rotation('x', rotation_angle)
            print(f"Rotating around X axis")
        elif keycode == 'T':
            self.update_rotation('x', -rotation_angle)
            print(f"Rotating around X axis (reverse)")
        elif keycode == 'F':
            self.update_rotation('y', rotation_angle)
            print(f"Rotating around Y axis")
        elif keycode == 'G':
            self.update_rotation('y', -rotation_angle)
            print(f"Rotating around Y axis (reverse)")
        elif keycode == 'V':
            self.update_rotation('z', rotation_angle)
            print(f"Rotating around Z axis")
        elif keycode == 'B':
            self.update_rotation('z', -rotation_angle)
            print(f"Rotating around Z axis (reverse)")
        
        self.target_l = np.clip(self.target_l, [self.x_min, self.y_min, self.z_min], [self.x_max, self.y_max, self.z_max])
        self.target_r = np.clip(self.target_r, [self.x_min, self.y_min, self.z_min], [self.x_max, self.y_max, self.z_max])

        self.left_gripper_pos = np.clip(self.left_gripper_pos, 0.02, 0.037)
        self.right_gripper_pos = np.clip(self.right_gripper_pos, 0.02, 0.037)

        self.targets_updated = True

    def control_gripper(self, left_gripper_position, right_gripper_position):
        # Args: gripper_position (float): A value between 0.002 (closed) and 0.037 (open).
        left_gripper_position = np.clip(left_gripper_position, 0.02, 0.037)
        right_gripper_position = np.clip(right_gripper_position, 0.02, 0.037)
        self.data.ctrl[self.left_gripper_actuator_id] = left_gripper_position
        self.data.ctrl[self.right_gripper_actuator_id] = right_gripper_position
        
    def on_scroll(self, x, y, dx, dy):
        if dx > 0:
            self.target_l[0] += self.delta
            self.left_gripper_pos += self.delta
            self.right_gripper_pos += self.delta
        elif dx < 0:
            self.target_l[0] -= self.delta
            self.left_gripper_pos -= self.delta
            self.right_gripper_pos -= self.delta

        if dy < 0:
            self.target_l[1] += self.delta
        elif dy > 0:
            self.target_l[1] -= self.delta

        self.target_l[0] = np.clip(self.target_l[0], self.x_min, self.x_max)
        self.target_l[1] = np.clip(self.target_l[1], self.y_min, self.y_max)
        self.left_gripper_pos = np.clip(self.left_gripper_pos, 0.02, 0.037)
        self.right_gripper_pos = np.clip(self.right_gripper_pos, 0.02, 0.037)

        self.targets_updated = True
        # print(f"New target_l x position: {self.target_l[0]}")

    def on_click(self, x, y, button, pressed):
        if pressed:
            if button == mouse.Button.left:
                self.target_l[2] += self.delta
            elif button == mouse.Button.right:
                self.target_l[2] -= self.delta

            self.target_l[2] = np.clip(self.target_l[2], self.z_min, self.z_max)

            self.targets_updated = True
            # print(f"New target_l z position: {self.target_l[2]}")
    
    def run(self):
        model = self.model
        data = self.data

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

        collision_avoidance_limit = mink.CollisionAvoidanceLimit(
            model=model,
            geom_pairs=[],  
            minimum_distance_from_collisions=0.1,
            collision_detection_distance=0.1,
        )

        limits = [
            mink.VelocityLimit(model, velocity_limits),
            collision_avoidance_limit,
        ]

        solver = "osqp"
        pos_threshold = 0.1
        max_iters = 20

        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False, key_callback=self.keyboard_callback
        ) as viewer:

            # viewer._loop.thread._window.set_key_callback(self.keyboard_callback)
            
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            self.add_target_sites()
            mujoco.mj_forward(model, data)

            l_target_pose = mink.SE3.from_rotation_and_translation(self.rot_l, self.target_l)
            r_target_pose = mink.SE3.from_rotation_and_translation(self.rot_r, self.target_r)

            l_ee_task.set_target(l_target_pose)
            r_ee_task.set_target(r_target_pose)

            rate = RateLimiter(frequency=200.0)
            while viewer.is_running():
                self.control_gripper(self.left_gripper_pos, self.right_gripper_pos)
                if self.targets_updated:
                    l_target_pose = mink.SE3.from_rotation_and_translation(self.rot_l, self.target_l)
                    l_ee_task.set_target(l_target_pose)
                    r_target_pose = mink.SE3.from_rotation_and_translation(self.rot_r, self.target_r)
                    r_ee_task.set_target(r_target_pose)

                    self.update_target_sites(self.target_l, self.target_r, self.rot_l, self.rot_r)
                    self.targets_updated = False

                for _ in range(max_iters):
                    left_vel = mink.solve_ik(
                        left_configuration,
                        [l_ee_task],
                        rate.dt,
                        solver,
                        limits=limits,
                        damping=1e-3,
                    )

                    # right_vel = np.zeros_like(right_configuration.dq)
                    right_vel = mink.solve_ik(
                        right_configuration,
                        [r_ee_task],
                        rate.dt,
                        solver,
                        limits=limits,
                        damping=1e-3,
                    )

                    left_configuration.integrate_inplace(left_vel, rate.dt)
                    right_configuration.integrate_inplace(right_vel, rate.dt)

                    data.qpos[left_relevant_qpos_indices] = left_configuration.q
                    data.qpos[right_relevant_qpos_indices] = right_configuration.q

                    data.qvel[left_relevant_qvel_indices] = left_configuration.dq
                    data.qvel[right_relevant_qvel_indices] = right_configuration.dq

                    data.ctrl[left_actuator_ids] = left_configuration.q
                    data.ctrl[right_actuator_ids] = right_configuration.q

                    self.control_gripper(self.left_gripper_pos, self.right_gripper_pos)
                    # print(f"self.left_gripper_pos: {self.left_gripper_pos}, self.right_gripper_pos: {self.right_gripper_pos}")

                    mujoco.mj_step(model, data)

                    viewer.sync()
                    rate.sleep()

            self.mouse_listener.stop()

    def add_target_sites(self):
        self.target_site_id_l = self.model.site('aloha_scene/target').id
        self.target_site_id_r = self.model.site('aloha_scene/target2').id
        self.update_target_sites(self.target_l, self.target_r, self.rot_l, self.rot_r)

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

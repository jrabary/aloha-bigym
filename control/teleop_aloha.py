import numpy as np
import mujoco
import mujoco.viewer
import mink
import h5py
import time
from bigym.envs.dishwasher import DishwasherClose
from bigym.action_modes import AlohaPositionActionMode
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.robots.configs.aloha import AlohaRobot
from mink import SO3
from reduced_configuration import ReducedConfiguration
from loop_rate_limiters import RateLimiter
from pyjoycon import JoyCon, get_R_id, get_L_id
import os

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
        self.joycon_L = JoyCon(*get_L_id())
        self.joycon_R = JoyCon(*get_R_id())

        self.env = DishwasherClose(
            action_mode=AlohaPositionActionMode(floating_base=False, absolute=False, control_all_joints=True),
            observation_config=ObservationConfig(
                cameras=[
                    CameraConfig(name="wrist_cam_left", rgb=True, depth=False, resolution=(1280, 720)),
                    CameraConfig(name="wrist_cam_right", rgb=True, depth=False, resolution=(1280, 720)),
                    CameraConfig(name="overhead_cam", rgb=True, depth=False, resolution=(1280, 720)),
                    CameraConfig(name="teleoperator_pov", rgb=True, depth=False, resolution=(1280, 720)),
                ],
            ),
            render_mode="human",
            robot_cls=AlohaRobot
        )
        self.env.reset()
        self.model = self.env.unwrapped._mojo.model
        self.data = self.env.unwrapped._mojo.data

        self.camera_renderers = {}
        for camera_config in self.env.observation_config.cameras:
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_config.name)
            renderer = mujoco.Renderer(self.model, camera_config.resolution[0], camera_config.resolution[1])
            self.camera_renderers[camera_config.name] = (renderer, camera_id)

        self.target_l = np.array([-0.4, 0.5, 1.1])
        self.target_r = np.array([0.4, 0.5, 1.1])
        self.rot_l = SO3.identity()
        self.rot_r = SO3.from_matrix(-np.eye(3))

        self.update_rotation('z', np.pi/2, 'left')
        self.update_rotation('z', -np.pi/2, 'right')
        self.update_rotation('y', np.pi/2, 'left')
        self.update_rotation('y', np.pi/2, 'right')

        self.targets_updated = False

        self.x_min, self.x_max = -0.6, 0.6
        self.y_min, self.y_max = -0.6, 0.6
        self.z_min, self.z_max = 0.78, 1.6

        self.left_gripper_actuator_id = self.model.actuator("aloha_scene/aloha_gripper_left/gripper_actuator").id
        self.right_gripper_actuator_id = self.model.actuator("aloha_scene/aloha_gripper_right/gripper_actuator").id
        self.left_gripper_pos = 0.037
        self.right_gripper_pos = 0.037

        self.calibrate()

        self.initialize_hdf5_storage()

        self.action = np.zeros(14)

    def initialize_hdf5_storage(self):
        # each time this file is run the name of the dataset should be the next number compared with the highest episode number already in the dataset directory
        # stop collecting data and save the data when ctrl c called
        # each episode has its own file
        
        self.dataset_dir = 'data'
        self.hdf5_file = h5py.File(f'{self.dataset_dir}/episode_0.hdf5', 'w') #manually set

        self.data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }

        self.camera_names = ['wrist_cam_left', 'wrist_cam_right', 'overhead_cam', 'teleoperator_pov']

        for cam_name in self.camera_names:
            self.data_dict[f'/observations/images/{cam_name}'] = []

    def store_data(self):
        # at each timestep append current data to data_dict
        self.data_dict['/observations/qpos'].append(self.get_qpos())
        self.data_dict['/observations/qvel'].append(self.get_qvel())
        self.data_dict['/action'].append(self.get_action())
        for cam_name in self.camera_names:
            self.data_dict[f'/observations/images/{cam_name}'].append(self.get_obs(cam_name))

    def get_qpos(self):
        print(f"qpos: {self.data.qpos}, len qpos: {len(self.data.qpos)}")
        return self.data.qpos.copy()
        
    def get_qvel(self):
        print(f"qvel: {self.data.qvel}, len qvel: {len(self.data.qvel)}")
        return self.data.qvel.copy()
    
    def get_action(self):
        return self.action.copy()
        
    def get_obs(self, cam_name):
        renderer, cam_id = self.camera_renderers[cam_name]
        renderer.update_scene(self.data, cam_id)
        img = renderer.render()
        return img

    def final_save(self):
        episode_idx = 0
        max_timesteps = 1000

        # straight from Tony Zhao ACT record_sim_episodes.py
        # HDF5
        t0 = time.time()
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in self.camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            for name, array in self.data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')

    def calibrate(self):
        num_samples = 100
        right_samples = []
        left_samples = []
        for _ in range(num_samples):
            status = self.joycon_L.get_status()
            status_R = self.joycon_R.get_status()
            accel = status['accel']
            accel_R = status_R['accel']
            rot = status['gyro']
            rot_R = status_R['gyro']
            joystick = status['analog-sticks']['left']
            joystick_R = status_R['analog-sticks']['right']

            left_samples.append([accel['x'], accel['y'], accel['z'], rot['x'], rot['y'], rot['z'], joystick['horizontal'], joystick['vertical']])
            right_samples.append([accel_R['x'], accel_R['y'], accel_R['z'], rot_R['x'], rot_R['y'], rot_R['z'], joystick_R['horizontal'], joystick_R['vertical']])
            time.sleep(0.01)
        
        self.right_calibration_offset = np.mean(right_samples, axis=0)
        self.left_calibration_offset = np.mean(left_samples, axis=0)

    def so3_to_matrix(self, so3_rotation: SO3) -> np.ndarray:
        return so3_rotation.as_matrix()

    def matrix_to_so3(self, rotation_matrix: np.ndarray) -> SO3:
        return SO3.from_matrix(rotation_matrix)

    def apply_rotation(self, current_rotation: SO3, rotation_change: np.ndarray) -> SO3:
        rotation_matrix = self.so3_to_matrix(current_rotation)
        change_matrix = SO3.exp(rotation_change).as_matrix()
        new_rotation_matrix = change_matrix @ rotation_matrix
        return self.matrix_to_so3(new_rotation_matrix)

    def update_rotation(self, axis: str, angle: float, side: str):
        rotation_change = np.zeros(3)
        if axis == 'x':
            rotation_change[0] = angle
        elif axis == 'y':
            rotation_change[1] = angle
        elif axis == 'z':
            rotation_change[2] = angle

        if side == 'left':
            self.rot_l = self.apply_rotation(self.rot_l, rotation_change)
        else:
            self.rot_r = self.apply_rotation(self.rot_r, rotation_change)
        self.targets_updated = True

    def joycon_control_l(self):
        status = self.joycon_L.get_status()
        rotation = status['gyro']
        button_lower = status['buttons']['left']['zl']
        button_higher = status['buttons']['left']['l']
        joystick = status['analog-sticks']['left']
        up = status['buttons']['left']['up']
        down = status['buttons']['left']['down']

        self.action[0] = (joystick['horizontal'] - self.left_calibration_offset[6]) * 0.00005
        self.action[1] = (joystick['vertical'] - self.left_calibration_offset[7]) * 0.00005
        self.action[2] = -0.03 if button_lower == 1 else 0.03 if button_higher == 1 else 0
        self.action[3] = -rotation['y'] * 0.0001
        self.action[4] = rotation['x'] * 0.0001
        self.action[5] = rotation['z'] * 0.0001
        self.action[6] = 0.037 if up == 1 else 0.002 if down == 1 else 0

        self.target_l[0] += self.action[0]
        self.target_l[1] += self.action[1]
        self.target_l[2] += self.action[2]

        self.target_l = np.clip(self.target_l, [self.x_min, self.y_min, self.z_min], [self.x_max, self.y_max, self.z_max])

        #rotation
        self.update_rotation('x', self.action[3], 'left')
        self.update_rotation('y', self.action[4], 'left')
        self.update_rotation('z', self.action[5], 'left')

        #gripper
        self.left_gripper_pos = self.action[6]
        self.targets_updated = True

    def joycon_control_r(self):
        status = self.joycon_R.get_status()
        rotation = status['gyro']
        button_lower = status['buttons']['right']['zr']
        button_higher = status['buttons']['right']['r']
        joystick = status['analog-sticks']['right']
        up = status['buttons']['right']['x']
        down = status['buttons']['right']['b']

        self.action[7] = (joystick['horizontal'] - self.right_calibration_offset[6]) * 0.00005
        self.action[8] = (joystick['vertical'] - self.right_calibration_offset[7]) * 0.00005
        self.action[9] = -0.03 if button_lower == 1 else 0.03 if button_higher == 1 else 0
        self.action[10] = rotation['y'] * 0.0001
        self.action[11] = rotation['x'] * 0.0001
        self.action[12] = -rotation['z'] * 0.0001
        self.action[13] = 0.037 if up == 1 else 0.002 if down == 1 else 0

        self.target_r[0] += self.action[7]
        self.target_r[1] += self.action[8]
        self.target_r[2] += self.action[9]

        self.target_r = np.clip(self.target_r, [self.x_min, self.y_min, self.z_min], [self.x_max, self.y_max, self.z_max])

        #rotation
        self.update_rotation('x', self.action[10], 'right')
        self.update_rotation('y', self.action[11], 'right')
        self.update_rotation('z', self.action[12], 'right')

        #gripper
        self.right_gripper_pos = self.action[13]
        self.targets_updated = True

    def control_gripper(self, left_gripper_position, right_gripper_position):
        left_gripper_position = np.clip(left_gripper_position, 0.02, 0.037)
        right_gripper_position = np.clip(right_gripper_position, 0.02, 0.037)
        self.data.ctrl[self.left_gripper_actuator_id] = left_gripper_position
        self.data.ctrl[self.right_gripper_actuator_id] = right_gripper_position

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
        max_iters = 20

        try:
            with mujoco.viewer.launch_passive(
                model=model, data=data, show_left_ui=True, show_right_ui=False
            ) as viewer:
                mujoco.mjv_defaultFreeCamera(model, viewer.cam)

                self.add_target_sites()
                mujoco.mj_forward(model, data)

                l_target_pose = mink.SE3.from_rotation_and_translation(self.rot_l, self.target_l)
                r_target_pose = mink.SE3.from_rotation_and_translation(self.rot_r, self.target_r)

                l_ee_task.set_target(l_target_pose)
                r_ee_task.set_target(r_target_pose)

                sim_rate = RateLimiter(frequency=200.0) 

                # data recording should be 50hz, 
                # loop is currently 200hz, thus record every 4th loop
                data_recording_interval = 4

                while viewer.is_running():
                    self.joycon_control_l()
                    self.joycon_control_r()

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
                            sim_rate.dt,
                            solver,
                            limits=limits,
                            damping=1e-1,
                        )

                        right_vel = mink.solve_ik(
                            right_configuration,
                            [r_ee_task],
                            sim_rate.dt,
                            solver,
                            limits=limits,
                            damping=1e-1,
                        )

                        left_configuration.integrate_inplace(left_vel, sim_rate.dt)
                        right_configuration.integrate_inplace(right_vel, sim_rate.dt)

                        data.qpos[left_relevant_qpos_indices] = left_configuration.q
                        data.qpos[right_relevant_qpos_indices] = right_configuration.q

                        data.qvel[left_relevant_qvel_indices] = left_configuration.dq
                        data.qvel[right_relevant_qvel_indices] = right_configuration.dq

                        data.ctrl[left_actuator_ids] = left_configuration.q
                        data.ctrl[right_actuator_ids] = right_configuration.q

                        self.control_gripper(self.left_gripper_pos, self.right_gripper_pos)

                        mujoco.mj_step(model, data)

                        iters += 1
                        if iters == data_recording_interval:
                            self.store_data()
                            iters = 0

                        viewer.sync()
                        sim_rate.sleep()
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        # if ctrl+c is called (currently only way to end this), call self.final_save() to save the data
        self.final_save()
        self.hdf5_file.close()
        for renderer, _ in self.camera_renderers.values():
            renderer.close()

if __name__ == "__main__":
    try:
        controller = AlohaMocapControl()
        controller.run()
    except KeyboardInterrupt:
        controller.close()
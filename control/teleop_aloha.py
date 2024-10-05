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

        self.dt = 1/60

        self.gravity_magnitude_right = 0
        self.gravity_magnitude_left = 0

        self.calibrate()

        self.initialize_hdf5_storage()

    def initialize_hdf5_storage(self):
        self.hdf5_file = h5py.File('robot_data.hdf5', 'w')

        self.num_qpos = len(self.data.qpos)
        self.num_qvel = len(self.data.qvel)

        # qpos and qvel
        self.qpos_dataset = self.hdf5_file.create_dataset(
            'joint_data/qpos', (0, self.num_qpos), maxshape=(None, self.num_qpos), chunks=(1, self.num_qpos))
        self.qvel_dataset = self.hdf5_file.create_dataset(
            'joint_data/qvel', (0, self.num_qvel), maxshape=(None, self.num_qvel), chunks=(1, self.num_qvel))
        
        # target action data (rotations, translations, gripper pos)
        self.action_dataset = {
            'rot_l': self.hdf5_file.create_dataset(
                'action_data/rot_l', (0, 3, 3), maxshape=(None, 3, 3), chunks=True),
            'rot_r': self.hdf5_file.create_dataset(
                'action_data/rot_r', (0, 3, 3), maxshape=(None, 3, 3), chunks=True),
            'target_l': self.hdf5_file.create_dataset(
                'action_data/target_l', (0, 3), maxshape=(None, 3), chunks=True),
            'target_r': self.hdf5_file.create_dataset(
                'action_data/target_r', (0, 3), maxshape=(None, 3), chunks=True),
            'left_gripper_pos': self.hdf5_file.create_dataset(
                'action_data/left_gripper_pos', (0,), maxshape=(None,), chunks=True),
            'right_gripper_pos': self.hdf5_file.create_dataset(
                'action_data/right_gripper_pos', (0,), maxshape=(None,), chunks=True)
        }

        # camera data (compressed)
        self.camera_feeds = {}
        for cam_name in ['wrist_cam_left', 'wrist_cam_right', 'overhead_cam', 'teleoperator_pov']:
            self.camera_feeds[cam_name] = self.hdf5_file.create_dataset(
                f'camera_feeds/{cam_name}', (0, 720, 1280, 3), maxshape=(None, 720, 1280, 3),
                compression='gzip', compression_opts=9, chunks=(1, 720, 1280, 3))

    def store_data(self):
        """Append new timestep data to HDF5."""
        # joint pos and vel
        self.qpos_dataset.resize(self.qpos_dataset.shape[0] + 1, axis=0)
        self.qvel_dataset.resize(self.qvel_dataset.shape[0] + 1, axis=0)
        self.qpos_dataset[-1, :] = self.data.qpos.copy()
        self.qvel_dataset[-1, :] = self.data.qvel.copy()

        # target action 
        for key, value in self.action_dataset.items():
            value.resize(value.shape[0] + 1, axis=0)
            if key == 'rot_l':
                value[-1] = self.rot_l.as_matrix()
            elif key == 'rot_r':
                value[-1] = self.rot_r.as_matrix()
            elif key == 'target_l':
                value[-1] = self.target_l.copy()
            elif key == 'target_r':
                value[-1] = self.target_r.copy()
            elif key == 'left_gripper_pos':
                value[-1] = self.left_gripper_pos
            elif key == 'right_gripper_pos':
                value[-1] = self.right_gripper_pos

        # 4 cameras (left wrist, right wrist, overhead, teleop pov)
        for cam_name, (renderer, cam_id) in self.camera_renderers.items():
            renderer.update_scene(self.data, cam_id)
            img = renderer.render()
            self.camera_feeds[cam_name] = img

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

        self.gravity_magnitude_right = self.right_calibration_offset[2] 
        self.gravity_magnitude_left = self.left_calibration_offset[2] 

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

        #translation
        if button_lower == 1:
            self.target_l[2] -= 0.03
        elif button_higher == 1:
            self.target_l[2] += 0.03

        self.target_l[0] += (joystick['horizontal'] - self.left_calibration_offset[6]) * 0.00005
        self.target_l[1] += (joystick['vertical'] - self.left_calibration_offset[7]) * 0.00005

        self.target_l = np.clip(self.target_l, [self.x_min, self.y_min, self.z_min], [self.x_max, self.y_max, self.z_max])

        #rotation
        self.update_rotation('x', -rotation['y'] * 0.0001, 'left')
        self.update_rotation('y', rotation['x'] * 0.0001, 'left')
        self.update_rotation('z', rotation['z'] * 0.0001, 'left')

        #gripper
        if up == 1:
            self.left_gripper_pos = 0.037
        elif down == 1:
            self.left_gripper_pos = 0.002
        self.targets_updated = True

    def joycon_control_r(self):
        status = self.joycon_R.get_status()
        rotation = status['gyro']
        button_lower = status['buttons']['right']['zr']
        button_higher = status['buttons']['right']['r']
        joystick = status['analog-sticks']['right']
        up = status['buttons']['right']['x']
        down = status['buttons']['right']['b']

        #translation
        if button_higher == 1:
            self.target_r[2] += 0.03
        elif button_lower == 1:
            self.target_r[2] -= 0.03

        self.target_r[0] += (joystick['horizontal'] - self.right_calibration_offset[6]) * 0.00005
        self.target_r[1] += (joystick['vertical'] - self.right_calibration_offset[7]) * 0.00005

        self.target_r = np.clip(self.target_r, [self.x_min, self.y_min, self.z_min], [self.x_max, self.y_max, self.z_max])

        #rotation
        self.update_rotation('x', rotation['y'] * 0.0001, 'right')
        self.update_rotation('y', rotation['x'] * 0.0001, 'right')
        self.update_rotation('z', -rotation['z'] * 0.0001, 'right')

        #gripper
        if up == 1: 
            self.right_gripper_pos = 0.0037
        elif down == 1:
            self.right_gripper_pos = 0.002
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

            # currently recording data at 50Hz (same as in paper)
            data_recording_interval = 0.02  # 1 / Hz
            last_data_record_time = time.time()

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

                    viewer.sync()
                    sim_rate.sleep()

                # current_time = time.time()
                # if current_time - last_data_record_time >= data_recording_interval:
                #     self.store_data()
                #     print(f"qpos shape: {self.data.qpos.shape}, dataset shape: {self.qpos_dataset.shape}")
                #     print(f"qvel shape: {self.data.qvel.shape}, dataset shape: {self.qvel_dataset.shape}")
                #     last_data_record_time = current_time

        self.close()

    def close(self):
        self.hdf5_file.close()
        for renderer, _ in self.camera_renderers.values():
            renderer.close()

if __name__ == "__main__":
    try:
        controller = AlohaMocapControl()
        controller.run()
    except KeyboardInterrupt:
        controller.close()
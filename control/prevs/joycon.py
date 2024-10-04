from pyjoycon import JoyCon, get_L_id, get_R_id
import time
import numpy as np

joycon_L = JoyCon(*get_L_id())
joycon_R = JoyCon(*get_R_id())

num_samples = 100
right_samples = []
left_samples = []
for _ in range(num_samples):
    status = joycon_L.get_status()
    status_R = joycon_R.get_status()
    accel = status['accel']
    accel_R = status_R['accel']
    rot = status['gyro']
    rot_R = status_R['gyro']
    button = status['buttons']['left']['zl']
    button_R = status_R['buttons']['right']['zr']
    joystick = status['analog-sticks']['left']
    joystick_R = status_R['analog-sticks']['right']

    right_samples.append([accel['x'], accel['y'], accel['z'], rot['x'], rot['y'], rot['z'], joystick['horizontal'], joystick['vertical']])
    left_samples.append([accel_R['x'], accel_R['y'], accel_R['z'], rot_R['x'], rot_R['y'], rot_R['z'], joystick_R['horizontal'], joystick_R['vertical']])
    time.sleep(0.01)

right_calibration_offset = np.mean(right_samples, axis=0)
left_calibration_offset = np.mean(left_samples, axis=0)

while(True):
    # status = joycon_L.get_status()  # or joycon_R.get_status()
    # status_R = joycon_R.get_status()
    # accel = status['accel']
    # accel_R = status_R['accel']
    # rot = status['gyro']
    # rot_R = status_R['gyro']
    # button = status['buttons']['left']['zl']
    # button_R = status_R['buttons']['right']['zr']
    # joystick = status['analog-sticks']['left']
    # joystick_R = status_R['analog-sticks']['right']

    # accel_x = accel['x'] - 450
    # accel_y = accel['y'] + 44
    # accel_z = accel['z'] - 4130
    # accel_R_x = accel_R['x'] - 250
    # accel_R_y = accel_R['y'] + 150
    # accel_R_z = accel_R['z'] + 3725

    status = joycon_L.get_status()  
    status_R = joycon_R.get_status()
    accel = status['accel']
    accel_R = status_R['accel']
    rot = status['gyro']
    rot_R = status_R['gyro']
    button = status['buttons']['left']['zl']
    button_R = status_R['buttons']['right']['zr']
    joystick = status['analog-sticks']['left']
    joystick_R = status_R['analog-sticks']['right']




    up = status['buttons']['left']['up']
    down = status['buttons']['left']['down']

    accel_x = accel['x'] - right_calibration_offset[0]
    accel_y = accel['y'] - right_calibration_offset[1]
    accel_z = accel['z'] - right_calibration_offset[2]
    accel_R_x = accel_R['x'] - left_calibration_offset[0]
    accel_R_y = accel_R['y'] - left_calibration_offset[1]
    accel_R_z = accel_R['z'] - left_calibration_offset[2]

    rot_x = rot['x'] - right_calibration_offset[3]
    rot_y = rot['y'] - right_calibration_offset[4]
    rot_z = rot['z'] - right_calibration_offset[5]
    rot_R_x = rot_R['x'] - left_calibration_offset[3]
    rot_R_y = rot_R['y'] - left_calibration_offset[4]
    rot_R_z = rot_R['z'] - left_calibration_offset[5]

    joystick_horizontal = joystick['horizontal'] - right_calibration_offset[6]
    joystick_vertical = joystick['vertical'] - right_calibration_offset[7]
    joystick_R_horizontal = joystick_R['horizontal'] - left_calibration_offset[6]
    joystick_R_vertical = joystick_R['vertical'] - left_calibration_offset[7]

    # rot_x = rot['x'] + 2.6
    # rot_y = rot['y'] - 1.75
    # rot_z = rot['z'] + 0.87
    # rot_R_x = rot_R['x'] + 7.85
    # rot_R_y = rot_R['y'] + 0.87
    # rot_R_z = rot_R['z'] + 3.5

    print(f"Acceleration: X={accel_x:.2f}, Y={accel_y:.2f}, Z={accel_z:.2f}")
    print(f"Acceleration R: X={accel_R_x:.2f}, Y={accel_R_y:.2f}, Z={accel_R_z:.2f}")

    print(f"Rotation: X={rot_x:.2f}, Y={rot_y:.2f}, Z={rot_z:.2f}")
    print(f"Rotation R: X={rot_R_x:.2f}, Y={rot_R_y:.2f}, Z={rot_R_z:.2f}")

    print(f"Joystick: X={joystick_horizontal:.2f}, Y={joystick_vertical:.2f}")
    print(f"Joystick R: X={joystick_R_horizontal:.2f}, Y={joystick_R_vertical:.2f}")

    print(f"Button up: {up}")
    print(f"Button down: {down}")

    # print(f"Acceleration: X={accel['x']:.2f}, Y={accel['y']:.2f}, Z={accel['z']:.2f}")
    # print(f"Rotation: X={rot['x']:.2f}, Y={rot['y']:.2f}, Z={rot['z']:.2f}")
    # print(f"Acceleration R: X={accel_R['x']:.2f}, Y={accel_R['y']:.2f}, Z={accel_R['z']:.2f}")
    # print(f"Rotation R: X={rot_R['x']:.2f}, Y={rot_R['y']:.2f}, Z={rot_R['z']:.2f}")
    # print(f"Button: {button}")
    # print(f"Button R: {button_R}")
    # print(f"Joystick: X={joystick['horizontal']:.2f}, Y={joystick['vertical']:.2f}")
    # print(f"Joystick R: X={joystick_R['horizontal']:.2f}, Y={joystick_R['vertical']:.2f}")
    time.sleep(1)
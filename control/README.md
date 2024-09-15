[control](/control/) contains aloha bigym environments with controllable arms through mink inverse kinematics

[manual_minkaloha.py](manual_minkaloha.py)
Most updated file which allows for mouse-control of ALOHA arms
Run `mjpython manual_minkaloha.py`
- Scroll left/right to control xpos
- Scroll up/down to control ypos
- rightclick/leftclick to control zpos

[mink_default](mink_default.py)
Uses the mink default aloha example adjusted to minimally work with xml (no bigym)
Sets one target

[mink_aloha.py](mink_default.py)
Works with the `bigym` `robot` class supporting the definition of one target

[mink_alohadual.py](mink_alohadual.py)
Randomly defines two targets near the arms and samples new target positions when previous ones are reached
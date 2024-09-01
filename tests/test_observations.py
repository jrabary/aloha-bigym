"""Test observations."""
import numpy as np
import pytest

from bigym.action_modes import JointPositionActionMode
from bigym.envs.reach_target import ReachTarget

from bigym.utils.observation_config import CameraConfig, ObservationConfig


@pytest.mark.parametrize(
    "rgb,",
    [True, False],
)
@pytest.mark.parametrize(
    "depth,",
    [True, False],
)
@pytest.mark.parametrize(
    "resolution,",
    [(4, 4), (8, 8)],
)
class TestObservations:
    """Test observations."""

    def test_visual_observations_exist(self, rgb, depth, resolution):
        """Test visual observations exist."""
        camera_name = "head"
        env = ReachTarget(
            action_mode=JointPositionActionMode(floating_base=True, absolute=True),
            observation_config=ObservationConfig(
                cameras=[
                    CameraConfig(
                        name=camera_name,
                        rgb=rgb,
                        depth=depth,
                        resolution=resolution,
                    )
                ],
            ),
        )
        observation, _ = env.reset()
        assert (f"rgb_{camera_name}" in observation) == rgb
        if rgb:
            assert observation[f"rgb_{camera_name}"].shape == (3, *resolution)
        assert (f"depth_{camera_name}" in observation) == depth
        if depth:
            assert observation[f"depth_{camera_name}"].shape == resolution


def test_no_visual_observations():
    """Test no visual observations."""
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        observation_config=ObservationConfig(),
    )
    observation, _ = env.reset()
    assert not np.all([key.startswith("rgb_") for key in observation.keys()])
    assert not np.all([key.startswith("depth_") for key in observation.keys()])

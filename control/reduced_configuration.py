import numpy as np
from typing import Optional
import mujoco
from mink.configuration import Configuration  # Adjust the import based on your setup

class ReducedConfiguration(Configuration):
    def __init__(self, model, relevant_qpos_indices, relevant_qvel_indices):
        super().__init__(model)
        self.relevant_qpos_indices = relevant_qpos_indices
        self.relevant_qvel_indices = relevant_qvel_indices

    def update(self, q: Optional[np.ndarray] = None) -> None:
        if q is not None:
            # Update only relevant qpos entries
            self.data.qpos[self.relevant_qpos_indices] = q
        super().update()

    def get_frame_jacobian(self, frame_name: str, frame_type: str) -> np.ndarray:
        full_jacobian = super().get_frame_jacobian(frame_name, frame_type)
        # Extract relevant columns
        reduced_jacobian = full_jacobian[:, self.relevant_qvel_indices]
        return reduced_jacobian

    def integrate_inplace(self, velocity: np.ndarray, dt: float) -> None:
        # Create a full velocity vector
        full_velocity = np.zeros(self.model.nv)
        full_velocity[self.relevant_qvel_indices] = velocity
        super().integrate_inplace(full_velocity, dt)

    @property
    def nv(self) -> int:
        return len(self.relevant_qvel_indices)

    @property
    def q_relevant(self) -> np.ndarray:
        return self.data.qpos[self.relevant_qpos_indices]

    @q_relevant.setter
    def q_relevant(self, value: np.ndarray):
        self.data.qpos[self.relevant_qpos_indices] = value

from brax import base
from brax import math
from brax.envs import env
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp


class FreeFall(env.PipelineEnv):
    
    def __init__(self,
               backend='generalized',
               **kwargs):
        sys = mjcf.load('./envs/free_fall.xml')
        print("inertias: \n", sys.link.inertia.i)

        super().__init__(sys=sys, backend=backend, **kwargs)
        
    def reset(self, rng: jp.ndarray) -> env.State:  # pytype: disable=signature-mismatch  # jax-ndarray
        qpos = self.sys.init_q
        qvel = jp.zeros(self.sys.qd_size())
        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state)
        reward, done = jp.array(0.0), jp.array(0.0)
        return env.State(pipeline_state, obs, reward, done)
        
    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        obs = self._get_obs(pipeline_state)

        return state.replace(
        pipeline_state=pipeline_state, obs=obs
    )

    def _get_obs(self, pipeline_state: base.State) -> jp.ndarray:
        """Observe ant body position and velocities."""
        qpos = pipeline_state.q
        qvel = pipeline_state.qd

        return jp.concatenate([qpos] + [qvel])

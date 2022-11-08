#@title Environment
import numpy as np

from rlba.types import (Array, ArraySpec, DiscreteArraySpec, 
                        NestedArray, NestedArraySpec, 
                        NestedDiscreteArraySpec, BoundedArraySpec)
                        
class RandomWalkEnv:

    def __init__(self,
                seed: int,
                length: int=20) -> None:
        self._rng = np.random.default_rng(seed)
        self._length = length
        self._position = 0

        self._observation_spec: ArraySpec = {
            'reward': BoundedArraySpec(
                shape=(0,),
                dtype=float,
                minimum=-np.inf,
                maximum=np.inf,
                name="reward",
            ),
            'state': BoundedArraySpec(
                shape=(0,),
                dtype=float,
                minimum=0,
                maximum=self._length-1,
                name="state",
            ),
            'terminate': BoundedArraySpec(
                shape=(0,),
                dtype=bool,
                minimum=0,
                maximum=1,
                name='terminate'
            )
        }
        self._action_spec = DiscreteArraySpec(1, name="action_spec")

    def step(self, action:NestedArray):
        if self._rng.binomial(1, 0.5):
            self._position += 1
        else:
            self._position = max(0, self._position-1)

        if self._position == self._length:
            terminate = True
            self._position = 0
        else:
            terminate = False
        reward = 1
        observation = {
            'reward': reward,
            'state': self._position,
            'terminate': terminate
        }
        return observation

    @property
    def observation_spec(self) -> NestedArraySpec:
        return self._observation_spec
    
    @property
    def action_spec(self) -> NestedDiscreteArraySpec:
        return self._action_spec

    def close(self):
        """Frees any resources used by the environment.
        Implement this method for an environment backed by an external process.
        This method can be used directly
        ```python
        env = Env(...)
        # Use env.
        env.close()
        ```
        or via a context manager
        ```python
        with Env(...) as env:
          # Use env.
        ```
        """
        pass

    def __enter__(self):
        """Allows the environment to be used in a with-statement context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Allows the environment to be used in a with-statement context."""
        del exc_type, exc_value, traceback  # Unused.
        self.close()
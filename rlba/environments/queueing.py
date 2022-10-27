import numpy as np

from rlba.types import Array, ArraySpec, DiscreteArraySpec, NestedArray, NestedArraySpec, NestedDiscreteArraySpec, BoundedArraySpec

class QueueingEnv:

    def __init__(self,
                seed: int,
                buffer_size: int = 20,
                prob_arrival: float = 0.5,
                prob_fast_service: float = 0.7,
                prob_slow_service: float = 0.6,
                fast_cost: float = 10.0,
                slow_cost: float = 0.0,
                abandonment_cost: float = 500.0):
        self._rng = np.random.default_rng(seed)
        self._buffer_size = buffer_size
        self._prob_arrival = prob_arrival
        self._prob_fast_service = prob_fast_service
        self._prob_slow_service = prob_slow_service
        self._fast_cost = fast_cost
        self._slow_cost = slow_cost
        self._abandonment_cost = abandonment_cost
        self._queue_length = self._buffer_size

        self._observation_spec: ArraySpec = {
            'reward': BoundedArraySpec(
                shape=(1,),
                dtype=float,
                minimum=-np.inf,
                maximum=np.inf,
                name="reward",
            ),
            'queue_length': BoundedArraySpec(
                shape=(1,),
                dtype=int,
                minimum=0,
                maximum=self._buffer_size,
                name="queue_length",
            ),
            'abandonment': BoundedArraySpec(
                shape=(1,),
                dtype=int,
                minimum=0,
                maximum=1,
                name="abandonment",
            ),
            'termination': BoundedArraySpec(
                shape=(1,),
                dtype=int,
                minimum=0,
                maximum=1,
                name="termination",
            ),
        }

        self._action_spec = DiscreteArraySpec(2, name="action_spec")

    def get_reward(self, action: NestedArray, length: int, abandonment: int) -> float:
        _action = self._validate_action(action)
        reward = length*(-1.0) + abandonment*(-1.0)*self._abandonment_cost - \
            (self._fast_cost if _action=='fast' else self._slow_cost)
        return reward

    def _validate_action(self, action: NestedArray):
        # print(f"validate_action, {action}")
        assert(action == 1 or action == 0), "undefined action type"
        _action = 'fast' if action == 1 else 'slow'
        return _action

    def step(self, action: NestedArray):
        _action = self._validate_action(action)
        arrival_token = self._rng.binomial(1, self._prob_arrival)
        departure_token = self._rng.binomial(1, self._prob_fast_service 
                            if _action=='fast' else self._prob_slow_service)
        abandonment = max(self._queue_length + 
                        arrival_token - departure_token - self._buffer_size, 0)
        self._queue_length = max(min(self._queue_length + arrival_token - departure_token, self._buffer_size), 0)
        termination = int(self._queue_length == 0)
        reward = self.get_reward(action, self._queue_length, abandonment)
        observation = {
            'reward': np.array([reward]),
            'queue_length': np.array([self._queue_length]),
            'abandonment': np.array([abandonment]),
            'termination': np.array([termination]),
        }
        if termination == 1:
            self._queue_length = self._buffer_size
        return observation

    def transition_probs(self, queue_length: int, action: NestedArray):
        _action = self._validate_action(action)
        prob_departure = self._prob_fast_service if _action=='fast' \
            else self._prob_slow_service
        prob_increase = self._prob_arrival * (1-prob_departure) \
            if queue_length < self._buffer_size else 0.0
        prob_decrease = (1-self._prob_arrival) * prob_departure
        prob_same = 1 - prob_increase - prob_decrease
        prob_abandonment = 0 if queue_length<self._buffer_size \
            else self._prob_arrival*(1-prob_departure)
        return prob_decrease, prob_same, prob_increase, prob_abandonment

    def get_buffer_size(self):
        return self._buffer_size

    def get_queue_length(self):
        return self._queue_length

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
# Copyright 2022 Morteza Ibrahimi. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple agent-environment training loop."""

import time
from typing import Optional, Sequence

from rlba import agent, environment
from rlba.utils import counting
from rlba.utils import loggers
from rlba.utils import observers as observers_lib


class EnvironmentLoop:
    """A simple RL environment loop.

    This takes `Environment` and `agent` instances and coordinates their
    interaction. Agent is updated if `should_update=True`. This can be used as:

      loop = EnvironmentLoop(environment, agent)
      loop.run(num_episodes)

    A `Counter` instance can optionally be given. If not given a local Counter will be
    created to maintain counts between calls to the `run` method.

    A `Logger` instance can also be passed in order to control the output of the
    loop. If not given a platform-specific default logger will be used as defined
    by utils.loggers.make_default_logger. A string `label` can be passed to easily
    change the label associated with the default logger; this is ignored if a
    `Logger` instance is given.

    A list of 'Observer' instances can be specified to generate additional metrics
    to be logged by the logger. They have access to the 'Environment' instance,
    the current timestep datastruct and the current action.
    """

    def __init__(
        self,
        environment: environment.Environment,
        agent: agent.Agent,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        label: str = "environment_loop",
        observers: Sequence[observers_lib.EnvLoopObserver] = (),
    ):
        # Internalize agent and environment.
        self._environment = environment
        self._agent = agent
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(
            label, steps_key=self._counter.get_steps_key()
        )
        self._observers = observers

    def run(
        self,
        n_step: int = 1000,
    ) -> loggers.LoggingData:
        """Run the environment loop.

        Run a loop which simulate interactions between the environment and the agent
        by iteratively performing the following loop:
        1. Get an action from the agent
        2. Submit the action to the environment and get an observation
        3. Give the observation to the agent to update its (summary) of history.

        Returns:
          An instance of `loggers.LoggingData`.
        """
        # Reset any counts and start the environment.
        start_time = time.time()
        max_n_step = n_step
        n_step = 0

        # For evaluation, this keeps track of the total undiscounted reward
        # accumulated.
        cumulative_return = 0.0
        result_list = []

        # Run an episode.
        while n_step < max_n_step:
            # Generate an action from the agent's policy and step the environment.
            action = self._agent.select_action()
            obs = self._environment.step(action)

            # Have the agent observe the observation and potentially update itself.
            reward = self._agent.observe(action, obs=obs)

            for observer in self._observers:
                # One environment step was completed. Observe the current state of the
                # environment, the current timestep and the action.
                observer.observe(self._environment, action, obs, reward)

            # Book-keeping.
            n_step += 1
            cumulative_return += reward

            # Record counts.
            counts = self._counter.increment(steps=1)

            # Collect the results and combine with counts.
            steps_per_second = n_step / (time.time() - start_time)
            result = {
                "cumulative_return": cumulative_return,
                "steps_per_second": steps_per_second,
            }
            result.update(counts)
            for observer in self._observers:
                result.update(observer.get_metrics())

            self._logger.write(result)
            result_list.append(result)
        return result_list

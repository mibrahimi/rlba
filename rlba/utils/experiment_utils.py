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

"""Utility for running experiments."""

from typing import Optional


from typing import Callable, Iterator, Tuple
import pandas as pd
from rlba.agent import Agent
from rlba.environment import Environment
from rlba.environment_loop import EnvironmentLoop
from rlba.utils.loggers import Logger, make_default_logger, TerminalLogger
from rlba.utils.observers.base import EnvLoopObserver
import time


class Experiment(object):
    """A simple experiment object that can run agent/env loop for multiple trials."""

    def __init__(
        self,
        loop_factory: Callable[[int], Tuple[EnvironmentLoop, Environment, Agent]],
        logger: Logger = TerminalLogger("experiment", print_fn=print),
    ):
        """
        Args:
          loop_factory: a callable that get an integer seed and returns
              environment loop plus environment and agent.
          logger: a logger to log the loop stats.
        """

        self._loop_factory = loop_factory
        self._logger = logger
        self._result_df = None

    def run(self, seeds: Iterator[int], time_horizon: int) -> None:
        """Runs an agent on an environment.
        Args:
          time_horizon: Number of steps in each trial.
          seeds: iterator of seeds value. Each seed corresponds to one trial.
        Return:
            a pandas dataframe containing the results of the experiment.
        """

        results = []
        for sidx, seed in enumerate(seeds):
            loop, _, _ = self._loop_factory(seed)
            start_time = time.time()
            res = loop.run(time_horizon)
            res = pd.DataFrame(res)
            res["trial_id"] = sidx
            res["seed"] = seed
            results.append(res)
            self._logger.write(
                {"trial ": sidx, "exec time(s)": time.time() - start_time}
            )
        result_df = pd.concat(results)
        return result_df


def make_experiment_logger(
    label: str, steps_key: Optional[str] = None, task_instance: int = 0
) -> Logger:
    del task_instance
    if steps_key is None:
        steps_key = f"{label}_steps"
    return make_default_logger(label=label, steps_key=steps_key)

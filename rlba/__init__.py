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

"""RLBA is an interface adhering to the RL formulation presented in the course "Reinforcement Learning: Behaviors and Applications"."""

# Internal import.

# Expose specs and types modules.
from rlba import types

# Make __version__ accessible.
from rlba._metadata import __version__

# Expose core interfaces.
from rlba.agent import Agent
from rlba.agent import Learner

# Expose the environment loop.
from rlba.environment_loop import EnvironmentLoop

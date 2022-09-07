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

"""Install script for setuptools."""

import datetime
from importlib import util as import_util
import os
import sys

from setuptools import find_packages
from setuptools import setup
import setuptools.command.build_py
import setuptools.command.develop

spec = import_util.spec_from_file_location('_metadata', 'rlba/_metadata.py')
_metadata = import_util.module_from_spec(spec)
spec.loader.exec_module(_metadata)


core_requirements = [
    'absl-py',
    'dm-env',
    'dm-tree',
    'numpy',
    'pillow',
]

jax_requirements = [
    'chex',
    'jax',
    'jaxlib',
    'dm-haiku',
    'flax',
    'optax',
    'rlax',
]

testing_requirements = [
    'pytype',
    'pytest-xdist',
]


long_description = """RLBA is a library for RL formulation and interface
definition that contains the code for the "Reinforcement Learning: Behaviors and
Applications" course.
For more information see [github repository](https://github.com/mibrahimi/rlba)."""

# Get the version from metadata.
version = _metadata.__version__


setup(
    name='rlba',
    version=version,
    description='The companion code for the course "Reinforcement Learning: Behaviors and Applications.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Morteza Ibrahimi, Zheng Wen, Benjamin Van Roy',
    license='Apache License, Version 2.0',
    keywords='reinforcement-learning, python, machine learning, course',
    packages=find_packages(),
    package_data={'': ['requirements.txt']},
    include_package_data=True,
    install_requires=core_requirements,
    extras_require={
        'jax': jax_requirements,
        'testing': testing_requirements,
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)

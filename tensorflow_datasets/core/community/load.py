# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

"""Utils to load community datasets."""

import importlib
import inspect
from typing import Any, Type

from tensorflow_datasets.core import dataset_builder
from tensorflow_datasets.core import registered


def builder_cls_from_module(
    module_name: str,
) -> Type[dataset_builder.DatasetBuilder]:
  """Imports the module and extract the `tfds.core.DatasetBuilder`.

  Args:
    module_name: Dataset module to import containing the dataset definition
      (e.g. `tensorflow_datasets.image.mnist.mnist`)

  Returns:
    The extracted tfds.core.DatasetBuilder builder class.
  """
  # Module can be created during execution, so call invalidate_caches() to
  # make sure the new module is noticed by the import system.
  importlib.invalidate_caches()

  with registered.skip_registration():
    module = importlib.import_module(module_name)

  builder_classes = [v for v in module.__dict__.values() if _is_builder_cls(v)]
  if len(builder_classes) != 1:
    raise ValueError(
        f'Could not load DatasetBuilder from: {module_name}. '
        'Make sure the module only contains a single `DatasetBuilder`. '
        f'Detected builders: {builder_classes}'
    )
  return builder_classes[0]


def _is_builder_cls(obj: Any) -> bool:
  """Returns True if obj is a `DatasetBuilder` class."""
  return (
      isinstance(obj, type)
      and issubclass(obj, dataset_builder.DatasetBuilder)
      and not inspect.isabstract(obj)
  )

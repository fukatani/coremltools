#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import sys
import itertools
import numpy as np
import pytest
import torch.nn as nn

from .testing_utils import (
    contains_op,
    generate_input_data,
    ModuleWrapper,
    TorchBaseTest
)
from coremltools import RangeDim
from coremltools.models.utils import _python_version
from coremltools.models.utils import _macos_version
from coremltools.converters.mil import testing_reqs

from coremltools import TensorType
from coremltools._deps import version_lt


backends = testing_reqs.backends
torch = pytest.importorskip("torch")
torch.manual_seed(30)
np.random.seed(30)

# Set of common shapes for testing. Not all layers support 1D, so these two
# set of shapes are kept separate
COMMON_SHAPES = [(1, 10), (1, 5, 6), (1, 3, 5, 6), (1, 3, 4, 5, 6)]
COMMON_SHAPES_ALL = [(1, )] + COMMON_SHAPES


class TestElementWiseUnary(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, shape, diagonal",
        itertools.product(
            backends,
            [(2, 3), (3, 1), (1, 4), (5, 8)],
            [None, 1, 3],
        ),
    )
    def test_triu(self, backend, shape, diagonal):
        params_dict = {}
        if diagonal is not None:
            params_dict["diagonal"] = diagonal
        model = ModuleWrapper(function=torch.triu)
        self.run_compare_torch(
            shape, model, backend=backend, rand_range=(-5, 5)
        )

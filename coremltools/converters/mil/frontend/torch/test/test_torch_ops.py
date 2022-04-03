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
        "backend, shape, op_string",
        itertools.product(
            backends,
            [(1, 3, 5, 8)],
            [
                "abs",
                "acos",
                "asin",
                "atan",
                "ceil",
                "cos",
                "cosh",
                "exp",
                "floor",
                "round",
                "sin",
                "sinh",
                "sqrt",
                "square",
                "tan",
                "tanh",
                "sign",
            ],
        ),
    )
    def test_elementwise_no_params(self, backend, shape, op_string):
        if not contains_op(torch, op_string):
            return
        op_func = getattr(torch, op_string)
        model = ModuleWrapper(function=op_func)
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape, clamp_range",
        itertools.product(
            backends,
            [(1, 3, 5, 8)],
            [(0.0, 1.0), (-1.0, 0.5), (0.2, 0.7), (None, 4.0), (-3.0, None)],
        ),
    )
    def test_clamp(self, backend, shape, clamp_range):
        params_dict = {}
        if clamp_range[0] is not None:
            params_dict["min"] = clamp_range[0]
        if clamp_range[1] is not None:
            params_dict["max"] = clamp_range[1]

        model = ModuleWrapper(torch.clamp, params_dict)
        self.run_compare_torch(
            shape, model, backend=backend, rand_range=(-5, 5)
        )

    @pytest.mark.parametrize(
        "backend, shape, diagonal",
        itertools.product(
            backends,
            [(1, 3, 5, 8)],
            [None, 1, 3],
        ),
    )
    def test_triu(self, backend, shape, diagonal):
        params_dict = {}
        if diagonal is not None:
            params_dict["diagonal"] = diagonal
        model = ModuleWrapper(function=torch.triu)
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape, diagonal",
        itertools.product(
            backends,
            [(1, 3, 5, 8)],
            [None, 1],
        ),
    )
    def test_tril(self, backend, shape, diagonal):
        params_dict = {}
        if diagonal is not None:
            params_dict["diagonal"] = diagonal
        model = ModuleWrapper(function=torch.tril)
        self.run_compare_torch(
            shape, model, backend=backend,
        )
    @pytest.mark.parametrize(
        "backend, shape, threshold",
        itertools.product(
            backends,
            [(1, 3, 5, 8)],
            [(0.0, 0.0), (0.5, 0.5), (0.5, 10), (0.9, 0.0)]
        ),
    )

    def test_threshold(self, backend, shape, threshold):
        model = torch.nn.Threshold(threshold[0], threshold[1]).eval()
        self.run_compare_torch(
            shape, model, backend=backend,
            use_cpu_for_conversion=True, # TODO: change this to False (rdar://78343191)
        )

    @pytest.mark.parametrize(
        "backend, shape, op_string",
        itertools.product(
            backends,
            [(1, 3, 5, 8)],
            [
                "log",
                "rsqrt",
                "reciprocal",
            ],
        ),
    )
    def test_elementwise_numerically_stable(self, backend, shape, op_string):
        op_func = getattr(torch, op_string)
        model = ModuleWrapper(function=op_func)
        self.run_compare_torch(
            shape, model, backend=backend, rand_range=(20, 100)
        )

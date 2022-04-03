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

class TestScriptedModels(TorchBaseTest):

    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend", itertools.product([True, False], backends)
    )
    def test_cond(self, use_cpu_for_conversion, backend):
        if backend[0] == "mlprogram":
            pytest.skip("rdar://81169758 (Cond tests hang on mlprogram backend)")
        if backend[0] == "mlprogram" and not use_cpu_for_conversion:
            pytest.xfail("rdar://78343191 ((MIL GPU) Core ML Tools Unit Test failures [failure to load or Seg fault])")

        class TestNet(nn.Module):
            def forward(self, x):
                if torch.squeeze(x) < 10.:
                    return x*10.
                else:
                    return x*2.

        torch_model = TestNet().eval()

        self.run_compare_torch(torch.tensor([1.]), torch_model,
            input_as_shape=False, backend=backend,
            use_cpu_for_conversion=use_cpu_for_conversion, use_scripting=True)
        self.run_compare_torch(torch.tensor([11.]), torch_model,
            input_as_shape=False, backend=backend,
            use_cpu_for_conversion=use_cpu_for_conversion, use_scripting=True)

    @pytest.mark.parametrize("backend", backends)
    def test_for_loop(self, backend):
        class TestLayer(nn.Module):
            def __init__(self):
                super(TestLayer, self).__init__()

            def forward(self, x):
                x = 2.0 * x
                return x

        class TestNet(nn.Module):
            input_size = (64,)

            def __init__(self):
                super(TestNet, self).__init__()
                layer = TestLayer()
                self.layer = torch.jit.trace(layer, torch.rand(self.input_size))

            def forward(self, x):
                for _ in range(7):
                    x = self.layer(x)
                return x

        model = TestNet().eval()
        
        self.run_compare_torch(model.input_size, model, backend=backend, use_scripting=True)

    @pytest.mark.parametrize("backend", backends)
    def test_while_loop(self, backend):
        class TestLayer(nn.Module):
            def __init__(self):
                super(TestLayer, self).__init__()

            def forward(self, x):
                x = 0.5 * x
                return x

        class TestNet(nn.Module):
            input_size = (1,)

            def __init__(self):
                super(TestNet, self).__init__()
                layer = TestLayer()
                self.layer = torch.jit.trace(layer, torch.rand(self.input_size))

            def forward(self, x):
                while x > 0.01:
                    x = self.layer(x)
                return x

        model = TestNet().eval()

        self.run_compare_torch(model.input_size, model, backend=backend, use_scripting=True)

    @pytest.mark.parametrize("backend", backends)
    def test_if(self, backend):
        if backend[0] == 'mlprogram':
            pytest.xfail("Not supported on ML Program backend")

        class TestLayer(nn.Module):
            def __init__(self):
                super(TestLayer, self).__init__()

            def forward(self, x):
                x = torch.mean(x)
                return x

        class TestNet(nn.Module):
            input_size = (64,)

            def __init__(self):
                super(TestNet, self).__init__()
                layer = TestLayer()
                self.layer = torch.jit.trace(layer, torch.rand(self.input_size))

            def forward(self, x):
                m = self.layer(x)
                if m < 0:
                    scale = -2.0
                else:
                    scale = 2.0
                x = scale * x
                return x

        model = TestNet().eval()

        self.run_compare_torch(model.input_size, model, backend=backend, use_scripting=True)

    @pytest.mark.parametrize("backend", backends)
    def test_linear(self, backend):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = Model().eval()
        
        self.run_compare_torch(
            torch.tensor([[1.,2.]]), 
            model,
            input_as_shape=False, 
            backend=backend,
            use_scripting=True,
        )

    @pytest.mark.parametrize("backend", backends)
    def test_conv(self, backend):
        pytest.xfail("rdar://88194776 ([Converter] coremltools is not working with scripted torch convolution model)")
        model = torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=1,
                                padding="same", stride=1, dilation=1, groups=1, bias=False)
        self.run_compare_torch(
            (1, 2, 4, 5), 
            model,
            backend=backend,
            use_scripting=True,
        )


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

    @pytest.mark.parametrize(
        "backend, shape, diagonal",
        itertools.product(
            backends,
            [(1, 3, 5, 8)],
            [None, 1],
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


class TestMatMul(TorchBaseTest):

    @pytest.mark.parametrize("backend", backends)
    def test_bmm(self, backend):
        shape_x, shape_y = (3,4,5), (3,5,6)
        model = ModuleWrapper(function=torch.bmm)
        self.run_compare_torch(
            [shape_x, shape_y], model, backend=backend,
        )


class TestSplit(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, split_size_or_sections, dim",
        itertools.product(backends, [1, 2, [1, 4]], [0, -2]),
    )
    def test_split(self, backend, split_size_or_sections, dim):
        input_shape = (5, 2)
        model = ModuleWrapper(function=torch.split,
                              kwargs={"split_size_or_sections": split_size_or_sections, "dim": dim})
        self.run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize(
        "backend, split_sizes, dim",
        itertools.product(backends, [[1, 4], [3, 2]], [-1, -2]),
    )
    def test_split_with_sizes(self, backend, split_sizes, dim):
        input_shape = (5, 5)
        model = ModuleWrapper(function=torch.split_with_sizes,
                              kwargs={"split_sizes": split_sizes, "dim": dim})
        self.run_compare_torch(input_shape, model, backend=backend)


class TestUnbind(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, dim",
        itertools.product(backends, [0,1,2]),
    )
    def test_unbind(self, backend, dim):
        input_shape = (3, 3, 4)
        model = ModuleWrapper(function=torch.unbind,
                              kwargs={"dim": dim})
        self.run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_unbind_one_dim_shape(self, backend):
        input_shape = (1,)
        dim = 0
        model = ModuleWrapper(function=torch.unbind,
                              kwargs={"dim": dim})
        self.run_compare_torch(input_shape, model, backend=backend)


class TestTranspose(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, shape, dims",
        itertools.product(backends, COMMON_SHAPES, [(0, 1), (-2, -1), (1, 0), (-1, -2)]),
    )
    def test(self, backend, shape, dims):
        model = ModuleWrapper(function=torch.transpose,
                              kwargs={"dim0": dims[0], "dim1": dims[1]})
        self.run_compare_torch(shape, model, backend=backend)

class TestTo(TorchBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend", itertools.product([True, False], backends,)
    )
    def test_cast_bug(self, use_cpu_for_conversion, backend):
        if backend[0] == "mlprogram" and not use_cpu_for_conversion:
            pytest.xfail("rdar://78343191 ((MIL GPU) Core ML Tools Unit Test failures [failure to load or Seg fault])")

        if backend[0] == "mlprogram" and use_cpu_for_conversion:
            pytest.xfail("numerical mismatch : rdar://78952850")

        class TestModel(torch.nn.Module):
            def forward(self, spans, embedding):
                spans = spans.float().relu().int()

                max1, _ = torch.max(spans, dim=1, keepdim=False)
                max1, _ = torch.max(max1, dim=1, keepdim=False)
                max2, _ = torch.max(embedding, dim=1, keepdim=False)
                max2, _ = torch.max(max2, dim=1, keepdim=False)
                sigmoided_scores = max1 + max2
                return sigmoided_scores

        model = TestModel()
        self.run_compare_torch([(1, 21, 2), (1, 6, 384)], model, backend=backend,
                               use_cpu_for_conversion=use_cpu_for_conversion)# [spans.shape, embedding.shape]

class TestSlice(TorchBaseTest):
    @pytest.mark.skipif(_python_version() < (3, 6), reason="requires python 3.6")
    @pytest.mark.parametrize(
        "backend", backends,
    )
    def test_dynamic_slice(self, backend):
        class DynamicSlicer(torch.nn.Module):
            def __init__(self):
                super(DynamicSlicer, self).__init__()

            def forward(self, x, context_length):
                return x[context_length:, :, :]

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.tokens_embedding = torch.nn.Embedding(10, 10, 0)
                self.context_embedding = torch.nn.Embedding(10, 10, 0)
                self.dynamic_slicer = DynamicSlicer()

            def forward(self, tokens, context, context_length):
                # CoreML requires rank1~5 input, so we use rank 1 for
                # context-length
                tokens_embeddings = self.tokens_embedding(tokens)
                context_embeddings = self.context_embedding(context)
                embeddings = torch.cat((context_embeddings, tokens_embeddings), dim=0)
                embeddings = self.dynamic_slicer(embeddings,
                        torch.squeeze(context_length))

                return embeddings

        model = Model()
        batch_size = 5
        inputs = [ TensorType(name="tokens", shape=(10, batch_size), dtype=np.int64),
                   TensorType(name="context", shape=(3, batch_size), dtype=np.int64),
                   TensorType(name="context_length", shape=(1,), dtype=np.int32),
                   ]
        self.run_compare_torch(inputs, model, rand_range=(0, 8),
                               backend=backend)


class TestRepeat(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, rank",
        itertools.product(backends, list(range(1, 6))),
    )
    def test_repeat(self, backend, rank):
        input_shape = np.random.randint(low=2, high=6, size=rank)
        repeats = np.random.randint(low=2, high=4, size=rank)
        input_shape = tuple(input_shape)

        model = ModuleWrapper(function=lambda x: x.repeat(*repeats))
        self.run_compare_torch(input_shape, model, backend=backend)

class TestStd(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, unbiased",
        itertools.product(backends, [True, False]),
    )
    def test_std_2_inputs(self, backend, unbiased):
        model = ModuleWrapper(function=torch.std,
                              kwargs={"unbiased": unbiased})
        x = torch.randn(1, 5, 10) * 3
        out = torch.std(x, unbiased=unbiased).unsqueeze(0)
        self.run_compare_torch(x, model, expected_results=out,
                           input_as_shape=False, backend=backend)


    @pytest.mark.parametrize(
        "backend, unbiased, dim, keepdim",
        itertools.product(backends, [True, False], [[0,2], [1], [2]], [True, False]),
    )
    def test_std_4_inputs(self, backend, unbiased, dim, keepdim):
        model = ModuleWrapper(function=torch.std,
                              kwargs={"unbiased": unbiased, "dim" : dim, "keepdim": keepdim})
        input_shape = (2, 5, 10)
        self.run_compare_torch(input_shape, model, backend=backend)

class TestZeros(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, rank",
        itertools.product(
            backends,
            [1, 3],
        ),
    )
    def test_zeros_like_static(self, backend, rank):
        if backend[0] == 'mlprogram':
            pytest.xfail("Not supported with ML Program backend")

        class ZerosLikeStaticModel(nn.Module):
            def __init__(self):
                super(ZerosLikeStaticModel, self).__init__()

            def forward(self, x):
                return torch.zeros_like(x)

        input_shape = np.random.randint(low=2, high=6, size=rank)
        input_shape = tuple(input_shape)
        model = ZerosLikeStaticModel()
        self.run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize(
        "backend, rank",
        itertools.product(
            backends,
            [1, 3],
        ),
    )
    def test_zeros_like_dynamic(self, backend, rank):
        if backend[0] == 'mlprogram':
            pytest.xfail("Not supported with ML Program backend")

        class ZerosLikeDynamicModel(nn.Module):
            def __init__(self):
                super(ZerosLikeDynamicModel, self).__init__()

            def forward(self, x):
                if rank == 1:
                    h = x[0]
                    x = torch.zeros(h)
                elif rank == 3:
                    h, w, d = x[0], x[1], x[2]
                    x = torch.zeros(h, w, d)
                return torch.zeros_like(x)

        input_shape = np.random.randint(low=2, high=6, size=rank)
        torch_in = torch.tensor(input_shape)
        model = ZerosLikeDynamicModel()
        torch_out = model(torch_in)
        self.run_compare_torch(torch_in, model, expected_results=torch_out,
                           input_as_shape=False, backend=backend)

    @pytest.mark.parametrize(
        "backend, rank",
        itertools.product(
            backends,
            [1, 3],
        ),
    )
    def test_zeros_static(self, backend, rank):
        if backend[0] == 'mlprogram':
            pytest.xfail("Not supported with ML Program backend")

        class ZerosStaticModel(nn.Module):
            def __init__(self):
                super(ZerosStaticModel, self).__init__()

            def forward(self, x):
                if rank == 1:
                    return torch.zeros(1)
                elif rank == 3:
                    return torch.zeros(2, 3, 5)

        input_shape = np.random.randint(low=2, high=6, size=rank)
        input_shape = tuple(input_shape)
        model = ZerosStaticModel()
        self.run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize(
        "backend, rank",
        itertools.product(
            backends,
            [1, 3],
        ),
    )
    def test_zeros_dynamic(self, backend, rank):
        if backend[0] == 'mlprogram':
            pytest.xfail("Not supported with ML Program backend")

        class ZerosDynamicModel(nn.Module):
            def __init__(self):
                super(ZerosDynamicModel, self).__init__()

            def forward(self, x):
                if rank == 1:
                    h = x[0]
                    x = torch.zeros(h)
                elif rank == 3:
                    h, w, d = x[0], x[1], x[2]
                    x = torch.zeros(h, w, d)
                return x

        input_shape = np.random.randint(low=2, high=6, size=rank)
        torch_in = torch.tensor(input_shape)
        model = ZerosDynamicModel()
        torch_out = model(torch_in)
        self.run_compare_torch(torch_in, model, expected_results=torch_out,
                           input_as_shape=False, backend=backend)

class TestTopk(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, largest, shape_dim_k",
        itertools.product(
            backends,
            [True, False],
            [
             ((4, 6, 7, 3), -1, 2),
             ((10, 3, 4), 2, 2),
             ((5,), 0, 2)
             ],
        ),
    )
    def test_topk(self, backend, largest, shape_dim_k):
        input_shape = shape_dim_k[0]
        dim = shape_dim_k[1]
        k = shape_dim_k[2]

        class TopkModel(nn.Module):
            def __init__(self):
                super(TopkModel, self).__init__()

            def forward(self, x):
                return torch.topk(x, k, dim=dim, largest=largest)

        input_data = torch.rand(input_shape)
        model = TopkModel()
        expected_results = model(input_data)
        expected_results = [expected_results.values, expected_results.indices]
        self.run_compare_torch(
            input_data,
            model,
            expected_results=expected_results,
            input_as_shape=False,
            backend=backend,
        )

class TestLog10(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, rank", itertools.product(backends, range(1, 6)),
    )
    def test_log10(self, backend, rank):

        class Log10Model(nn.Module):
            def __init__(self):
                super(Log10Model, self).__init__()

            def forward(self, x):
                return torch.log10(x)

        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        model = Log10Model()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

class TestFlip(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, rank_dim",
        itertools.product(
            backends,
            [
                (1, [0]),
                (2, [0, 1]),
                (3, [1]),
                (4, [0, 1, 2, 3])
            ]
        ),
    )
    def test_flip(self, backend, rank_dim):

        rank, dim = rank_dim
        class FlipModel(nn.Module):
            def __init__(self):
                super(FlipModel, self).__init__()

            def forward(self, x):
                return torch.flip(x, dim)

        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        model = FlipModel()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

class TestWhere(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [(2, 6), (3, 4, 5)]
        ),
    )
    def test_where_test1(self, backend, shape):

        class WhereModel(nn.Module):
            def __init__(self):
                super(WhereModel, self).__init__()

            def forward(self, x, y):
                return torch.where(x > 0.5, x, y)

        input_shape = [shape, shape]
        model = WhereModel()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [(2, 6), (3, 4, 5)]
        ),
    )
    def test_where_test2(self, backend, shape):

        class WhereModel(nn.Module):
            def __init__(self):
                super(WhereModel, self).__init__()

            def forward(self, cond, x, y):
                return torch.where(cond, x, y)

        cond = torch.rand(*shape) > 0.5
        inputs = [cond, torch.rand(*shape), torch.rand(*shape)]
        model = WhereModel()
        expected_results = model(*inputs)
        self.run_compare_torch(
            inputs,
            model,
            backend=backend,
            expected_results=expected_results,
            input_as_shape=False,
        )

    @pytest.mark.parametrize(
        "backend, shapes",
        itertools.product(
            backends,
            [
                [(1, 2), (1, 2), (1, 1)],
                [(1, 2, 3), (1, 1, 1), (1, 1, 3)],
            ]
        ),
    )
    def test_where_test3(self, backend, shapes):

        class WhereModel(nn.Module):
            def __init__(self):
                super(WhereModel, self).__init__()

            def forward(self, cond, x, y):
                return torch.where(cond, x, y)
        cond_shape, x_shape, y_shape = shapes
        cond = torch.rand(*cond_shape) > 0.5
        inputs = [cond, torch.rand(*x_shape), torch.rand(*y_shape)]
        model = WhereModel()
        expected_results = model(*inputs)
        self.run_compare_torch(
            inputs,
            model,
            backend=backend,
            expected_results=expected_results,
            input_as_shape=False,
        )


class TestSelect(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, dim_index",
        itertools.product(
            backends,
            [
                [0, 0],
                [1, 1],
                [-1, -1],
            ]
        ),
    )
    def test_select(self, backend, dim_index):
        dim, index = dim_index

        class SelectModel(nn.Module):
            def __init__(self):
                super(SelectModel, self).__init__()

            def forward(self, x):
                return x.select(dim, index)

        input_shape = (1,2,3)
        model = SelectModel()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

class TestNonZero(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, rank",
        itertools.product(
            backends,
            [1, 3],
        ),
    )
    def test_non_zero(self, backend, rank):

        if rank == 1:
            input_shape = (10)
            zeros_indices = np.array([1, 4, 7, 9])
        elif rank == 3:
            input_shape = (2, 7, 3)
            zeros_indices = np.array([1, 12, 33, 40])

        input = np.arange(np.prod(input_shape)).astype(np.float32)
        input[zeros_indices] = 0
        input = np.reshape(input, input_shape)
        input = torch.tensor(input)

        model = ModuleWrapper(
            torch.nonzero,
        )

        self.run_compare_torch(input, model,
            input_as_shape=False, backend=backend)

class TestTorchTensor(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, rank",
        itertools.product(
            backends,
            [1, 2, 3, 4, 5],
        ),
    )   
    def test_torch_tensor(self, backend, rank):
        
        class Model(nn.Module):
            def __init__(self, rank):
                super(Model, self).__init__()
                self.rank = rank

            def forward(self, x):
                with torch.no_grad():
                    if self.rank == 1:
                        return self.generate_tensor_rank_1(x)
                    if self.rank == 2:
                        return self.generate_tensor_rank_2(x)
                    if self.rank == 3:
                        return self.generate_tensor_rank_3(x)
                    if self.rank == 4:
                        return self.generate_tensor_rank_4(x)
                    if self.rank == 5:
                        return self.generate_tensor_rank_5(x)   

            @torch.jit.script
            def generate_tensor_rank_1(x):
                _, _, h, w = x.shape
                return torch.tensor([h, w, 0, 1], dtype=torch.int32)

            @torch.jit.script
            def generate_tensor_rank_2(x):
                _, _, h, w = x.shape
                return torch.tensor([[0, h], [h, w], [w, w]], dtype=torch.float32)

            @torch.jit.script
            def generate_tensor_rank_3(x):
                _, _, h, w = x.shape
                return torch.tensor([[[h, 1]],[[3, w]]], dtype=torch.int32)

            @torch.jit.script
            def generate_tensor_rank_4(x):
                _, _, h, w = x.shape
                return torch.tensor([[[[h, h], [h, w]],[[w, w], [w, 1]]],[[[0, 0], [1, 1]],[[0, h], [h, w]]]], dtype=torch.float32)

            @torch.jit.script
            def generate_tensor_rank_5(x):
                _, _, h, w = x.shape
                return torch.tensor([[[[[h, w], [w, w]],[[1, 1],[0, h]]]]], dtype=torch.float32)

        shape = (1, 1, 3, 4)
        model = Model(rank)
        self.run_compare_torch(
            shape, model, backend=backend,
        )
        

class TestTensorAssign(TorchBaseTest):

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_tensor_assign_case_1(self, backend):
        # single dimension assignment for a 1D tensor
        class TensorAssignModel(torch.nn.Module):
            def __init__(self):
                super(TensorAssignModel, self).__init__()

            def forward(self, x):
                x[0] = 0
                x[1] = 1
                y = x + 1
                x[1] = 2 * y[1]
                return x, y

        shape = (5,)
        model = TensorAssignModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_tensor_assign_case_2(self, backend):
        # single dimension assignment for two 1D tensors
        class TensorAssignModel(torch.nn.Module):
            def __init__(self):
                super(TensorAssignModel, self).__init__()

            def forward(self, x, y):
                x[0] = 0
                y[1] = 2
                y = x + y
                x = 2 * y
                y[3] = x[1] + 5
                y[0] = x[0] * 10
                z = x + y
                return z, x, y

        shape = (5,)
        model = TensorAssignModel()
        self.run_compare_torch(
            [shape, shape], model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (5,4),
                (5,4,3),
            ]
        ),
    )
    def test_tensor_assign_case_3(self, backend, shape):
        # broadcast assignment for two n-D tensors
        class TensorAssignModel(torch.nn.Module):
            def __init__(self):
                super(TensorAssignModel, self).__init__()

            def forward(self, x, y):
                x[0] = 0
                x[3] = 1
                y[2] = 2
                return x

        model = TensorAssignModel()
        self.run_compare_torch(
            [shape, shape], model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_itensor_assign_case_4(self, backend):
        # single dimension assignment for two n-D tensors
        class TensorAssignModel(torch.nn.Module):
            def __init__(self):
                super(TensorAssignModel, self).__init__()

            def forward(self, x, y):
                x[0] = torch.tensor([1., 2., 3., 4.])
                x[3] = 1
                y[0] = x[0]
                return x, y

        shape = (5,4)
        model = TensorAssignModel()
        self.run_compare_torch(
            [shape, shape], model, backend=backend,
        )


    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_tensor_assign_case_5(self, backend):
        # slice dimension assigment
        class TensorAssignModel(torch.nn.Module):
            def __init__(self):
                super(TensorAssignModel, self).__init__()

            def forward(self, x):
                x[:,1] = torch.tensor([1., 2.])
                return x

        shape = (2,10)
        model = TensorAssignModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_tensor_assign_case_6(self, backend):
        # a more complicated slice dimension assigment
        class TensorAssignModel(torch.nn.Module):
            def __init__(self):
                super(TensorAssignModel, self).__init__()

            def forward(self, x):
                x[:,1,:] = torch.tensor([1., 2., 3., 4., 5., 6.]).view(2,3)
                return x

        shape = (2,10,3)
        model = TensorAssignModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

class TestIndexPut(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_index_put_case_1(self, backend):
        class IndexPutModel(torch.nn.Module):
            def __init__(self):
                super(IndexPutModel, self).__init__()

            def forward(self, x, y):
                y = x + 1
                mask = torch.tensor([True, False, False, False, True, True]).view(3,2)
                x[mask] = y[mask]
                return x

        shape = (3,2)
        model = IndexPutModel()
        self.run_compare_torch(
            [shape, shape], model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, rank",
        itertools.product(
            backends,
            [0, 1],
        ),
    )
    def test_index_put_case_2(self, backend, rank):
        class IndexPutModel(torch.nn.Module):
            def __init__(self):
                super(IndexPutModel, self).__init__()

            def forward(self, x):
                mask = torch.tensor([True, False, False, False, True, True]).view(3,2)
                if rank == 0:
                    x[mask] = 0.
                if rank == 1:
                    x[mask] = torch.tensor([1.])
                return x

        shape = (3,2)
        model = IndexPutModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_index_put_case_3(self, backend):
        pytest.xfail("rdar://84892125 (Empty tensors handling for non_zero, tile and scatter_nd)")
        class IndexPutModel(torch.nn.Module):
            def __init__(self):
                super(IndexPutModel, self).__init__()

            def forward(self, x, y):
                mask = y > 1
                x[y > 1] = 0.
                return x

        inputs = [
            torch.Tensor([1., 2., 3., 4., 5., 6]),
            torch.Tensor([0., 0., 0., 0., 0., 0.]),
        ]
        model = IndexPutModel()
        self.run_compare_torch(
            inputs, model, backend=backend, input_as_shape=False,
        )

    @pytest.mark.parametrize(
        "backend, rank, accumulate",
        itertools.product(
            backends,
            [1, 2],
            [True, False]
        ),
    )
    def test_index_put_case_4(self, backend, rank, accumulate):
        class IndexPutModel(torch.nn.Module):
            def __init__(self):
                super(IndexPutModel, self).__init__()

            def forward(self, x, indices, values):
                x.index_put_(tuple(indices.t()), values, accumulate=accumulate)
                return x

        if rank == 1:
            inputs = [
                torch.Tensor([1., 2., 3., 4., 5., 6]),
                torch.LongTensor([[0], [4]]),
                torch.Tensor([3., 7.])
            ]
        elif rank == 2:
            inputs = [
                torch.ones([3, 4]),
                torch.LongTensor([[0, 1], [1, 2], [2, 2]]),
                torch.Tensor([1., 5., 8.]),
            ]

        model = IndexPutModel()
        self.run_compare_torch(
            inputs, model, backend=backend, input_as_shape=False,
        )


class TestIndex(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (10,),
                (3, 4, 5, 6),
            ]
        ),
    )
    def test_index_bool_index(self, backend, shape):
        class IndexModel(torch.nn.Module):
            def __init__(self):
                super(IndexModel, self).__init__()

            def forward(self, x):
                return x[x > 0.5]

        model = IndexModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (1, 2),
                (3, 4, 5, 6),
            ]
        ),
    )
    def test_index_int_index_case_1(self, backend, shape):
        # all elements are selected
        class IndexModel(torch.nn.Module):
            def __init__(self):
                super(IndexModel, self).__init__()

            def forward(self, x):
                if len(shape) == 2:
                    return x[:, :]
                elif len(shape) == 4:
                    return x[:]

        model = IndexModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (1, 2),
                (3, 4, 5, 6),
            ]
        ),
    )
    def test_index_int_index_case_2(self, backend, shape):
        # only one axis is sliced
        class IndexModel(torch.nn.Module):
            def __init__(self):
                super(IndexModel, self).__init__()

            def forward(self, x):
                if len(shape) == 2:
                    index = torch.tensor([0])
                    return x[index, :]
                elif len(shape) == 4:
                    index = torch.tensor([1, 2])
                    return x[:, :, index]

        model = IndexModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (1, 2, 3),
                (2, 3, 4, 5),
            ]
        ),
    )
    def test_index_int_index_case_3(self, backend, shape):
        # only two axes are sliced, and connected
        class IndexModel(torch.nn.Module):
            def __init__(self):
                super(IndexModel, self).__init__()

            def forward(self, x):
                if len(shape) == 3:
                    index_1 = torch.tensor([0])
                    index_2 = torch.tensor([1])
                    return x[index_1, index_2, :]

                elif len(shape) == 4:
                    index_1 = torch.tensor([0, 1, 1])
                    index_2 = torch.tensor([2, 1, 0])
                    return x[:, index_1, index_2, :]

        model = IndexModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (1, 2, 3),
                (2, 3, 4, 5),
            ]
        ),
    )
    def test_index_int_index_case_4(self, backend, shape):
        # only two axes are sliced, and not connected
        class IndexModel(torch.nn.Module):
            def __init__(self):
                super(IndexModel, self).__init__()

            def forward(self, x):
                if len(shape) == 3:
                    index_1 = torch.tensor([0])
                    index_2 = torch.tensor([1])
                    return x[index_1, :,index_2]

                elif len(shape) == 4:
                    index_1 = torch.tensor([0, 1, 1])
                    index_2 = torch.tensor([3, 3, 4])
                    return x[index_1, :, :, index_2]

        model = IndexModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (1, 2, 3),
                (2, 3, 4, 5),
            ]
        ),
    )
    def test_index_int_index_case_5(self, backend, shape):
        # all axes are sliced
        class IndexModel(torch.nn.Module):
            def __init__(self):
                super(IndexModel, self).__init__()

            def forward(self, x):
                if len(shape) == 3:
                    index_1 = torch.tensor([0])
                    index_2 = torch.tensor([1])
                    index_3 = torch.tensor([2])
                    return x[index_1, index_2, index_3]

                elif len(shape) == 4:
                    index_1 = torch.tensor([0, 1, 1, 0, 0])
                    index_2 = torch.tensor([1, 2, 0, 0, 0])
                    index_3 = torch.tensor([0, 1, 2, 3, 3])
                    index_4 = torch.tensor([2, 1, 0, 4, 4])
                    return x[index_1, index_2, index_3, index_4]

        model = IndexModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (1, 2),
                (3, 4, 5, 6),
            ]
        ),
    )
    def test_index_int_index_case_6(self, backend, shape):
        # only one axis is sliced + nd mode
        class IndexModel(torch.nn.Module):
            def __init__(self):
                super(IndexModel, self).__init__()

            def forward(self, x):
                if len(shape) == 2:
                    index = torch.tensor([0,0,0,0,0,0])
                    index = index.view(2, 3)
                    return x[index, :]
                elif len(shape) == 4:
                    index = torch.tensor([0,1,2,3,0,1])
                    index = index.view(3, 2)
                    return x[:, index]

        model = IndexModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (1, 2, 3),
                (2, 3, 4, 5),
            ]
        ),
    )
    def test_index_int_index_case_7(self, backend, shape):
        # two axes are sliced, and connected + nd mode
        class IndexModel(torch.nn.Module):
            def __init__(self):
                super(IndexModel, self).__init__()

            def forward(self, x):
                if len(shape) == 3:
                    index_1 = torch.tensor([0,0,0,0,0,0,0,0]).view(4,2)
                    index_2 = torch.tensor([1,0,0,0,1,1,1,1]).view(4,2)
                    return x[index_1, index_2, :]

                elif len(shape) == 4:
                    index_1 = torch.tensor([0,0,2,2,1,1,2,0]).view(2,4)
                    index_2 = torch.tensor([0,1,2,3,0,1,2,3]).view(2,4)
                    return x[:, index_1, index_2, :]

        model = IndexModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (1, 2, 3),
                (2, 3, 4, 5),
            ]
        ),
    )
    def test_index_int_index_case_8(self, backend, shape):
        # two axes are sliced, and not connected + nd mode
        class IndexModel(torch.nn.Module):
            def __init__(self):
                super(IndexModel, self).__init__()

            def forward(self, x):
                if len(shape) == 3:
                    index_1 = torch.tensor([0,0,0,0,0,0,0,0]).view(2,4)
                    index_2 = torch.tensor([1,0,0,2,2,1,1,1]).view(2,4)
                    return x[index_1, :,index_2]

                elif len(shape) == 4:
                    index_1 = torch.tensor([0,1,1,1,1,1,0,0]).view(4,2)
                    index_2 = torch.tensor([0,1,2,3,4,0,1,2]).view(4,2)
                    return x[index_1, :, :, index_2]

        model = IndexModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (1, 2, 3),
                (2, 3, 4, 5),
            ]
        ),
    )
    def test_index_int_index_case_9(self, backend, shape):
        # one axis is sliced through bool mask
        class IndexModel(torch.nn.Module):
            def __init__(self):
                super(IndexModel, self).__init__()

            def forward(self, x):
                if len(shape) == 3:
                    return x[:, [True, False], :]

                elif len(shape) == 4:
                    return x[[True, False], :, :, :]

        model = IndexModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (1, 2, 3),
                (2, 3, 4, 5),
            ]
        ),
    )
    def test_index_int_index_case_10(self, backend, shape):
        # multiple axes are sliced through bool masks
        class IndexModel(torch.nn.Module):
            def __init__(self):
                super(IndexModel, self).__init__()

            def forward(self, x):
                if len(shape) == 3:
                    return x[[True], [True, False], [False, True, False]]

                elif len(shape) == 4:
                    return x[[True, True], :, [True, True, False, False], [True, False, False, True, False]]

        model = IndexModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

class TestPad(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, rank, mode",
        itertools.product(backends, range(3, 5), ['reflect', 'replicate'])
    )
    def test_pad_reflect_replicate(self, backend, rank: int, mode: str):
        if rank == 3:
            pad_len = 2
            input_shape = (5, 10, 10)
        elif rank == 4:
            pad_len = 4
            input_shape = (10, 5, 5, 10)
        else:
            raise NotImplementedError("Only 3D, 4D padding with non-constant padding are supported for now")
        max_pad = min(input_shape[-1], input_shape[-2])
        pad = list(np.random.randint(low=0, high=max_pad,
                                     size=pad_len))
        model = ModuleWrapper(function=torch.nn.functional.pad,
                              kwargs={"pad": pad, "mode": mode})
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, rank",
        itertools.product(backends, range(1, 6))
    )
    def test_pad_constant(self, backend, rank: int):
        if rank > 5:
            raise NotImplementedError("Only supports < 6D constant padding")
        val = float(np.random.random(1))
        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        pad_dims = np.random.randint(low=1, high=rank + 1)
        pad = list(np.random.randint(low=0, high=10,
                                     size=pad_dims * 2))
        model = ModuleWrapper(function=torch.nn.functional.pad,
                              kwargs={"pad": pad, "mode": "constant", "value": val})
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    @pytest.mark.parametrize("backend", backends)
    def test_constant_pad_1d(self, backend):
        input_shape = (3, 4, 5)
        model = torch.nn.ConstantPad1d((5, 6), 3.5).eval()
        self.run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize("backend", backends)
    def test_constant_pad_2d(self, backend):
        input_shape = (3, 4, 5, 6)
        model = torch.nn.ConstantPad2d((5, 6, 3, 8), 3.5).eval()
        self.run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize("backend", backends)
    def test_constant_pad_3d(self, backend):
        input_shape = (3, 4, 5, 6, 2)
        model = torch.nn.ConstantPad3d((5, 6, 3, 8, 2, 4), 3.5).eval()
        self.run_compare_torch(input_shape, model, backend=backend)

class TestMeshgrid(TorchBaseTest):
    @pytest.mark.parametrize(
        "rows, cols, dtype, inp_mode, backend",
        itertools.product(
            [1, 2, 3], [1, 2, 3], [torch.int, torch.float], ["norm", "list"], backends
        ),
    )
    def test_meshgrid(
        self,
        rows,
        cols,
        dtype,
        inp_mode,
        backend,
    ):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()

            def forward(self, rows, cols):
                if inp_mode == "norm":
                    return torch.meshgrid(rows, cols)
                elif inp_mode == "list":
                    return torch.meshgrid([rows, cols])
                else:
                    raise ValueError("Unsupported mode: {mode}".format(mode=inp_mode))

        inputs = (
            torch.arange(start=0, end=rows, step=1, dtype=dtype),
            torch.arange(start=0, end=cols, step=1, dtype=dtype)
        )
        model = TestModel().eval()
        expected_results = model(*inputs)
        self.run_compare_torch(
            inputs, model, expected_results, input_as_shape=False, backend=backend,
        )

class TestSacatterAdd(TorchBaseTest):
    @pytest.mark.parametrize(
        "shapes_dims, backend",
        itertools.product(
            [
                [(10,), (0, -1)],
                [(2, 3), (1, -1)],
                [(2, 3, 4, 5), (0, -2)],
            ],
            backends
        ),
    )
    def test_scatter_add(self, shapes_dims, backend):
        shapes, dims = shapes_dims
        for dim in dims:

            class TestModel(nn.Module):
                def __init__(self):
                    super(TestModel, self).__init__()
                    self.source = torch.rand(*(shapes))
                    self.index = torch.randint(0, shapes[dim], size=shapes)

                def forward(self, x):
                    return x.scatter_add_(dim, self.index, self.source)

            self.run_compare_torch(shapes, TestModel().eval(), backend=backend)

class TestBroadcastTensors(TorchBaseTest):
    @pytest.mark.parametrize(
        "shapes, backend",
        itertools.product(
            [(1,), (1, 2)],
            backends
        ),
    )
    def test_one_tensor(self, shapes, backend):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()

            def forward(self, a):
                return torch.broadcast_tensors(a)
        self.run_compare_torch(shapes, TestModel().eval(), backend=backend)

    @pytest.mark.parametrize(
        "shapes, backend",
        itertools.product(
            [
                [(2,1), (1,3)],
                [(5,1,4,1), (3,1,1)],
                [(1,), (3,1,7)],
                [(2,1), (4,3,2,1,)]
            ],
            backends
        ),
    )
    def test_two_tensors(self, shapes, backend):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()

            def forward(self, a, b):
                return torch.broadcast_tensors(a, b)
        self.run_compare_torch(shapes, TestModel().eval(), backend=backend)

    @pytest.mark.parametrize(
        "shapes, backend",
        itertools.product(
            [
                [(2,1), (1,3), (1,), (1,1)],
                [(5,1,4,1), (3,1,1), (1,), (4,8)],
                [(1,), (2,1), (3,2,1), (5,4,3,2,1)],
            ],
            backends
        ),
    )
    def test_four_tensors(self, shapes, backend):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()

            def forward(self, a, b, c, d):
                return torch.broadcast_tensors(a, b, c, d)
        self.run_compare_torch(shapes, TestModel().eval(), backend=backend)

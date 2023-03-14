# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tvm
from tvm.script import tir as T, relax as R, ir as I
from tvm import relax
import tvm.testing
import numpy as np


# fmt: off

@I.ir_module
class Module:
    @T.prim_func
    def add(rxplaceholder: T.Buffer((T.int64(8),), "float32"), rxplaceholder_1: T.Buffer((), "float32"), T_add: T.Buffer((T.int64(8),), "float32")):
        T.func_attr({"global_symbol": "add", "tir.noalias": True})
        # with T.block("root"):
        for i0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for i0_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                with T.block("T_add"):
                    ax0 = T.axis.spatial(T.int64(8), i0_fused_0 * T.int64(8) + i0_fused_1)
                    T.reads(rxplaceholder[ax0], rxplaceholder_1[()])
                    T.writes(T_add[ax0])
                    T_add[ax0] = rxplaceholder[ax0] + rxplaceholder_1[()]

    @T.prim_func
    def exp(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), compute: T.Buffer((T.int64(2), T.int64(4)), "float32")):
        T.func_attr({"global_symbol": "exp", "tir.noalias": True})
        # with T.block("root"):
        for i0_i1_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for i0_i1_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                with T.block("compute"):
                    i0 = T.axis.spatial(T.int64(2), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) // T.int64(4))
                    i1 = T.axis.spatial(T.int64(4), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) % T.int64(4))
                    T.reads(rxplaceholder[i0, i1])
                    T.writes(compute[i0, i1])
                    compute[i0, i1] = T.exp(rxplaceholder[i0, i1])

    @T.prim_func
    def pad(rxplaceholder: T.Buffer((T.int64(8),), "float32"), PadInput: T.Buffer((T.int64(10),), "float32")):
        T.func_attr({"global_symbol": "pad", "tir.noalias": True})
        # with T.block("root"):
        for i0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for i0_fused_1 in T.thread_binding(T.int64(10), thread="threadIdx.x"):
                with T.block("PadInput"):
                    i0 = T.axis.spatial(T.int64(10), i0_fused_0 * T.int64(10) + i0_fused_1)
                    T.reads(rxplaceholder[i0 - T.int64(1)])
                    T.writes(PadInput[i0])
                    PadInput[i0] = T.if_then_else(T.int64(1) <= i0 and i0 < T.int64(9), rxplaceholder[i0 - T.int64(1)], T.float32(1))

    @T.prim_func
    def reshape(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), T_reshape: T.Buffer((T.int64(8),), "float32")):
        T.func_attr({"global_symbol": "reshape", "tir.noalias": True})
        # with T.block("root"):
        for i0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for i0_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    ax0 = T.axis.spatial(T.int64(8), i0_fused_0 * T.int64(8) + i0_fused_1)
                    T.reads(rxplaceholder[T.Cast("int64", ax0) % T.int64(8) // T.int64(4), T.Cast("int64", ax0) % T.int64(4)])
                    T.writes(T_reshape[ax0])
                    T_reshape[ax0] = rxplaceholder[T.Cast("int64", ax0) % T.int64(8) // T.int64(4), T.Cast("int64", ax0) % T.int64(4)]

    @R.function
    def cuda_graph_capture_func_alloc() -> R.Tuple(R.Object, R.Tensor((2, 4), dtype="float32"), R.Object, R.Tensor((8,), dtype="float32"), R.Tensor((8,), dtype="float32")):
        R.func_attr({"global_symbol": "cuda_graph_capture_func_alloc"})
        shape_heap: R.Object = R.null_value()
        gv: R.Object = R.vm.alloc_storage(R.shape([32]), R.prim_value(0), R.dtype("float32"))
        gv1: R.Tensor(dtype="float32") = R.vm.alloc_tensor(gv, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
        gv2: R.Object = R.vm.alloc_storage(R.shape([32]), R.prim_value(0), R.dtype("float32"))
        gv3: R.Tensor(dtype="float32") = R.vm.alloc_tensor(gv2, R.prim_value(0), R.shape([8]), R.dtype("float32"))
        gv4: R.Tensor(dtype="float32") = R.vm.alloc_tensor(gv, R.prim_value(0), R.shape([8]), R.dtype("float32"))
        gv5: R.Tuple(R.Object, R.Tensor(dtype="float32"), R.Object, R.Tensor(dtype="float32"), R.Tensor(dtype="float32")) = (gv, gv1, gv2, gv3, gv4)
        gv3_1: R.Tensor(dtype="float32") = gv5[1]
        _7: R.Tuple = R.call_packed("vm.builtin.check_tensor_info", gv3_1, R.prim_value(2), R.dtype("float32"), R.str("ErrorContext(fn=cuda_graph_capture_func_alloc, loc=return, annotation=R.Tuple(R.Object, R.Tensor((2, 4), dtype=\"float32\"), R.Object, R.Tensor((8,), dtype=\"float32\"), R.Tensor((8,), dtype=\"float32\"))) "), sinfo_args=(R.Tuple,))
        gv4_1: R.Tensor(dtype="float32") = gv5[3]
        _8: R.Tuple = R.call_packed("vm.builtin.check_tensor_info", gv4_1, R.prim_value(1), R.dtype("float32"), R.str("ErrorContext(fn=cuda_graph_capture_func_alloc, loc=return, annotation=R.Tuple(R.Object, R.Tensor((2, 4), dtype=\"float32\"), R.Object, R.Tensor((8,), dtype=\"float32\"), R.Tensor((8,), dtype=\"float32\"))) "), sinfo_args=(R.Tuple,))
        gv5_1: R.Tensor(dtype="float32") = gv5[4]
        _9: R.Tuple = R.call_packed("vm.builtin.check_tensor_info", gv5_1, R.prim_value(1), R.dtype("float32"), R.str("ErrorContext(fn=cuda_graph_capture_func_alloc, loc=return, annotation=R.Tuple(R.Object, R.Tensor((2, 4), dtype=\"float32\"), R.Object, R.Tensor((8,), dtype=\"float32\"), R.Tensor((8,), dtype=\"float32\"))) "), sinfo_args=(R.Tuple,))
        _10: R.Tuple = R.call_packed("vm.builtin.match_shape", gv3_1, shape_heap, R.prim_value(2), R.prim_value(0), R.prim_value(2), R.prim_value(0), R.prim_value(4), R.str("ErrorContext(fn=cuda_graph_capture_func_alloc, loc=return, annotation=R.Tuple(R.Object, R.Tensor((2, 4), dtype=\"float32\"), R.Object, R.Tensor((8,), dtype=\"float32\"), R.Tensor((8,), dtype=\"float32\"))) "), sinfo_args=(R.Tuple,))
        _11: R.Tuple = R.call_packed("vm.builtin.match_shape", gv4_1, shape_heap, R.prim_value(1), R.prim_value(0), R.prim_value(8), R.str("ErrorContext(fn=cuda_graph_capture_func_alloc, loc=return, annotation=R.Tuple(R.Object, R.Tensor((2, 4), dtype=\"float32\"), R.Object, R.Tensor((8,), dtype=\"float32\"), R.Tensor((8,), dtype=\"float32\"))) "), sinfo_args=(R.Tuple,))
        _12: R.Tuple = R.call_packed("vm.builtin.match_shape", gv5_1, shape_heap, R.prim_value(1), R.prim_value(0), R.prim_value(8), R.str("ErrorContext(fn=cuda_graph_capture_func_alloc, loc=return, annotation=R.Tuple(R.Object, R.Tensor((2, 4), dtype=\"float32\"), R.Object, R.Tensor((8,), dtype=\"float32\"), R.Tensor((8,), dtype=\"float32\"))) "), sinfo_args=(R.Tuple,))
        return gv5

    @R.function
    def cuda_graph_capture_func_capture(allocs: R.Tuple(R.Object, R.Tensor((2, 4), dtype="float32"), R.Object, R.Tensor((8,), dtype="float32"), R.Tensor((8,), dtype="float32"))) -> R.Tuple:
        cls = Module
        R.func_attr({"global_symbol": "cuda_graph_capture_func_capture"})
        shape_heap: R.Object = R.null_value()
        _: R.Tuple = R.call_packed("vm.builtin.check_tuple_info", allocs, R.prim_value(5), R.str("ErrorContext(fn=cuda_graph_capture_func_capture, loc=param[0], param=allocs, annotation=R.Tuple(R.Object, R.Tensor((2, 4), dtype=\"float32\"), R.Object, R.Tensor((8,), dtype=\"float32\"), R.Tensor((8,), dtype=\"float32\"))) "), sinfo_args=(R.Tuple,))
        gv: R.Tensor((2, 4), dtype="float32") = allocs[1]
        _1: R.Tuple = R.call_packed("vm.builtin.check_tensor_info", gv, R.prim_value(2), R.dtype("float32"), R.str("ErrorContext(fn=cuda_graph_capture_func_capture, loc=param[0], param=allocs, annotation=R.Tuple(R.Object, R.Tensor((2, 4), dtype=\"float32\"), R.Object, R.Tensor((8,), dtype=\"float32\"), R.Tensor((8,), dtype=\"float32\"))) "), sinfo_args=(R.Tuple,))
        gv1: R.Tensor((8,), dtype="float32") = allocs[3]
        _2: R.Tuple = R.call_packed("vm.builtin.check_tensor_info", gv1, R.prim_value(1), R.dtype("float32"), R.str("ErrorContext(fn=cuda_graph_capture_func_capture, loc=param[0], param=allocs, annotation=R.Tuple(R.Object, R.Tensor((2, 4), dtype=\"float32\"), R.Object, R.Tensor((8,), dtype=\"float32\"), R.Tensor((8,), dtype=\"float32\"))) "), sinfo_args=(R.Tuple,))
        gv2: R.Tensor((8,), dtype="float32") = allocs[4]
        _3: R.Tuple = R.call_packed("vm.builtin.check_tensor_info", gv2, R.prim_value(1), R.dtype("float32"), R.str("ErrorContext(fn=cuda_graph_capture_func_capture, loc=param[0], param=allocs, annotation=R.Tuple(R.Object, R.Tensor((2, 4), dtype=\"float32\"), R.Object, R.Tensor((8,), dtype=\"float32\"), R.Tensor((8,), dtype=\"float32\"))) "), sinfo_args=(R.Tuple,))
        _4: R.Tuple = R.call_packed("vm.builtin.match_shape", gv, shape_heap, R.prim_value(2), R.prim_value(0), R.prim_value(2), R.prim_value(0), R.prim_value(4), R.str("ErrorContext(fn=cuda_graph_capture_func_capture, loc=param[0], param=allocs, annotation=R.Tuple(R.Object, R.Tensor((2, 4), dtype=\"float32\"), R.Object, R.Tensor((8,), dtype=\"float32\"), R.Tensor((8,), dtype=\"float32\"))) "), sinfo_args=(R.Tuple,))
        _5: R.Tuple = R.call_packed("vm.builtin.match_shape", gv1, shape_heap, R.prim_value(1), R.prim_value(0), R.prim_value(8), R.str("ErrorContext(fn=cuda_graph_capture_func_capture, loc=param[0], param=allocs, annotation=R.Tuple(R.Object, R.Tensor((2, 4), dtype=\"float32\"), R.Object, R.Tensor((8,), dtype=\"float32\"), R.Tensor((8,), dtype=\"float32\"))) "), sinfo_args=(R.Tuple,))
        _6: R.Tuple = R.call_packed("vm.builtin.match_shape", gv2, shape_heap, R.prim_value(1), R.prim_value(0), R.prim_value(8), R.str("ErrorContext(fn=cuda_graph_capture_func_capture, loc=param[0], param=allocs, annotation=R.Tuple(R.Object, R.Tensor((2, 4), dtype=\"float32\"), R.Object, R.Tensor((8,), dtype=\"float32\"), R.Tensor((8,), dtype=\"float32\"))) "), sinfo_args=(R.Tuple,))
        storage: R.Object = allocs[0]
        alloc: R.Tensor((2, 4), dtype="float32") = allocs[1]
        storage1: R.Object = allocs[2]
        alloc1: R.Tensor((8,), dtype="float32") = allocs[3]
        alloc2: R.Tensor((8,), dtype="float32") = allocs[4]
        _1_1: R.Tuple = cls.reshape(alloc, alloc1)
        gv0: R.Tensor((), dtype="float32") = R.const(1, "float32")
        _2_1: R.Tuple = cls.add(alloc1, gv0, alloc2)
        gv6: R.Tuple = R.tuple()
        return gv6

    @R.function
    def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
        cls = Module
        R.func_attr({"global_symbol": "main"})
        shape_heap: R.Object = R.null_value()
        _13: R.Tuple = R.call_packed("vm.builtin.check_tensor_info", x, R.prim_value(2), R.dtype("float32"), R.str("ErrorContext(fn=main, loc=param[0], param=x, annotation=R.Tensor((2, 4), dtype=\"float32\")) "), sinfo_args=(R.Tuple,))
        _14: R.Tuple = R.call_packed("vm.builtin.match_shape", x, shape_heap, R.prim_value(2), R.prim_value(0), R.prim_value(2), R.prim_value(0), R.prim_value(4), R.str("ErrorContext(fn=main, loc=param[0], param=x, annotation=R.Tensor((2, 4), dtype=\"float32\")) "), sinfo_args=(R.Tuple,))
        gv: R.Tuple(R.Object, R.Tuple(R.Object, R.Tensor((2, 4), dtype="float32"), R.Object, R.Tensor((8,), dtype="float32"), R.Tensor((8,), dtype="float32")), R.Tuple) = R.call_builtin_with_ctx("vm.builtin.get_captured_cuda_graph", (cls.cuda_graph_capture_func_alloc, cls.cuda_graph_capture_func_capture), sinfo_args=(R.Tuple(R.Object, R.Tuple(R.Object, R.Tensor((2, 4), dtype="float32"), R.Object, R.Tensor((8,), dtype="float32"), R.Tensor((8,), dtype="float32")), R.Tuple),))
        gv1: R.Tuple(R.Object, R.Tensor((2, 4), dtype="float32"), R.Object, R.Tensor((8,), dtype="float32"), R.Tensor((8,), dtype="float32")) = gv[1]
        gv2: R.Object = gv[0]
        gv3: R.Tensor((2, 4), dtype="float32") = gv1[1]
        _: R.Tuple = cls.exp(x, gv3)
        gv4: R.Tuple = R.call_packed("vm.builtin.cuda_graph_launch", gv2, sinfo_args=(R.Tuple,))
        storage: R.Object = R.vm.alloc_storage(R.shape([40]), R.prim_value(0), R.dtype("float32"))
        alloc3: R.Tensor(dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([10]), R.dtype("float32"))
        gv5: R.Tensor((8,), dtype="float32") = gv1[4]
        _3: R.Tuple = cls.pad(gv5, alloc3)
        gv6: R.Object = gv1[0]
        gv7: R.Object = gv1[2]
        _15: R.Tuple = R.call_packed("vm.builtin.check_tensor_info", alloc3, R.prim_value(1), R.dtype("float32"), R.str("ErrorContext(fn=main, loc=return, annotation=R.Tensor((10,), dtype=\"float32\")) "), sinfo_args=(R.Tuple,))
        _16: R.Tuple = R.call_packed("vm.builtin.match_shape", alloc3, shape_heap, R.prim_value(1), R.prim_value(0), R.prim_value(10), R.str("ErrorContext(fn=main, loc=return, annotation=R.Tensor((10,), dtype=\"float32\")) "), sinfo_args=(R.Tuple,))
        return alloc3

# fmt: on


def codegen(mod, target, exec_mode="bytecode"):
    builder = relax.ExecBuilder()
    leftover_mod = relax.vm_build._vmcodegen(builder, mod, exec_mode=exec_mode)
    tir_mod = relax.vm_build._filter_tir(leftover_mod)
    return relax.vm_build._vmlink(builder, target, tir_mod)


@tvm.testing.requires_cuda
def test_vm_run():
    mod = Module
    target = tvm.target.Target("cuda", host="llvm")
    ex = codegen(mod, target)
    dev = tvm.cuda(0)
    vm = relax.VirtualMachine(ex, dev)
    x_np = np.random.uniform(size=(2, 4)).astype("float32")
    y_np = np.exp(x_np)
    y_np = y_np.reshape((8,))
    y_np = y_np + 1.0
    pad_value = np.ones((1,)).astype("float32")
    y_np = np.concatenate([pad_value, y_np, pad_value], axis=0)

    x = tvm.nd.array(x_np, dev)
    y = vm["main"](x)
    tvm.testing.assert_allclose(y.asnumpy(), y_np, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()

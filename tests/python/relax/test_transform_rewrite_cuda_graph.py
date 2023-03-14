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
from tvm import relax
from tvm.script import tir as T, relax as R, ir as I
import tvm.testing


def test_rewrite_cuda_graph():
    # fmt: off
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def reshape(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), T_reshape: T.Buffer(T.int64(8), "float32")):
            # function attr dict
            T.func_attr({"tir.noalias": True, "global_symbol": "reshape"})
            # body
            # with T.block("root")
            for i0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for i0_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                    with T.block("T_reshape"):
                        ax0 = T.axis.spatial(T.int64(8), i0_fused_0 * T.int64(8) + i0_fused_1)
                        T.reads(rxplaceholder[T.Cast("int64", ax0) % T.int64(8) // T.int64(4), T.Cast("int64", ax0) % T.int64(4)])
                        T.writes(T_reshape[ax0])
                        T_reshape[ax0] = rxplaceholder[T.Cast("int64", ax0) % T.int64(8) // T.int64(4), T.Cast("int64", ax0) % T.int64(4)]

        @T.prim_func
        def exp(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), compute: T.Buffer((T.int64(2), T.int64(4)), "float32")):
            # function attr dict
            T.func_attr({"tir.noalias": True, "global_symbol": "exp"})
            # body
            # with T.block("root")
            for i0_i1_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for i0_i1_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                    with T.block("compute"):
                        i0 = T.axis.spatial(T.int64(2), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) // T.int64(4))
                        i1 = T.axis.spatial(T.int64(4), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) % T.int64(4))
                        T.reads(rxplaceholder[i0, i1])
                        T.writes(compute[i0, i1])
                        compute[i0, i1] = T.exp(rxplaceholder[i0, i1], dtype="float32")

        @T.prim_func
        def add(rxplaceholder: T.Buffer(T.int64(8), "float32"), rxplaceholder_1: T.Buffer((), "float32"), T_add: T.Buffer(T.int64(8), "float32")):
            # function attr dict
            T.func_attr({"tir.noalias": True, "global_symbol": "add"})
            # body
            # with T.block("root")
            for i0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for i0_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                    with T.block("T_add"):
                        ax0 = T.axis.spatial(T.int64(8), i0_fused_0 * T.int64(8) + i0_fused_1)
                        T.reads(rxplaceholder[ax0], rxplaceholder_1[()])
                        T.writes(T_add[ax0])
                        T_add[ax0] = rxplaceholder[ax0] + rxplaceholder_1[()]

        @T.prim_func
        def pad(rxplaceholder: T.Buffer(T.int64(8), "float32"), PadInput: T.Buffer(T.int64(10), "float32")):
            # function attr dict
            T.func_attr({"tir.noalias": True, "global_symbol": "pad"})
            # body
            # with T.block("root")
            for i0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for i0_fused_1 in T.thread_binding(T.int64(10), thread="threadIdx.x"):
                    with T.block("PadInput"):
                        i0 = T.axis.spatial(T.int64(10), i0_fused_0 * T.int64(10) + i0_fused_1)
                        T.reads(rxplaceholder[i0 - T.int64(1)])
                        T.writes(PadInput[i0])
                        PadInput[i0] = T.if_then_else(T.int64(1) <= i0 and i0 < T.int64(9), rxplaceholder[i0 - T.int64(1)], T.float32(1), dtype="float32")


        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            storage: R.Object = R.memory.alloc_storage(R.shape([32]), 0, "global", "float32")
            alloc: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, 0, R.shape([2, 4]), "float32")
            _: R.Tuple = exp(x, alloc)
            storage1: R.Object = R.memory.alloc_storage(R.shape([32]), 0, "global", "float32")
            alloc1: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(storage1, 0, R.shape([8]), "float32")
            _1: R.Tuple = reshape(alloc, alloc1)
            __1: R.Tuple = R.memory.kill_tensor(alloc)
            alloc2: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(storage, 0, R.shape([8]), "float32")
            gv0: R.Tensor((), dtype="float32") = R.const(1, "float32")
            _2: R.Tuple = add(alloc1, gv0, alloc2)
            _1_1: R.Tuple = R.memory.kill_tensor(alloc1)
            alloc3: R.Tensor((10,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), "float32", 0)
            _3: R.Tuple = pad(alloc2, alloc3)
            _2_1: R.Tuple = R.memory.kill_tensor(alloc2)
            _3_1: R.Tuple = R.memory.kill_storage(storage)
            _4: R.Tuple = R.memory.kill_storage(storage1)
            return alloc3


    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def reshape(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), T_reshape: T.Buffer(T.int64(8), "float32")):
            # function attr dict
            T.func_attr({"tir.noalias": True, "global_symbol": "reshape"})
            # body
            # with T.block("root")
            for i0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for i0_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                    with T.block("T_reshape"):
                        ax0 = T.axis.spatial(T.int64(8), i0_fused_0 * T.int64(8) + i0_fused_1)
                        T.reads(rxplaceholder[T.Cast("int64", ax0) % T.int64(8) // T.int64(4), T.Cast("int64", ax0) % T.int64(4)])
                        T.writes(T_reshape[ax0])
                        T_reshape[ax0] = rxplaceholder[T.Cast("int64", ax0) % T.int64(8) // T.int64(4), T.Cast("int64", ax0) % T.int64(4)]

        @T.prim_func
        def exp(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), compute: T.Buffer((T.int64(2), T.int64(4)), "float32")):
            # function attr dict
            T.func_attr({"tir.noalias": True, "global_symbol": "exp"})
            # body
            # with T.block("root")
            for i0_i1_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for i0_i1_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                    with T.block("compute"):
                        i0 = T.axis.spatial(T.int64(2), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) // T.int64(4))
                        i1 = T.axis.spatial(T.int64(4), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) % T.int64(4))
                        T.reads(rxplaceholder[i0, i1])
                        T.writes(compute[i0, i1])
                        compute[i0, i1] = T.exp(rxplaceholder[i0, i1], dtype="float32")

        @T.prim_func
        def add(rxplaceholder: T.Buffer(T.int64(8), "float32"), rxplaceholder_1: T.Buffer((), "float32"), T_add: T.Buffer(T.int64(8), "float32")):
            # function attr dict
            T.func_attr({"tir.noalias": True, "global_symbol": "add"})
            # body
            # with T.block("root")
            for i0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for i0_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                    with T.block("T_add"):
                        ax0 = T.axis.spatial(T.int64(8), i0_fused_0 * T.int64(8) + i0_fused_1)
                        T.reads(rxplaceholder[ax0], rxplaceholder_1[()])
                        T.writes(T_add[ax0])
                        T_add[ax0] = rxplaceholder[ax0] + rxplaceholder_1[()]

        @T.prim_func
        def pad(rxplaceholder: T.Buffer(T.int64(8), "float32"), PadInput: T.Buffer(T.int64(10), "float32")):
            # function attr dict
            T.func_attr({"tir.noalias": True, "global_symbol": "pad"})
            # body
            # with T.block("root")
            for i0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for i0_fused_1 in T.thread_binding(T.int64(10), thread="threadIdx.x"):
                    with T.block("PadInput"):
                        i0 = T.axis.spatial(T.int64(10), i0_fused_0 * T.int64(10) + i0_fused_1)
                        T.reads(rxplaceholder[i0 - T.int64(1)])
                        T.writes(PadInput[i0])
                        PadInput[i0] = T.if_then_else(T.int64(1) <= i0 and i0 < T.int64(9), rxplaceholder[i0 - T.int64(1)], T.float32(1), dtype="float32")

        @R.function
        def cuda_graph_capture_func_alloc() -> R.Tuple(R.Object, R.Tensor((2, 4), dtype="float32"), R.Object, R.Tensor((8,), dtype="float32"), R.Tensor((8,), dtype="float32")):
            gv: R.Object = R.memory.alloc_storage(R.shape([32]), 0, "global", "float32")
            gv1: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(gv, 0, R.shape([2, 4]), "float32")
            gv2: R.Object = R.memory.alloc_storage(R.shape([32]), 0, "global", "float32")
            gv3: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(gv2, 0, R.shape([8]), "float32")
            gv4: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(gv, 0, R.shape([8]), "float32")
            gv5: R.Tuple(R.Object, R.Tensor((2, 4), dtype="float32"), R.Object, R.Tensor((8,), dtype="float32"), R.Tensor((8,), dtype="float32")) = (gv, gv1, gv2, gv3, gv4)
            return gv5

        @R.function
        def cuda_graph_capture_func_capture(allocs: R.Tuple(R.Object, R.Tensor((2, 4), dtype="float32"), R.Object, R.Tensor((8,), dtype="float32"), R.Tensor((8,), dtype="float32"))) -> R.Tuple:
            # block 0
            storage: R.Object = allocs[0]
            alloc: R.Tensor((2, 4), dtype="float32") = allocs[1]
            storage1: R.Object = allocs[2]
            alloc1: R.Tensor((8,), dtype="float32") = allocs[3]
            alloc2: R.Tensor((8,), dtype="float32") = allocs[4]
            _1: R.Tuple() = reshape(alloc, alloc1)
            _2: R.Tuple() = R.memory.kill_tensor(alloc)
            gv0: R.Tensor((), dtype="float32") = R.const(1, "float32")
            _3: R.Tuple() = add(alloc1, gv0, alloc2)
            _4: R.Tuple() = R.memory.kill_tensor(alloc1)
            gv4: R.Tuple() = R.tuple()
            return gv4

        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            gv: R.Tuple(R.Object, R.Tuple(R.Object, R.Tensor((2, 4), dtype="float32"), R.Object, R.Tensor((8,), dtype="float32"), R.Tensor((8,), dtype="float32"))) = R.call_builtin_with_ctx("vm.builtin.get_captured_cuda_graph", (cuda_graph_capture_func_alloc, cuda_graph_capture_func_capture), sinfo_args=(R.Tuple(R.Object, R.Tuple(R.Object, R.Tensor((2, 4), dtype="float32"), R.Object, R.Tensor((8,), dtype="float32"), R.Tensor((8,), dtype="float32"))),))
            gv1: R.Tuple(R.Object, R.Tensor((2, 4), dtype="float32"), R.Object, R.Tensor((8,), dtype="float32"), R.Tensor((8,), dtype="float32")) = gv[1]
            gv2: R.Object = gv[0]
            gv3: R.Tensor((2, 4), dtype="float32") = gv1[1]
            _: R.Tuple = exp(x, gv3)
            gv4: R.Tuple = R.call_packed("vm.builtin.cuda_graph_launch", gv2, sinfo_args=(R.Tuple,))
            alloc3: R.Tensor((10,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), R.dtype("float32"), R.prim_value(0))
            gv5: R.Tensor((8,), dtype="float32") = gv1[4]
            _3: R.Tuple = pad(gv5, alloc3)
            _2: R.Tuple = R.memory.kill_tensor(gv5)
            gv6: R.Object = gv1[0]
            _3_1: R.Tuple = R.memory.kill_storage(gv6)
            gv7: R.Object = gv1[2]
            _4: R.Tuple = R.memory.kill_storage(gv7)
            return alloc3

    # fmt: on
    after = relax.transform.RewriteCUDAGraph()(Before)
    tvm.ir.assert_structural_equal(after, Expected)

    passes = []
    passes.append(relax.transform.VMBuiltinLower())
    passes.append(relax.transform.VMShapeLower())
    passes.append(relax.transform.AttachGlobalSymbol())
    print(tvm.transform.Sequential(passes)(after))


if __name__ == "__main__":
    tvm.testing.main()

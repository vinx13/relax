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
"""External function interface to cuBLASlt libraries."""
import tvm
from tvm import te


def matmul(lhs, rhs, transa=False, transb=False, n=0, m=0, dtype=None):
    """Create an extern op that compute matrix mult of A and rhs with cuBLAS

    Parameters
    ----------
    lhs : Tensor
        The left matrix operand
    rhs : Tensor
        The right matrix operand
    transa : bool
        Whether transpose lhs
    transb : bool
        Whether transpose rhs

    Returns
    -------
    C : Tensor
        The result tensor.
    """
    if n == 0:
        n = lhs.shape[1] if transa else lhs.shape[0]
    if m == 0:
        m = rhs.shape[0] if transb else rhs.shape[1]
    dtype = dtype if dtype is not None else lhs.dtype
    return te.extern(
        (n, m),
        [lhs, rhs],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.cublaslt.matmul", ins[0], ins[1], outs[0], transa, transb
        ),
        dtype=dtype,
        name="C",
    )

def get_cublas_lt_epilogue(epilogue):
    r = tvm.get_global_func("tvm.contrib.cublaslt.get_epilogue")(epilogue)
    print(r)
    return r

def batch_matmul(lhs, rhs, transa=False, transb=False, b=0, n=0, m=0, dtype=None, bias=None, epilogue="default"):
    """Create an extern op that compute matrix mult of A and rhs with cuBLAS

    Parameters
    ----------
    lhs : Tensor
        The left matrix operand
    rhs : Tensor
        The right matrix operand
    transa : bool
        Whether transpose lhs
    transb : bool
        Whether transpose rhs
    b : int
        The batch size
    n : int
        The size of the n dimension
    m : int
        The size of the m dimension
    dtype : str
        The output data type
    bias : Optional[Tensor]
        The optional bias
    epilogue : Literal["default", "bias", "relu", "gelu", "bias_relu", "bias_gelu"]
        The epilogue

    Returns
    -------
    C : Tensor
        The result tensor.
    """
    if n == 0:
        n = lhs.shape[2] if transa else lhs.shape[1]
    if m == 0:
        m = rhs.shape[1] if transb else rhs.shape[2]
    if b == 0:
        b = max(lhs.shape[0], rhs.shape[0])
    dtype = dtype if dtype is not None else lhs.dtype
    input_tensors = [lhs, rhs]
    if bias is not None:
        input_tensors.append(bias)

    def _get_call_packed_args(ins, outs):
        args = [ins[0], ins[1], outs[0], transa, transb]
        if epilogue != "default":
            args.append(get_cublas_lt_epilogue(epilogue))
        if bias is not None:
            args.append(ins[2])
        return args
    return te.extern(
        (b, n, m),
        input_tensors,
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.cublaslt.batch_matmul", *_get_call_packed_args(ins, outs)),
        dtype=dtype,
        name="C",
    )

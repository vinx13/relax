/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file Use external cudnn utils function
 */
#include "cublas_utils.h"

#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>

#include "../../cuda/cuda_common.h"

namespace tvm {
namespace contrib {

CuBlasThreadEntry::CuBlasThreadEntry() {
  CHECK_CUBLAS_ERROR(cublasCreate(&handle));
  CHECK_CUBLAS_ERROR(cublasLtCreate(&lt_handle));
}

CuBlasThreadEntry::~CuBlasThreadEntry() {
  if (lt_handle) {
    cublasLtDestroy(lt_handle);
    lt_handle = nullptr;
  }
  if (handle) {
    cublasDestroy(handle);
    handle = nullptr;
  }
  if (workspace) {
    cudaFree(workspace);
    workspace = nullptr;
  }
}

typedef dmlc::ThreadLocalStore<CuBlasThreadEntry> CuBlasThreadStore;

CuBlasThreadEntry* CuBlasThreadEntry::ThreadLocal() {
  auto stream = runtime::CUDAThreadEntry::ThreadLocal()->stream;
  CuBlasThreadEntry* retval = CuBlasThreadStore::Get();
  CHECK_CUBLAS_ERROR(cublasSetStream(retval->handle, static_cast<cudaStream_t>(stream)));
  return retval;
}

cublasLtEpilogue_t GetCublasLtEpilogue(const String& epilogue_name) {
  if (epilogue_name == "default") {
    return CUBLASLT_EPILOGUE_DEFAULT;
  }
  if (epilogue_name == "bias") {
    return CUBLASLT_EPILOGUE_BIAS;
  }
  if (epilogue_name == "relu") {
    return CUBLASLT_EPILOGUE_RELU;
  }
  if (epilogue_name == "bias_relu") {
    return CUBLASLT_EPILOGUE_RELU_BIAS;
  }
  if (epilogue_name == "gelu") {
    return CUBLASLT_EPILOGUE_GELU;
  }
  if (epilogue_name == "bias_gelu") {
    return CUBLASLT_EPILOGUE_GELU_BIAS;
  }
  LOG(FATAL) << "Unsupported epilogue " << epilogue_name;
  throw;
}

TVM_REGISTER_GLOBAL("tvm.contrib.cublaslt.get_epilogue").set_body_typed([](String epilogue_name) {
  return static_cast<int64_t>(GetCublasLtEpilogue(epilogue_name));
});

}  // namespace contrib
}  // namespace tvm

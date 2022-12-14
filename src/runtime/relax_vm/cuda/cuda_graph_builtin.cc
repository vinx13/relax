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
 * \file src/runtime/relax_vm/cuda_graph_builtin.cc
 * \brief The CUDA graph related builtin functions for Relax virtual machine.
 */

#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/bytecode.h>
#include <tvm/runtime/relax_vm/memory_manager.h>
#include <tvm/runtime/relax_vm/vm.h>

#include "../../cuda/cuda_common.h"
namespace tvm {
namespace runtime {
namespace relax_vm {

using tvm::runtime::NDArray;

class CUDAGraphNode : public Object {
 public:
  cudaGraph_t handle_;
  TVM_DECLARE_FINAL_OBJECT_INFO(CUDAGraphNode, Object);
};

class CUDAGraph : public ObjectRef {
 public:
  explicit CUDAGraph(cudaGraph_t handle) {
    auto n = make_object<CUDAGraphNode>();
    n->handle_ = handle;
    data_ = std::move(n);
  }
  TVM_DEFINE_OBJECT_REF_METHODS(CUDAGraph, ObjectRef, CUDAGraphNode);
};

class CUDAGraphCache : public Object {
 public:
  struct Entry {
    std::vector<ObjectRef> states;
    CUDAGraph graph;
  };
  std::unordered_map<ObjectRef, Entry, ObjectPtrHash, ObjectPtrEqual> entries;

  static CUDAGraphCache* Get() { return dmlc::ThreadLocalStore<CUDAGraphCache>::Get(); }
};

struct CaptureContext {
  cudaGraph_t graph;
  cudaStream_t stream;
  // TODO: refactor CUDA graph rewrite pass so that warm up run is not needed
  bool is_warm_up;
};

typedef dmlc::ThreadLocalStore<CaptureContext*> CaptureContextStore;

TVM_REGISTER_GLOBAL("vm.builtin.get_captured_cuda_graph")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.size(), 2);
      VirtualMachine* vm = VirtualMachine::GetContextPtr(args[0]);
      ObjectRef func = args[1];

      CUDAGraphCache* cache = CUDAGraphCache::Get();
      if (auto it = cache->entries.find(func); it == cache->entries.end()) {
        CaptureContext ctx;
        CUDA_CALL(cudaStreamCreate(&ctx.stream));
        // ctx.stream = static_cast<cudaStream_t>(device_api->CreateStream(device));
        ctx.is_warm_up = true;
        *CaptureContextStore::Get() = &ctx;

        // PackedFunc func{nullptr};
        // func = vm->GetFunction(func_name, GetObjectPtr<Object>(vm));
        // ICHECK(func != nullptr);

        // warm up
        TVMArgs func_args(nullptr, nullptr, 0);
        TVMRetValue func_rv;
        vm->InvokeClosurePacked(func, func_args, &func_rv);

        // capture
        ctx.is_warm_up = false;
        vm->InvokeClosurePacked(func, func_args, &func_rv);

        ADT tensors = func_rv;
        CUDAGraphCache::Entry entry;
        for (size_t i = 0; i < tensors.size(); ++i) {
          NDArray tensor = Downcast<NDArray>(tensors[i]);
          entry.states.push_back(tensor);
        }
        entry.graph = CUDAGraph(ctx.graph);
        cache->entries[func] = entry;
        CUDA_CALL(cudaStreamDestroy(ctx.stream));
      }
      const auto& cached = cache->entries.at(func);
      *rv = ADT::Tuple(std::vector<ObjectRef>{cached.graph, ADT::Tuple(cached.states)});
    });

TVM_REGISTER_GLOBAL("vm.builtin.cuda_graph_begin_capture").set_body_typed([]() {
  CaptureContext* capture_context = *CaptureContextStore::Get();
  ICHECK(capture_context != nullptr);
  if (capture_context->is_warm_up) {
    return;
  }
  std::swap(capture_context->stream, CUDAThreadEntry::ThreadLocal()->stream);
  CUDA_CALL(
      cudaStreamBeginCapture(CUDAThreadEntry::ThreadLocal()->stream, cudaStreamCaptureModeGlobal));
});

TVM_REGISTER_GLOBAL("vm.builtin.cuda_graph_end_capture").set_body_typed([]() {
  CaptureContext* capture_context = *CaptureContextStore::Get();
  ICHECK(capture_context != nullptr);
  if (capture_context->is_warm_up) {
    return;
  }
  CUDA_CALL(cudaStreamEndCapture(CUDAThreadEntry::ThreadLocal()->stream, &capture_context->graph));

  cudaGraphNode_t* nodes = NULL;
  size_t numNodes = 0;
  CUDA_CALL(cudaGraphGetNodes(capture_context->graph, nodes, &numNodes));
  LOG(INFO) << "Num of nodes in the cuda graph created using stream capture API = " << numNodes;
  std::swap(capture_context->stream, CUDAThreadEntry::ThreadLocal()->stream);
});

TVM_REGISTER_GLOBAL("vm.builtin.cuda_graph_launch").set_body_typed([](CUDAGraph cuda_graph) {
  cudaGraphExec_t cuda_graph_exec;
  CUDA_CALL(cudaGraphInstantiate(&cuda_graph_exec, cuda_graph->handle_, NULL, NULL, 0));
  CUDA_CALL(cudaGraphLaunch(cuda_graph_exec, CUDAThreadEntry::ThreadLocal()->stream));
  CUDA_CALL(cudaGraphExecDestroy(cuda_graph_exec));
});

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm

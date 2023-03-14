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
#include <tvm/relax/backend.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace relax {

/*! \brief A statically executed region of the graph. A statically executed region is a region where
 * all the tensors are statically allocated and only contains kernel launches (control flow is not
 * allowed).
 */
struct StaticRegion {
  /*! \brief The function to allocate the tensors and return a tuple of allocated tensors. */
  Function alloc_func;
  /*! \brief The function that launches a group of kernels where all the inputs are constants or
   * statically allocated tensors. */
  Function capture_func;
  /*! \brief The tensor to the index of the elements of the tuple return value. */
  std::vector<std::pair<const VarNode*, int>> alloc_tensor_to_index;
  /*! \brief The location where the captured graph should be launched. */
  const VarBindingNode* launch_point = nullptr;
  /*! \brief The bindings in the original function that should be removed after lifting. */
  std::unordered_map<const VarNode*, const VarBindingNode*> bindings;
};

/*! \brief Extract the static region from the function. */
class StaticRegionExtractor : public ExprVisitor {
 public:
  static std::unordered_map<const BindingBlockNode*, std::vector<StaticRegion>> Extract(
      const IRModule& mod, const Function& func) {
    StaticRegionExtractor extractor(mod, {} /*OutputCollector::Collect(func)*/);
    extractor.VisitExpr(func->body);
    return extractor.block_to_static_regions_;
  }
  StaticRegionExtractor(const IRModule& mod, const std::unordered_set<const VarNode*>& outputs)
      : mod_(mod), outputs_(outputs) {}

 private:
  struct DependencyGraph {
    void AddBinding(const VarBindingNode* binding, bool is_alloc_tensor) {
      const auto* var = binding->var.get();
      if (is_alloc_tensor) {
        alloc_tensors.push_back(var);
      } else {
        nodes.push_back(var);
      }
      bindings.emplace(var, binding);
    }

    void AddEdge(const VarNode* src, const VarNode* dst) {
      dep_src2dst[src].push_back(dst);
      dep_dst2src[dst].push_back(src);
    }

    std::unordered_map<const VarNode*, std::vector<const VarNode*>> dep_src2dst;
    std::unordered_map<const VarNode*, std::vector<const VarNode*>> dep_dst2src;
    std::vector<const VarNode*> alloc_tensors;
    std::vector<const VarNode*> nodes;
    std::unordered_map<const VarNode*, const VarBindingNode*> bindings;
  };

  /*! \brief The information of the current scope. */
  struct ScopeInfo {
    DependencyGraph graph;  // The dependency graph between bindings of the current scope.
    std::vector<StaticRegion> static_regions;  // The list of static regions in the current scope.
  };

  void VisitBindingBlock_(const BindingBlockNode* block) final {
    ScopeInfo scope;
    std::swap(scope, scope_);
    for (const auto& binding : block->bindings) {
      VisitBinding(binding);
    }
    // Summarize the detected static regions since we reach the end of the scope.
    SummarizeStaticRegion();
    std::swap(scope, scope_);
    if (scope.static_regions.size()) {
      block_to_static_regions_.emplace(block, std::move(scope.static_regions));
    }
  }

  void VisitBinding_(const MatchCastNode* binding) final {
    ExprVisitor::VisitBinding_(binding);
    SummarizeStaticRegion();
  }

  bool IsStaticAlloc(const CallNode* alloc_tensor_call) {
    LOG(INFO) << alloc_tensor_call->args[0];
    auto shape = Downcast<ShapeExpr>(alloc_tensor_call->args[0]);
    LOG(INFO) << shape;
    bool r = std::all_of(shape->values.begin(), shape->values.end(),
                         [](const PrimExpr& expr) { return expr.as<IntImmNode>() != nullptr; });
    LOG(INFO) << r;
    return r;
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    ExprVisitor::VisitBinding_(binding);

    const Expr& value = binding->value;
    // static const auto& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    static const auto& mem_alloc_storage_op = Op::Get("relax.memory.alloc_storage");
    static const auto& mem_alloc_tensor_op = Op::Get("relax.memory.alloc_tensor");
    static const auto& mem_kill_storage_op = Op::Get("relax.memory.kill_storage");
    static const auto& mem_kill_tensor_op = Op::Get("relax.memory.kill_tensor");
    if (value->IsInstance<VarNode>()) {
      // var -> var re-binding is not needed as we expect the binding is canonicalized
      return;
    }
    if (value->IsInstance<ConstantNode>()) {
      scope_.graph.AddBinding(binding, false);
      return;
    }
    if (const auto* call = value.as<CallNode>()) {
      if (call->op == mem_alloc_storage_op && IsStaticAlloc(call)) {
        LOG(INFO) << "add " << binding->var;
        scope_.graph.AddBinding(binding, true);
        return;
      } else if (call->op == Op::Get("relax.memory.alloc_tensor")) {
        scope_.graph.AddBinding(binding, true);
        return;
      } else if (const auto* gv = call->op.as<GlobalVarNode>();
                 gv != nullptr || call->op == Op::Get("relax.memory.alloc_tensor") ||
                 call->op == Op::Get("relax.memory.kill_tensor") ||
                 call->op == Op::Get("relax.memory.kill_storage")

      ) {
        if (gv != nullptr) {
          auto func = mod_->Lookup(GetRef<GlobalVar>(gv));
          if (!func->IsInstance<tir::PrimFuncNode>()) {
            return;
          }
        }

        std::function<bool(const Expr&)> f_is_arg_static = [&](const Expr& arg) -> bool {
          if (const auto* var = arg.as<VarNode>()) {
            if (!scope_.graph.bindings.count(var)) {
              return false;
            }
            return scope_.graph.bindings.count(var);
          } else if (arg->IsInstance<PrimValueNode>() || arg->IsInstance<IntImmNode>() ||
                     arg->IsInstance<FloatImmNode>() || arg->IsInstance<ConstantNode>() ||
                     arg->IsInstance<DataTypeImmNode>()) {
            return true;
          } else if (arg->IsInstance<ShapeExprNode>()) {
            auto shape = Downcast<ShapeExpr>(arg);
            return std::all_of(
                shape->values.begin(), shape->values.end(),
                [](const PrimExpr& prim_expr) { return prim_expr.as<IntImmNode>() != nullptr; });
          } else if (arg->IsInstance<TupleNode>()) {
            auto tuple = arg.as<TupleNode>();
            return std::all_of(tuple->fields.begin(), tuple->fields.end(), f_is_arg_static);
          }
          return false;
        };
        bool is_all_args_static =
            std::all_of(call->args.begin(), call->args.end(), f_is_arg_static);
        if (is_all_args_static) {
          scope_.graph.AddBinding(binding, false);
          for (const Expr& arg : call->args) {
            if (const auto* var = arg.as<VarNode>()) {
              scope_.graph.AddEdge(var, binding->var.get());
            }
          }
          return;
        }
      }
    }
    SummarizeStaticRegion();
  }

  Function EmitStaticAllocations(BlockBuilderNode* builder, StaticRegion* region) {
    builder->BeginBindingBlock();
    Map<Var, Var> var_remap;
    Array<Expr> alloc_storage_or_tensors;
    static const auto& mem_alloc_tensor_op = Op::Get("relax.memory.alloc_tensor");
    for (const auto& var : scope_.graph.alloc_tensors) {
      region->alloc_tensor_to_index.emplace_back(var, alloc_storage_or_tensors.size());
      Call alloc = Downcast<Call>(scope_.graph.bindings.at(var)->value);
      if (alloc->op.same_as(mem_alloc_tensor_op)) {
        alloc.CopyOnWrite()->args.Set(0, var_remap.at(Downcast<Var>(alloc->args[0])));
      }

      auto new_alloc = builder->Emit(alloc);
      var_remap.Set(GetRef<Var>(var), new_alloc);
      alloc_storage_or_tensors.push_back(new_alloc);
    }
    auto output = builder->Emit(Tuple(alloc_storage_or_tensors));
    auto block = builder->EndBlock();
    auto func_body = builder->Normalize(SeqExpr({block}, output));
    auto func = Function({}, func_body, Downcast<StructInfo>(output->struct_info_.value()));
    return func;
  }

  Function EmitStaticGraph(BlockBuilderNode* builder, StaticRegion* region) {
    Var allocs{"allocs", region->alloc_func->ret_struct_info};
    builder->BeginBindingBlock();
    for (int i = 0; i < static_cast<int>(scope_.graph.alloc_tensors.size()); ++i) {
      const VarNode* tensor = scope_.graph.alloc_tensors[i];
      const Expr alloc = builder->Normalize(TupleGetItem(allocs, i));
      builder->EmitNormalized(VarBinding(GetRef<Var>(tensor), alloc));
    }
    for (const VarNode* node : scope_.graph.nodes) {
      const VarBindingNode* binding = scope_.graph.bindings.at(node);
      builder->EmitNormalized(GetRef<VarBinding>(binding));
    }
    auto void_output = builder->Emit(Tuple(Array<Expr>{}));
    auto block = builder->EndBlock();
    auto func_body = builder->Normalize(SeqExpr({block}, void_output));
    auto func = Function({allocs}, func_body, TupleStructInfo(Array<StructInfo>{}));
    return func;
  }

  void SummarizeStaticRegion() {
    LOG(INFO) << "End of Scope";

    // Find all bindings other than alloc_tensors to emit
    std::vector<VarBinding> bindings_to_emit;
    bool has_kernel_launch = false;
    for (const auto* var : scope_.graph.nodes) {
      auto it = scope_.graph.bindings.find(var);
      const VarBindingNode* binding = it->second;
      if (const CallNode* call = binding->value.as<CallNode>();
          call != nullptr && call->op.as<GlobalVarNode>()) {
        has_kernel_launch = true;
      }
      bindings_to_emit.push_back(GetRef<VarBinding>(binding));
    }
    if (!has_kernel_launch) return;

    BlockBuilder static_region_builder = BlockBuilder::Create(mod_);

    StaticRegion region;
    // Emit alloc_tensors as a separate function
    region.alloc_func = EmitStaticAllocations(static_region_builder.operator->(), &region);
    // Emit the rest of the graph, with original alloc_tensors replaced with the function parameters
    region.capture_func = EmitStaticGraph(static_region_builder.operator->(), &region);
    region.launch_point = scope_.graph.bindings.at(scope_.graph.nodes.front());
    region.bindings = std::move(scope_.graph.bindings);

    scope_.static_regions.push_back(std::move(region));
    scope_.graph = DependencyGraph();
  }

  IRModule mod_;
  std::unordered_set<const VarNode*> outputs_;
  ScopeInfo scope_;
  std::unordered_map<const BindingBlockNode*, std::vector<StaticRegion>> block_to_static_regions_;
};

/*! \brief Lift the static regions to separate functions to be run in CUDA graph capture mode.
 * Replace the original kernel calls with CUDA graph launches. */
class CUDAGraphRewriter : public ExprMutator {
 public:
  explicit CUDAGraphRewriter(IRModule mod) : ExprMutator(mod) {}

  IRModule Rewrite() {
    auto mod = builder_->GetContextIRModule();
    for (const auto& [gv, func] : mod->functions) {
      if (const auto* func_node = func.as<FunctionNode>()) {
        binding_block_to_regions_ =
            StaticRegionExtractor::Extract(mod, GetRef<Function>(func_node));
        Function new_func = Downcast<Function>(func);
        auto fptr = new_func.CopyOnWrite();
        fptr->body = VisitExpr(std::move(fptr->body));
        builder_->UpdateFunction(gv, new_func);
      }
    }
    return builder_->GetContextIRModule();
  }

  Expr MakeGetCapturedCUDAGraph(const GlobalVar& gv_alloc, const GlobalVar& gv_capture,
                                const StructInfo& alloc_tensors_sinfo) {
    return Call(call_builtin_with_ctx_op_,
                {builtin_get_captured_cuda_graph_, Tuple({gv_alloc, gv_capture})}, Attrs(),
                {TupleStructInfo({ObjectStructInfo(), alloc_tensors_sinfo})});
  }

  BindingBlock VisitBindingBlock_(const BindingBlockNode* binding_block) final {
    builder_->BeginBindingBlock();
    if (auto it = binding_block_to_regions_.find(binding_block);
        it != binding_block_to_regions_.end()) {
      for (const auto& region : it->second) {
        auto gv_alloc = builder_->AddFunction(region.alloc_func, "cuda_graph_capture_func_alloc");
        auto gv_capture =
            builder_->AddFunction(region.capture_func, "cuda_graph_capture_func_capture");
        auto graph_tensors = builder_->Emit(
            MakeGetCapturedCUDAGraph(gv_alloc, gv_capture, region.alloc_func->ret_struct_info));

        for (const auto& binding : region.bindings) {
          bindings_to_remove_.insert(binding.first);
        }
        auto tensors = builder_->Emit(TupleGetItem(graph_tensors, 1));
        for (const auto& [var, index] : region.alloc_tensor_to_index) {
          // TupleGetItem is emitted lazily because not all the tensors are needed by the rest of
          // the program
          auto tensor = TupleGetItem(tensors, index);
          LOG(INFO) << "var_remap_[" << GetRef<Var>(var) << "] = " << tensor;
          var_remap_[var] = {nullptr, std::move(tensor)};
        }
        auto graph = builder_->Emit(TupleGetItem(graph_tensors, 0));
        graph_launch_point_[region.launch_point->var.get()] = graph.get();
      }
    }
    for (const auto& binding : binding_block->bindings) {
      VisitBinding(binding);
    }
    return builder_->EndBlock();
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    if (auto it = graph_launch_point_.find(binding->var.get()); it != graph_launch_point_.end()) {
      // Emit launch graph
      builder_->Emit(Call(builtin_cuda_graph_launch_, {GetRef<Var>(it->second)}, Attrs(),
                          {TupleStructInfo(Array<StructInfo>{})}));
      return;
    } else if (bindings_to_remove_.count(binding->var.get())) {
      return;
    }
    ExprMutator::VisitBinding_(binding);
  }

  Expr VisitExpr_(const VarNode* var) final {
    if (auto it = var_remap_.find(var); it != var_remap_.end()) {
      auto& [var, expr] = it->second;
      if (var == nullptr) {
        var = builder_->Emit(expr).get();
      }
      return GetRef<Var>(var);
    }
    return GetRef<Expr>(var);
  }

 private:
  std::unordered_set<const VarNode*> bindings_to_remove_;
  std::unordered_map<const VarNode*, std::pair<const VarNode*, Expr>> var_remap_;
  std::unordered_map<const VarNode*, const VarNode*> graph_launch_point_;
  std::unordered_map<const BindingBlockNode*, std::vector<StaticRegion>> binding_block_to_regions_;
  const Op& call_builtin_with_ctx_op_ = Op::Get("relax.call_builtin_with_ctx");
  const ExternFunc builtin_get_captured_cuda_graph_{"vm.builtin.get_captured_cuda_graph"};
  const ExternFunc builtin_cuda_graph_launch_{"vm.builtin.cuda_graph_launch"};
};

IRModule RewriteCUDAGraph(IRModule mod) {
  CUDAGraphRewriter rewriter(mod);
  mod = rewriter.Rewrite();
  LOG(INFO) << mod;
  return mod;
}

namespace transform {

Pass RewriteCUDAGraph() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule m, PassContext pc) { return ::tvm::relax::RewriteCUDAGraph(std::move(m)); };
  return CreateModulePass(pass_func, 0, "RewriteCUDAGraph", {});
}

TVM_REGISTER_GLOBAL("relax.transform.RewriteCUDAGraph").set_body_typed(RewriteCUDAGraph);

}  // namespace transform

}  // namespace relax
}  // namespace tvm

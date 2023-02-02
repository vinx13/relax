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
 * \file tvm/relax/transform.h
 * \brief Relax specific transformation passes.
 */
#ifndef TVM_RELAX_TRANSFORM_H_
#define TVM_RELAX_TRANSFORM_H_

#include <tvm/ir/transform.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/tir_pattern.h>

namespace tvm {
namespace relax {
namespace transform {

using Pass = tvm::transform::Pass;
using PassInfo = tvm::transform::PassInfo;
using PassContext = tvm::transform::PassContext;
using Function = tvm::relax::Function;
using DataflowBlock = tvm::relax::DataflowBlock;

/*!
 * \brief Create a function pass.
 *
 * \param pass_func The packed function that contains the optimization.
 * \param opt_level The optimization level of the function pass.
 * \param name The name of the function pass.
 * \param required The list of the passes that the function pass is dependent on.
 *
 * \return The created function pass.
 */
TVM_DLL Pass CreateFunctionPass(
    const runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)>& pass_func,
    int opt_level, String name, tvm::Array<String> required, bool traceable = false);

/*!
 * \brief Create a dataflowblock pass.
 *
 * \param pass_func The packed function that contains the optimization.
 * \param opt_level The optimization level of the dataflowblock pass.
 * \param name The name of the dataflowblock pass.
 * \param required The list of the passes that the dataflowblock pass is dependent on.
 *
 * \return The created dataflowblock pass.
 */
TVM_DLL Pass CreateDataflowBlockPass(
    const runtime::TypedPackedFunc<DataflowBlock(DataflowBlock, IRModule, PassContext)>& pass_func,
    int opt_level, String name, tvm::Array<String> required, bool traceable = false);

/*!
 * \brief Incorrectly transform the dataflow structure as fail testcases.
 *
 * \return The Pass.
 */
TVM_DLL Pass FailTestRewrite();

/*!
 * \brief Perform fused multiply add rewriting in dataflow blocks.
 *
 * \return The Pass.
 */
TVM_DLL Pass FMARewrite();

/*!
 * \brief Perform lambda lifting to lift functions from nested into global.
 *
 * \return The Pass.
 */
TVM_DLL Pass LambdaLift();

/*!
 * \brief Transform all dataflow structure to non-dataflow version.
 *
 * \return The Pass.
 */
TVM_DLL Pass ToNonDataflow();

/*!
 * \brief Perform explicit tensor allocation for call_tir.
 *
 * \return The Pass.
 */
TVM_DLL Pass CallTIRRewrite();

/*!
 * \brief Convert all reshape-like call_tir to VM reshape operator call.
 * The VM reshape operator calls will be further lowered to a CreateView
 * operation at runtime, instead of doing real data copy.
 * Here "reshape-like" includes reshape, expand_dims, flatten, etc.
 *
 * \return The Pass.
 */
TVM_DLL Pass RewriteDataflowReshape();

/*!
 * \brief Attach global_symbol to Relax functions and TIR Primfuncs for codegen.
 *
 * \return The Pass.
 */
TVM_DLL Pass AttachGlobalSymbol();

/*!
 * \brief Simplify a Relax module by folding var bindings and match shape nodes.
 * May include other forms of expression simplification in the future.
 * Best used alongside constant folding and eliminating unused bindings.
 *
 * \return The Pass.
 */
TVM_DLL Pass CanonicalizeBindings();

/*!
 * \brief Transform Relax IR to normal form: transform AST to A-normal form, and fill the
 * checked_type_ and shape_ of expressions.
 *
 * \return The Pass.
 */
TVM_DLL Pass Normalize();

/*!
 * \brief Bind params of function of the module to constant tensors.
 *
 * \param func_name The name of the function to bind parameters.
 * \param params The parameters to bind.
 *
 * \return The Pass.
 */
TVM_DLL Pass BindParams(String name, Map<String, runtime::NDArray> params);

/*!
 * \brief Fold constant expressions.
 *
 * \return The Pass.
 */
TVM_DLL Pass FoldConstant();

/*!
 * \brief Annotate Op Pattern Kind for TIR functions, which is used in FuseOps.
 * \note It is an auto-detect pass for "unscheduled prim_funcs", the op_pattern will be
 *       "opaque" of we can't detect it. Users can manually annotate the attr `op_pattern`
 *       to prim_func.
 * \return The Pass.
 */
TVM_DLL Pass AnnotateTIROpPattern();

/*!
 * \brief This pass groups bindings in a dataflow block of Relax functions and generates a new
 * grouped Relax function for each group, according to the fusion algorithm described in the pass
 * implementation. By grouping bindings into new Relax functions, we substitute the bindings in the
 * function being manipulated into function calls to the new grouped function.
 *
 * A follow-up pass named "FuseTIR" will generate a TIR PrimFunc for each grouped function.
 * \param fuse_opt_level The level of fuse optimization.
 *        -1 indicates that the level will be inferred from pass context.
 * \return The Pass.
 */
TVM_DLL Pass FuseOps(int fuse_opt_level = -1);

/*!
 * \brief Apply pattern matching to each function in the given module, and group matched
 * expressions into a new function. The end result is similar to FuseOps, but fusion is driven
 * completely by the provided patterns.
 *
 * \param pattern_names The name of each pattern. It becomes the value of the kComposite attribute
 * of a fused function after successful matching.
 * \param patterns The patterns to detect. The order of the patterns determines the order
 * of priority in which they are matched. Higher-priority patterns should come earlier in the list.
 * \return The Pass.
 */
TVM_DLL Pass FuseOpsByPattern(const tvm::Array<runtime::String>& pattern_names,
                              const tvm::Array<DFPattern>& patterns);

/*!
 * \brief Group one or multiple composite functions created by FuseOpsByPattern into a new
 *  function. The new function will be annotated with kCodegen and GlobalSymbol attributes,
 *  and it is intented to be offloaded to an external backend.
 *
 *  Even if there is only one composite function, or a backend does not benefit from receiving
 *  larger subgraphs, this pass is required to run for offloading (BYOC) since a composite function
 *  needs to be wrapped by an outer function that are annotated with "Codegen" and "global_symbol"
 *  attributes.
 *
 * \return The Pass.
 */
TVM_DLL Pass MergeCompositeFunctions();

/*!
 * \brief Fuse relax sub-function into a larger TIR function if possible.
    this pass works together with FuseOps to perform operator fusion.

 * \return The Pass.
 */
TVM_DLL Pass FuseTIR();

/*!
 * \brief Remove unused global relax functions in an IRModule.
 * \param entry_functions list of entry functions
 * \return The Pass.
 */
TVM_DLL Pass RemoveUnusedFunctions(Array<runtime::String> entry_functions);

/*!
 * \brief Run codegen.
 * \param target_options pairs of target name and compilation options
 * \param entry_functions list of entry functions
 * \return The Pass.
 */
TVM_DLL Pass RunCodegen(Optional<Map<String, Map<String, ObjectRef>>> target_options,
                        Array<runtime::String> entry_functions);

/*!
 * \brief Automatic mixed precision pass. Currently the pass assumes the input module to be fp32
 * only, and will automatically cast fp32 to fp16 for certain ops.
 * \param out_dtype The output data type of gemm/conv, which is the data type of the accumulator.
 * \return The Pass.
 */
TVM_DLL Pass ToMixedPrecision(const DataType& out_dtype);

/*!
 * \brief Automatic layout conversion pass.
 * \param desired_layouts The desired layouts for some operators.
 * \return The Pass.
 */
TVM_DLL Pass ConvertLayout(Map<String, Array<String>> desired_layouts);

/*!
 * \brief Reverse-mode automatic differentiation.
 *
 * Now only supports differentiating one function in the IRModule with one dataflow block
 * with respect to the only return value of the function, which needs to be scalar.
 *
 * For a given function specified by the input global var, it generates a new function with the name
 * `[name of original function] + "_adjoint"`. The new function computes the adjoints of the
 * specified arguments of the original function with respect to the only one return value of the
 * original function.
 *
 * For examples, see the MLP examples in `tests/python/relax/test_transform_gradient.py` and
 * `tests/python/relax/test_transform_gradient_numeric.py`.
 *
 * \param global_var The GlobalVar of the specified function.
 * \param require_grads The relax variables whose adjoints are needed. Must be parameters of the
 * given function. If it is not specified, adjoints of all arguments would be computed.
 * \return The Pass.
 */
TVM_DLL Pass Gradient(GlobalVar global_var, Optional<Array<Var>> require_grads = NullOpt);

/*!
 * \brief Split a PrimFunc into 2 parts: the first part is a TIR PrimFunc which is
 *        matched with some cutlass kernels, and the second part is the rest of the
 *        original PrimFunc that is not fused with cutlass kernels.
 * \param patterns The list of TIR patterns.
 * \param fcodegen The function to generate the TIR PrimFunc for the matched pattern.
 * \return The Pass.
 */
TVM_DLL Pass SplitCallTIRByPattern(Array<TIRPattern> patterns, FCodegen fcodegen);

}  // namespace transform
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORM_H_

// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Utilities/DataFlow.h"
#include "llvm/ADT/TypeSwitch.h"

namespace zirgen::Zll {

using namespace mlir;
using namespace dsl;

class Degree : public LatticeValue<unsigned, Degree> {
public:
  using LatticeValue::LatticeValue;

  void addContribution(Value val) { contributions.push_back(val); }

  // Track which program points were used to compute this particular degree to
  // use in diagnostics
  SmallVector<Value, 2> contributions;
};

class DegreeLattice : public PointLattice<Degree> {
public:
  using PointLattice::PointLattice;

  void addContribution(Value val) { getValue().addContribution(val); }
};

class DegreeAnalysis : public OperationDataflowAnalysis {
public:
  using Element = DegreeLattice;

  using OperationDataflowAnalysis::OperationDataflowAnalysis;

  LogicalResult visitOperation(Operation* op) override {
    TypeSwitch<Operation*>(op)
        .Case<ConstOp,
              GetOp,
              GetGlobalOp,
              AddOp,
              SubOp,
              NegOp,
              MulOp,
              PowOp,
              EqualZeroOp,
              TrueOp,
              AndEqzOp,
              AndCondOp,
              IfOp,
              CallableOpInterface>([&](auto op) { visitOp(op); })
        .Default([&](Operation* op) {
          if (op->hasTrait<OpTrait::ReturnLike>())
            return visitReturnLikeOp(op);
        });
    return success();
  }

private:
  void visitOp(ConstOp op);
  void visitOp(GetOp op);
  void visitOp(GetGlobalOp op);
  void visitOp(AddOp op);
  void visitOp(SubOp op);
  void visitOp(NegOp op);
  void visitOp(MulOp op);
  void visitOp(PowOp op);
  void visitOp(EqualZeroOp op);
  void visitOp(TrueOp op);
  void visitOp(AndEqzOp op);
  void visitOp(AndCondOp op);
  void visitOp(IfOp op);
  void visitOp(CallableOpInterface op);
  void visitReturnLikeOp(Operation* op);
};

} // namespace zirgen::Zll

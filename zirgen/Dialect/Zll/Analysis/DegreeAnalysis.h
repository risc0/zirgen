// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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

  void visitOperation(Operation* op) override {
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

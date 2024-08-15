// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/Zll/Analysis/DegreeAnalysis.h"

namespace zirgen::ZStruct {

using namespace mlir;
using namespace dataflow;

class DegreeAnalysis : public Zll::DegreeAnalysis {
public:
  using Zll::DegreeAnalysis::DegreeAnalysis;
  using Zll::DegreeAnalysis::Element;

  void visitOperation(Operation* op) override {
    TypeSwitch<Operation*>(op)
        .Case<LookupOp, SubscriptOp, LoadOp, BindLayoutOp>([&](auto op) { visitOp(op); })
        .Default([&](auto op) { Zll::DegreeAnalysis::visitOperation(op); });
  }

private:
  void visitOp(LookupOp op);
  void visitOp(SubscriptOp op);
  void visitOp(LoadOp op);
  void visitOp(BindLayoutOp op);
};

} // namespace zirgen::ZStruct

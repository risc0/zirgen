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

#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/Zll/Analysis/DegreeAnalysis.h"

namespace zirgen::ZStruct {

using namespace mlir;
using namespace dataflow;

class DegreeAnalysis : public Zll::DegreeAnalysis {
public:
  using Zll::DegreeAnalysis::DegreeAnalysis;
  using Zll::DegreeAnalysis::Element;

  LogicalResult visitOperation(Operation* op) override {
    return TypeSwitch<Operation*, LogicalResult>(op)
        .Case<LookupOp, SubscriptOp, LoadOp, BindLayoutOp>([&](auto op) {
          visitOp(op);
          return success();
        })
        .Default([&](auto op) { return Zll::DegreeAnalysis::visitOperation(op); });
  }

private:
  void visitOp(LookupOp op);
  void visitOp(SubscriptOp op);
  void visitOp(LoadOp op);
  void visitOp(BindLayoutOp op);
};

} // namespace zirgen::ZStruct

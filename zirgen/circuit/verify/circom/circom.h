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

#pragma once

#include <iostream>

#include "zirgen/Dialect/IOP/IR/IR.h"
#include "zirgen/Dialect/Zll/IR/IR.h"

namespace zirgen::snark {

class CircomGenerator {
public:
  CircomGenerator(std::ostream& outs);
  void emit(mlir::func::FuncOp func, bool encodeOutput);

private:
  void emitHeader();
  void emitFooter();
  void emit(Zll::ConstOp op);
  void emit(Zll::NegOp op);
  void emitBinary(char symbol, mlir::Operation& op);
  void emit(Iop::CommitOp op);
  void emit(Iop::ReadOp op);
  void emit(Iop::RngBitsOp op);
  void emit(Iop::RngValOp op);
  void emit(Zll::EqualZeroOp op);
  void emit(Zll::HashAssertEqOp op);
  void emit(Zll::SelectOp op);
  void emit(Zll::NormalizeOp op);

  std::ostream& outs;
  size_t iopOffset = 0;
  size_t idCount = 0;
  llvm::DenseMap<mlir::Value, std::string> signal;
  std::string curIop;
};

} // namespace zirgen::snark

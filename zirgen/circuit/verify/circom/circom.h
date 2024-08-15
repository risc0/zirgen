// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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

// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/Dialect/IOP/IR/IR.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"

using zirgen::Zll::BigIntRange;

namespace zirgen::Iop {

#define GET(x) get(ranges, x)
#define SET(x, y) set(ranges, x, y)

bool ReadOp::updateRanges(mlir::DenseMap<mlir::Value, BigIntRange>& ranges) {
  for (mlir::Value v : getOuts()) {
    if (mlir::isa<Zll::ValType>(v.getType())) {
      SET(v, BigIntRange::rangeP());
    }
  }
  return true;
}

bool CommitOp::updateRanges(mlir::DenseMap<mlir::Value, BigIntRange>& ranges) {
  return true;
}

bool RngBitsOp::updateRanges(mlir::DenseMap<mlir::Value, BigIntRange>& ranges) {
  SET(getOut(), BigIntRange(0, (uint64_t(1) << getBits()) - 1));
  return true;
}

bool RngValOp::updateRanges(mlir::DenseMap<mlir::Value, BigIntRange>& ranges) {
  SET(getOut(), BigIntRange::rangeP());
  return true;
}

#undef GET
#undef SET

mlir::LogicalResult ReadOp::evaluate(Zll::Interpreter& interp,
                                     llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                     EvalAdaptor& adaptor) {
  auto iop = adaptor.getIop()->getIop();
  if (auto valType = llvm::dyn_cast<Zll::ValType>(getOuts()[0].getType())) {
    size_t k = valType.getFieldK();
    std::vector<uint32_t> arr(k * getOuts().size());
    iop->read(arr.data(), arr.size());
    for (size_t i = 0; i < arr.size(); i++) {
      arr[i] = (uint64_t(arr[i]) * kBabyBearFromMontgomery) % kBabyBearP;
    }
    for (size_t i = 0; i < getOuts().size(); i++) {
      llvm::SmallVector<uint64_t, 4> poly(k);
      for (size_t j = 0; j < k; j++) {
        if (getFlip()) {
          poly[j] = arr[i * k + j];
        } else {
          poly[j] = arr[j * getOuts().size() + i];
        }
      }
      outs[i]->setVal(poly);
    }
  } else {
    for (size_t i = 0; i < getOuts().size(); i++) {
      Digest digest;
      iop->read(&digest, 1);
      outs[i]->setDigest(digest);
    }
  }
  return mlir::success();
}

mlir::LogicalResult CommitOp::evaluate(Zll::Interpreter& interp,
                                       llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                       EvalAdaptor& adaptor) {
  adaptor.getIop()->getIop()->commit(adaptor.getDigest()->getDigest());
  return mlir::success();
}

mlir::LogicalResult RngBitsOp::evaluate(Zll::Interpreter& interp,
                                        llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                        EvalAdaptor& adaptor) {
  uint64_t rng = adaptor.getIop()->getIop()->generateBits(getBits());
  outs[0]->setVal(rng);
  return mlir::success();
}

mlir::LogicalResult RngValOp::evaluate(Zll::Interpreter& interp,
                                       llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                       EvalAdaptor& adaptor) {
  size_t k = llvm::cast<Zll::ValType>(getOut().getType()).getFieldK();
  llvm::SmallVector<uint64_t, 4> poly;
  for (size_t i = 0; i < k; i++) {
    poly.push_back(adaptor.getIop()->getIop()->generateFp());
  }
  outs[0]->setVal(poly);
  return mlir::success();
}

} // namespace zirgen::Iop

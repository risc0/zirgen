// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/compiler/codegen/codegen.h"

#include <fstream>

#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/circuit/recursion/encode.h"

using namespace mlir;

namespace zirgen {
namespace {

std::unique_ptr<llvm::raw_fd_ostream> openOutputFile(const std::string& path,
                                                     const std::string& name) {
  std::string filename = path + "/" + name;
  std::error_code ec;
  auto ofs = std::make_unique<llvm::raw_fd_ostream>(filename, ec);
  if (ec) {
    throw std::runtime_error("Unable to open file: " + filename);
  }
  return ofs;
}

} // namespace

void emitRecursion(const std::string& path, func::FuncOp func, recursion::EncodeStats* stats) {
  std::string name = func.getName().str();
  llvm::errs() << name << "\n";
  auto ofs = openOutputFile(path, name + ".zkr");

  llvm::DenseMap<Value, uint64_t> toId;
  auto encoded = recursion::encode(recursion::HashType::POSEIDON2, &func.front(), &toId, stats);
  ofs->write(reinterpret_cast<const char*>(encoded.data()), encoded.size() * sizeof(uint32_t));

  std::map<uint64_t, mlir::Location> locs;
  for (const auto& elem : toId) {
    locs.emplace(elem.second, elem.first.getLoc());
  }
  auto debugOfs = openOutputFile(path, name + ".zkr.dbg");
  for (const auto& elem : locs) {
    *debugOfs << elem.first << " <- " << elem.second << "\n";
  }
}

} // namespace zirgen

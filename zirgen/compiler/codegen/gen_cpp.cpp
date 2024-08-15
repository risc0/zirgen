// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/compiler/codegen/codegen.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/Zll/IR/IR.h"

using namespace mlir;
using namespace zirgen::Zll;

namespace zirgen {

namespace {

struct CppStreamEmitterImpl : CppStreamEmitter {
  llvm::raw_ostream& ofs;
  llvm::DenseMap<Value, std::string> names;
  size_t nextVal;
  size_t indent;

  CppStreamEmitterImpl(llvm::raw_ostream& ofs) : ofs(ofs), nextVal(0), indent(0) {}

  void header(func::FuncOp func) {
    ofs << "// This code is automatically generated\n\n";
    ofs << "#include \"impl.h\"\n\n";
    ofs << "using namespace risc0;\n\n";
    ofs << "namespace circuit::" << func.getName() << " {\n\n";
  }

  void footer(func::FuncOp func) { ofs << "}  // namespace circuit::" << func.getName() << "\n"; }

  void doIndent() {
    for (size_t i = 0; i < indent; i++) {
      ofs << "  ";
    }
  }

  void emitGeneric(const char* prefix, Operation* op) {
    doIndent();
    if (op->getNumResults() > 0) {
      ofs << "auto ";
    }
    if (op->getNumResults() > 1) {
      ofs << "[";
    }
    for (size_t i = 0; i < op->getNumResults(); i++) {
      std::string outName = "val" + std::to_string(nextVal++);
      names[op->getResult(i)] = outName;
      ofs << outName;
      if (i != op->getNumResults() - 1) {
        ofs << ", ";
      }
    }
    if (op->getNumResults() > 1) {
      ofs << "]";
    }
    if (op->getNumResults() > 0) {
      ofs << " = ";
    }
    ofs << prefix << "_" << op->getName().stripDialect() << "(";
    // Special case for get
    if (isa<GetOp>(op)) {
      ofs << "u, ";
      ofs << op->getAttrOfType<IntegerAttr>("tap").getUInt() << ", ";
      ofs << "\"" << getLocString(op->getLoc()) << "\");\n";
      return;
    }
    for (size_t i = 0; i < op->getNumOperands(); i++) {
      ofs << names[op->getOperand(i)];
      ofs << ", ";
    }
    for (const char* attrName : {"coefficients", "value", "offset", "back", "tap"}) {
      if (auto attr = op->getAttrOfType<IntegerAttr>(attrName)) {
        ofs << attr.getUInt() << ", ";
      }
      if (auto attr = op->getAttrOfType<PolynomialAttr>(attrName)) {
        ofs << "{";
        for (uint64_t elem : attr.asArrayRef()) {
          ofs << elem << ", ";
        }
        ofs << "}, ";
      }
    }
    ofs << "\"" << getLocString(op->getLoc()) << "\"";
    ofs << ");\n";
  }

  virtual void emitPoly(func::FuncOp func) override {
    header(func);
    emitPolyFunc("poly_edsl", "Val", "MixState", func);
    footer(func);
  }

  void
  emitPolyFunc(const char* prefix, const char* argType, const char* retType, func::FuncOp func) {
    ofs << retType << " CircuitImpl::" << prefix << "(\n";
    ofs << "    llvm::ArrayRef<Val> u,\n";
    ofs << "    llvm::ArrayRef<Val> out,\n";
    ofs << "    llvm::ArrayRef<Val> mix) const {\n";
    names[func.getArgument(1)] = "out";
    names[func.getArgument(3)] = "mix";
    indent++;
    for (Operation& op : func.front().without_terminator()) {
      emitGeneric(prefix, &op);
    }
    auto retOp = cast<func::ReturnOp>(func.front().getTerminator());
    ofs << "  return " << names[retOp.getOperand(0)] << ";\n";
    ofs << "}\n";
  }

  virtual void emitTaps(func::FuncOp func) override {
    header(func);
    auto regsArray = func->getAttrOfType<ArrayAttr>("tapRegs");
    auto combosArray = func->getAttrOfType<ArrayAttr>("tapCombos");
    ofs << "void CircuitImpl::add_taps() {\n";
    ofs << "  tapSet = {\n";
    ofs << "    { // groups\n";
    size_t tapPos = 0;
    for (size_t groupId = 0; groupId < 3; ++groupId) {
      if (groupId != 0) {
        ofs << ",";
      }
      ofs << "    { // group " << groupId << "\n";
      ofs << "{\n";
      bool firstReg = true;
      for (const auto& reg : regsArray.getAsRange<TapRegAttr>()) {
        if (reg.getRegGroupId() != groupId) {
          continue;
        }
        if (firstReg) {
          firstReg = false;
        } else {
          ofs << ",";
        }
        ofs << "{" << reg.getOffset() << "," << reg.getComboId() << "," << tapPos << ", {";
        for (size_t i = 0; i != reg.getBacks().size(); ++i) {
          if (i != 0) {
            ofs << ",";
          }
          ofs << reg.getBacks()[i];
          tapPos++;
        }
        ofs << "}}\n";
      }
      ofs << "}} // group\n";
    }
    ofs << " }, // groups\n";
    ofs << " { // combos\n";
    size_t comboIndex = 0;
    for (const auto& combo : combosArray.getAsRange<ArrayAttr>()) {
      if (comboIndex != 0) {
        ofs << ",";
      }
      ofs << "{" << comboIndex << ", {";
      ofs << llvm::join(
          llvm::map_range(combo.getAsRange<IntegerAttr>(),
                          [&](IntegerAttr val) { return std::to_string(val.getUInt()); }),
          ",");
      ofs << "}}\n";
      comboIndex++;
    }
    ofs << "}, // combos\n";
    ofs << tapPos << "\n";
    ofs << "};\n";
    ofs << "}\n";

    footer(func);
  }

  virtual void emitHeader(func::FuncOp func) override {
    // TODO: This is very lame
    ofs << "// This code is automatically generated\n\n";
    ofs << "#pragma once\n\n";
    ofs << "namespace circuit::" << func.getName() << " {\n\n";
    ofs << "class CircuitImpl : public CircuitBase {\n";
    ofs << "public:\n";
    ofs << "  void add_taps() override;\n";
    ofs << "  MixState poly_edsl(llvm::ArrayRef<Val> u, llvm::ArrayRef<Val> out, "
           "llvm::ArrayRef<Val>  mix) "
           "const override;\n";
    size_t out_size = func.getArgument(1).getType().dyn_cast<BufferType>().getSize();
    size_t mix_size = func.getArgument(3).getType().dyn_cast<BufferType>().getSize();
    ofs << "  size_t out_size() const override { return " << out_size << "; }\n";
    ofs << "  size_t mix_size() const override { return " << mix_size << "; }\n";
    ofs << "};\n\n";
    ofs << "}  // namespace circuit::" << func.getName() << "\n";
  }
};

} // namespace

std::unique_ptr<CppStreamEmitter> createCppStreamEmitter(llvm::raw_ostream& ofs) {
  return std::make_unique<CppStreamEmitterImpl>(ofs);
}

} // namespace zirgen

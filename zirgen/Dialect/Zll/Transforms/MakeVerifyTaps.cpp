// Copyright (c) 2023 RISC Zero, Inc.
//
// All rights reserved.

#include <cassert>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/Transforms/PassDetail.h"

using namespace mlir;

namespace zirgen::Zll {

namespace {

// TODO: Figure out where these should be and move them there.
constexpr uint32_t kInvRate = 4;
constexpr uint32_t kExtSize = 4;
constexpr uint32_t kCheckSize = kInvRate * kExtSize;

std::vector<Value> bufferToVals(OpBuilder& builder, Location loc, Value buf, size_t nvals) {
  std::vector<Value> vals;
  for (size_t i = 0; i != nvals; ++i) {
    Value val = builder.create<GetOp>(loc, buf, i, 0, IntegerAttr());
    vals.push_back(val);
  }

  return vals;
}

struct MakeVerifyTapsPass : public MakeVerifyTapsBase<MakeVerifyTapsPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    assert(mod->hasTrait<OpTrait::SymbolTable>());

    func::FuncOp func;
    mod.walk([&](func::FuncOp op) {
      if (func) {
        // TODO: Support more than one func at a time.  This pass has
        // to target ModuleOp instead of FuncOp because otherwise the
        // pass manager holds a reference to the Op we want to delete
        // and replace.
        mod->emitError() << "We should only have one func op to make taps out of\n";
        return;
      }
      func = op;
    });

    if (!func) {
      mod->emitError() << "No func ops found to make taps for\n";
      return;
    }

    auto locFileId = StringAttr::get(&getContext(), __builtin_FILE());
#define LOC FileLineColLoc::get(locFileId, __builtin_LINE(), 0)

    auto regs = func->getAttrOfType<ArrayAttr>("tapRegs");
    auto combos = func->getAttrOfType<ArrayAttr>("tapCombos");
    auto valTypeAttr = func->getAttrOfType<TypeAttr>("tapType");
    if (!regs || !combos || !valTypeAttr) {
      mod->emitError() << "Can't find tapRegs, tapCombos, and tapType attributes\n";
      return;
    }
    auto valType = cast<ValType>(valTypeAttr.getValue());

    size_t totComboBacks = 0;
    std::vector<size_t> comboBegin;
    for (auto combo : combos) {
      comboBegin.push_back(totComboBacks);
      totComboBacks += cast<ArrayAttr>(combo).size();
    }
    comboBegin.push_back(totComboBacks);

    std::vector<size_t> regGroupSizes;
    regGroupSizes.resize(3, 0);
    for (auto reg : regs.getAsRange<TapRegAttr>()) {
      ++regGroupSizes.at(reg.getRegGroupId());
    }

    size_t accumRowSize = regGroupSizes[0];
    size_t codeRowSize = regGroupSizes[1];
    size_t dataRowSize = regGroupSizes[2];

    std::vector<Type> inTypes = {{
        BufferType::get(&getContext(), valType, 4, BufferKind::Constant),                 // misc
        BufferType::get(&getContext(), valType, totComboBacks + 1, BufferKind::Constant), // comboU
        BufferType::get(&getContext(), valType, kCheckSize, BufferKind::Constant),        // check
        BufferType::get(&getContext(), valType, accumRowSize, BufferKind::Constant),      // accum
        BufferType::get(&getContext(), valType, codeRowSize, BufferKind::Constant),       // code
        BufferType::get(&getContext(), valType, dataRowSize, BufferKind::Constant),       // data
        BufferType::get(&getContext(), valType, 1, BufferKind::Mutable),                  // out
    }};

    OpBuilder builder(&getContext());
    builder.setInsertionPointAfter(func);
    auto newFunc = builder.create<func::FuncOp>(
        LOC, "verify_taps_" + func.getName().str(), builder.getFunctionType(inTypes, {}));
    Block* newBlock = newFunc.addEntryBlock();
    builder.setInsertionPointToStart(newBlock);

    // TODO: Once we have separate subfield and extended field Val
    // types, some of these (check_row, back_one, x) can be in the
    // subfield instead of the extended field.

    // Miscellaneous single inputs
    auto misc = bufferToVals(builder, LOC, newBlock->getArgument(0), 4);
    assert(misc.size() == 4);
    auto mix = misc[0];
    auto backOne = misc[1];
    auto x = misc[2];
    auto z = misc[3];

    // Flattened comboU plus a checkU
    auto comboU = bufferToVals(builder, LOC, newBlock->getArgument(1), totComboBacks + 1);

    auto checkRow = bufferToVals(builder, LOC, newBlock->getArgument(2), kCheckSize);

    // Rows from the merkle verifier
    auto accumRow = bufferToVals(builder, LOC, newBlock->getArgument(3), accumRowSize);
    auto codeRow = bufferToVals(builder, LOC, newBlock->getArgument(4), codeRowSize);
    auto dataRow = bufferToVals(builder, LOC, newBlock->getArgument(5), dataRowSize);
    auto out = newBlock->getArgument(6);
    std::vector<std::vector<Value>> rows = {accumRow, codeRow, dataRow};
    Type ty = ValType::get(builder.getContext(), kFieldPrimeDefault, 1);
    Value zero = builder.create<ConstOp>(LOC, ty, 0);
    Value one = builder.create<ConstOp>(LOC, ty, 1);

    // Allocate totals, one for each back for each combo, plus one at
    // the end for the check.  These correspond to elements of comboU.
    std::vector<Value> tot;
    tot.resize(combos.size() + 1, zero);

    // Precalculate all powers of mix we'll need.
    Value curMix = one;
    for (auto reg : regs.getAsRange<TapRegAttr>()) {
      tot.at(reg.getComboId()) = builder.create<AddOp>(
          LOC,
          tot.at(reg.getComboId()),
          builder.create<MulOp>(LOC, curMix, rows.at(reg.getRegGroupId()).at(reg.getOffset())));
      curMix = builder.create<MulOp>(LOC, curMix, mix);
    }
    for (auto check : checkRow) {
      tot.at(combos.size()) = builder.create<AddOp>(
          LOC, tot.at(combos.size()), builder.create<MulOp>(LOC, curMix, check));
      curMix = builder.create<MulOp>(LOC, curMix, mix);
    }

    Value ret = zero;
    for (size_t i = 0; i != combos.size(); ++i) {
      Value num = tot.at(i);
      size_t exponent = 0;
      for (size_t j = comboBegin.at(i); j != comboBegin.at(i + 1); ++j) {
        num = builder.create<SubOp>(
            LOC,
            num,
            builder.create<MulOp>(LOC, comboU.at(j), builder.create<PowOp>(LOC, x, exponent)));
        ++exponent;
      }
      Value divisor = one;
      for (auto backAttr : cast<ArrayAttr>(combos[i])) {
        uint32_t back = cast<IntegerAttr>(backAttr).getUInt();
        divisor = builder.create<MulOp>(
            LOC,
            divisor,
            builder.create<SubOp>(
                LOC, x, builder.create<MulOp>(LOC, z, builder.create<PowOp>(LOC, backOne, back))));
      }
      ret = builder.create<AddOp>(
          LOC, ret, builder.create<MulOp>(LOC, num, builder.create<InvOp>(LOC, divisor)));
    }

    Value checkNum = builder.create<SubOp>(LOC, tot.at(combos.size()), comboU.at(totComboBacks));
    Value checkDiv = builder.create<SubOp>(LOC, x, builder.create<PowOp>(LOC, z, kInvRate));
    ret = builder.create<AddOp>(
        LOC, ret, builder.create<MulOp>(LOC, checkNum, builder.create<InvOp>(LOC, checkDiv)));

    builder.create<SetOp>(LOC, out, 0, ret);
    builder.create<func::ReturnOp>(LOC);

    // Now we've got the taps evaluation function done, drop the
    // function we're evaluating taps for.  This requires the outer
    // loop to iterate over Moudle instead of FuncOp.
    func->erase();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createMakeVerifyTapsPass() {
  return std::make_unique<MakeVerifyTapsPass>();
}

} // namespace zirgen::Zll

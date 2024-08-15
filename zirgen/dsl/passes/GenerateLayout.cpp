// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ZHLT/IR/TypeUtils.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/dsl/Analysis/LayoutDAGAnalysis.h"
#include "zirgen/dsl/passes/PassDetail.h"

using namespace mlir;
using namespace zirgen::Zhlt;
using namespace zirgen::ZStruct;

namespace zirgen {
namespace dsl {
namespace {

struct LayoutGenerator {
  LayoutGenerator(StringAttr bufferName, DataFlowSolver& solver)
      : bufferName(bufferName), solver(solver) {}

  Attribute generate(ComponentOp component) {
    // Empty layout -> empty attribute
    if (!component.getLayout())
      return Attribute();

    Memo memo;
    auto layout = solver.lookupState<LayoutDAGAnalysis::Element>(component.getLayout());
    return materialize(layout->getValue().get(), memo);
  }

private:
  // A memo of previously generated abstract layouts
  using Memo = DenseMap<LayoutDAG*, Attribute>;

  // Materialize a concrete layout attribute from an abstract layout
  Attribute materialize(const std::shared_ptr<LayoutDAG>& abstract, Memo& memo) {
    // If a layout is already in the memo, increase refCounter past the end so
    // that we don't allocate other things on top of it
    if (memo.contains(abstract.get())) {
      Attribute attr = memo.at(abstract.get());
      refCounter = std::max(refCounter, getNextColumnIndex(attr));
      return attr;
    }

    Attribute attr;
    if (const auto* reg = std::get_if<AbstractRegister>(abstract.get())) {
      // Allocate multiple columns for extension field elements
      attr = RefAttr::get(reg->type.getContext(), refCounter, reg->type);
      refCounter += reg->type.getElement().getFieldK();
    } else if (const auto* arr = std::get_if<AbstractArray>(abstract.get())) {
      SmallVector<Attribute, 4> elements;
      for (auto element : arr->elements) {
        elements.push_back(materialize(element, memo));
      }
      attr = ArrayAttr::get(arr->type.getContext(), elements);
    } else if (const auto* str = std::get_if<AbstractStructure>(abstract.get())) {
      SmallVector<NamedAttribute> fields;
      if (str->type.getKind() == LayoutKind::Mux) {
        unsigned initialRefCounter = refCounter;
        unsigned finalRefCounter = refCounter;
        for (auto field : str->fields) {
          refCounter = initialRefCounter;
          fields.emplace_back(field.first, materialize(field.second, memo));
          finalRefCounter = std::max(finalRefCounter, refCounter);
        }
        refCounter = finalRefCounter;
      } else {
        for (auto field : str->fields) {
          fields.emplace_back(field.first, materialize(field.second, memo));
        }
      }
      auto members = DictionaryAttr::get(str->type.getContext(), fields);
      attr = StructAttr::get(str->type.getContext(), members, str->type);
    } else if (const auto* ref = std::get_if<std::shared_ptr<LayoutDAG>>(abstract.get())) {
      attr = materialize(*ref, memo);
    } else {
      llvm_unreachable("bad variant");
    }
    memo[abstract.get()] = attr;
    return attr;
  }

  unsigned getNextColumnIndex(Attribute attr) {
    unsigned nextColumnIndex = 0;
    attr.walk([&](RefAttr ref) {
      unsigned refNext = ref.getIndex() + ref.getType().getElement().getFieldK();
      nextColumnIndex = std::max(nextColumnIndex, refNext);
    });
    return nextColumnIndex;
  }

  // Name of buffer to allocate registers in
  StringAttr bufferName;

  // The solver used to query data flow analysis results
  DataFlowSolver& solver;

  // The offset of the next register to assign
  unsigned refCounter = 0;
};

struct GenerateLayoutPass : public GenerateLayoutBase<GenerateLayoutPass> {
  void runOnOperation() override {
    auto module = getOperation();
    auto ctx = module.getContext();
    OpBuilder builder(ctx);
    Location loc = builder.getUnknownLoc();

    DataFlowSolver solver;
    solver.load<LayoutDAGAnalysis>();
    if (failed(solver.initializeAndRun(module)))
      return signalPassFailure();

    module.walk([&](ComponentOp component) {
      builder.setInsertionPointToStart(module.getBody());
      StringAttr bufferName = getBufferName(component);
      if (!bufferName)
        return;
      LayoutGenerator layout(bufferName, solver);
      Type layoutType = component.getLayoutType();
      Attribute layoutAttr = layout.generate(component);

      // Only generate layout symbol for components which contain registers.
      if (layoutAttr) {
        StringAttr symName = builder.getStringAttr(getLayoutConstName(component.getName()));
        builder.create<GlobalConstOp>(loc, symName, layoutType, layoutAttr);
      }

      if (component.getName() == "@global") {
        assert(layoutAttr && "Unable to generate layout for globals");
      }
    });
  }

private:
  StringAttr getBufferName(ComponentOp component) {
    // TODO: Is there a better way to keep track of which components
    // get layouts and from which buffers?  Ideally these wouldn't
    // be hardcoded here.
    MLIRContext* ctx = getOperation().getContext();
    if (component.getName() == "@global")
      return StringAttr::get(ctx, "global");
    else if (component.getName().ends_with("$accum"))
      return StringAttr::get(ctx, "accum");
    else if (component.getName() == "@mix")
      return StringAttr::get(ctx, "mix");
    else if (component.getName().starts_with("test$"))
      return StringAttr::get(ctx, "test");
    else if (component.getName() == "Top")
      return StringAttr::get(ctx, "data");
    else
      return {};
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGenerateLayoutPass() {
  return std::make_unique<GenerateLayoutPass>();
}

} // namespace dsl
} // namespace zirgen

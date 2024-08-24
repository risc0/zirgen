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

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ZHLT/IR/TypeUtils.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/dsl/Analysis/LayoutDAGAnalysis.h"
#include "zirgen/dsl/passes/PassDetail.h"

using namespace mlir;
using namespace zirgen::Zhlt;
using namespace zirgen::ZStruct;

/*
 * Fundamentally, a layout is a mapping between the registers of a component and
 * the columns of the underlying STARK trace. The goal of GenerateLayoutPass is
 * to create such mapping that respects all of the layout constraints implied by
 * the semantics of Zirgen, while using as few total columns as possible. This
 * is guided by a few simple rules:
 *   1. LayoutAliasOp operands must be assigned to the same columns
 *   2. A column may be reused by registers on different arms of the same mux
 *   3. Registers otherwise must be assigned different columns
 *
 * We use LayoutDAGAnalysis to take care of the first rule: this data flow
 * analysis constructs a DAG with a vertex for each layout in the program,
 * merging layouts that alias, and edges representing "structural inclusion."
 * Generating the layout becomes a walk over this DAG, using a memo to ensure
 * that if a vertex is visited twice (i.e. aliased) that it is assigned to the
 * same columns.
 *
 * The second rule is handled by the AllocationTable, which keeps track of which
 * columns are allocated in the "current scope." We push a new scope when
 * visiting the arms of a mux, such that we can "pop" it and reuse those columns
 * on the next arm. Afterwards, we mark any columns used by any mux arm as used,
 * pursuant to the third rule. Thus, a mux typically needs as many columns as
 * its largest arm (keep reading).
 *
 * There is an extra complication with layout aliases around muxes: when layouts
 * are aliased between mux arms, those layouts must be placed in the exact same
 * columns, regardless of "when" they are visited relative to other layouts in
 * their respective mux arms. For this reason, it is necessary to reserve those
 * columns across the arms of the mux where they are shared -- which is referred
 * to here as "pinning."
 *
 * Note: Currently, only argument components and mux supers are aliased, so we
 * manually pin them rather than using the LayoutDAGAnalysis to figure this out,
 * and we make the simplifying but suboptimal decision to pin them all the way
 * up to the root layout since it seems to work relatively well with our own
 * circuits. This should be generalized when supporting manual layout aliasing.
 */

namespace zirgen {
namespace dsl {
namespace {

class AllocationTable {
public:
  AllocationTable() : parent(nullptr), storage(256, /*set=*/false) {}
  AllocationTable(AllocationTable* parent) : parent(parent), storage(parent->storage) {}

  // Return the index of the first k consecutive unallocated columns, and mark
  // them as allocated. If pinned, also mark them as allocated in the parent.
  size_t allocate(size_t k, bool pinned) {
    int n = 0;
    while (!canAllocateContiguously(n, k)) {
      n = nextIndex(n);
    }
    storage.set(n, n + k);
    AllocationTable* ancestor = parent;
    while (pinned && ancestor) {
      ancestor->storage.set(n, n + k);
      ancestor = ancestor->parent;
    }

    return n;
  }

  // If a column is allocated in either this or other, mark it as allocated
  AllocationTable& operator|=(const AllocationTable& other) {
    storage |= other.storage;
    return *this;
  }

private:
  // True iff k columns starting at n are all unallocated
  bool canAllocateContiguously(int n, size_t k) {
    // BitVector::find_first_in returns the index of the first set bit in a
    // range, or -1 if they're all unset. If they're all unset, all k of them
    // are unallocated.
    return storage.find_first_in(n, n + k, /*set=*/true) == -1;
  }

  // Return the index of the next unallocated column, resizing storage if necessary
  int nextIndex(int n) {
    int next = storage.find_next_unset(n);
    size_t capacity = storage.getBitCapacity();
    if (next == -1 || next >= capacity) {
      storage.resize(2 * capacity);

      AllocationTable* ancestor = parent;
      while (ancestor) {
        ancestor->storage.resize(2 * capacity);
        ancestor = ancestor->parent;
      }
      next = storage.find_next_unset(n);
    }
    return next;
  }

  AllocationTable* parent;
  BitVector storage;
};

struct LayoutGenerator {
  LayoutGenerator(StringAttr bufferName, DataFlowSolver& solver)
      : bufferName(bufferName), solver(solver) {}

  Attribute generate(ComponentOp component) {
    // Empty layout -> empty attribute
    if (!component.getLayout())
      return Attribute();

    Memo memo;
    AllocationTable allocator;
    auto layout = solver.lookupState<LayoutDAGAnalysis::Element>(component.getLayout());
    return materialize(layout->getValue().get(), memo, allocator);
  }

private:
  // A memo of previously generated abstract layouts
  using Memo = DenseMap<LayoutDAG*, Attribute>;

  // Materialize a concrete layout attribute from an abstract layout
  Attribute materialize(const std::shared_ptr<LayoutDAG>& abstract,
                        Memo& memo,
                        AllocationTable& allocator,
                        bool pinned = false) {
    if (memo.contains(abstract.get())) {
      return memo.at(abstract.get());
    }

    Attribute attr;
    if (const auto* reg = std::get_if<AbstractRegister>(abstract.get())) {
      // Allocate multiple columns for extension field elements
      size_t size = reg->type.getElement().getFieldK();
      size_t index = allocator.allocate(size, pinned);
      attr = RefAttr::get(reg->type.getContext(), index, reg->type);
    } else if (const auto* arr = std::get_if<AbstractArray>(abstract.get())) {
      SmallVector<Attribute, 4> elements;
      for (auto element : arr->elements) {
        elements.push_back(materialize(element, memo, allocator, pinned));
      }
      attr = ArrayAttr::get(arr->type.getContext(), elements);
    } else if (const auto* str = std::get_if<AbstractStructure>(abstract.get())) {
      SmallVector<NamedAttribute> fields;
      if (str->type.getKind() == LayoutKind::Mux) {
        AllocationTable finalAllocator = allocator;
        for (auto field : str->fields) {
          AllocationTable armAllocator(&allocator);
          bool armPinned = pinned || field.first == "@super";
          fields.emplace_back(field.first,
                              materialize(field.second, memo, armAllocator, armPinned));
          finalAllocator |= armAllocator;
        }
        allocator = finalAllocator;
      } else {
        bool strPinned = pinned || (str->type.getKind() == LayoutKind::Argument);
        for (auto field : str->fields) {
          fields.emplace_back(field.first, materialize(field.second, memo, allocator, strPinned));
        }
      }
      auto members = DictionaryAttr::get(str->type.getContext(), fields);
      attr = StructAttr::get(str->type.getContext(), members, str->type);
    } else if (const auto* ref = std::get_if<std::shared_ptr<LayoutDAG>>(abstract.get())) {
      attr = materialize(*ref, memo, allocator, pinned);
    } else {
      llvm_unreachable("bad variant");
    }
    memo[abstract.get()] = attr;
    return attr;
  }

  // Name of buffer to allocate registers in
  StringAttr bufferName;

  // The solver used to query data flow analysis results
  DataFlowSolver& solver;
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

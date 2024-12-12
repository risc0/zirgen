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

#include <variant>

#include "mlir/Analysis/DataFlowFramework.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace zirgen {
namespace dsl {

struct Uninitialized {
  bool operator==(const Uninitialized&) const { return true; }
  friend raw_ostream& operator<<(raw_ostream& os, Uninitialized) { return os << "uninitialized"; }
};

struct Overdefined {
  bool operator==(const Overdefined&) const { return true; }
  friend raw_ostream& operator<<(raw_ostream& os, Overdefined) { return os << "overdefined"; }
};

template <typename T, typename Self> struct LatticeValue {
  LatticeValue() : value(Uninitialized{}) {}
  LatticeValue(Overdefined) : value(Overdefined{}) {}
  LatticeValue(T value) : value(value) {}

  bool isUninitialized() const { return std::holds_alternative<Uninitialized>(value); }
  bool isOverdefined() const { return std::holds_alternative<Overdefined>(value); }
  bool isDefined() const { return std::holds_alternative<T>(value); }

  const T& get() const {
    assert(isDefined());
    return std::get<T>(value);
  }

  bool operator==(const LatticeValue& other) const { return value == other.value; }
  bool operator!=(const LatticeValue& other) const { return !(*this == other); }

  static Self join(const Self& lhs, const Self& rhs) {
    if (lhs.isUninitialized()) {
      return rhs;
    } else if (rhs.isUninitialized()) {
      return lhs;
    } else if (lhs == rhs) {
      return lhs;
    } else {
      return Overdefined{};
    }
  }

  ChangeResult reset() {
    if (isUninitialized())
      return ChangeResult::NoChange;
    value = Uninitialized{};
    return ChangeResult::Change;
  }

  void print(raw_ostream& os) const {
    if (const auto* ui = std::get_if<Uninitialized>(&value)) {
      os << *ui;
    } else if (const auto* od = std::get_if<Overdefined>(&value)) {
      os << *od;
    } else if (isDefined()) {
      static_cast<const Self&>(*this).printValue(os);
    } else {
      llvm_unreachable("bad variant");
    }
  };

  friend raw_ostream& operator<<(raw_ostream& os, const LatticeValue& lattice) {
    lattice.print(os);
    return os;
  }

protected:
  template <typename U, typename... Args>
  using has_print = decltype(std::declval<raw_ostream&>() << std::declval<U>());
  template <typename U> using value_has_print = llvm::is_detected<has_print, U>;

  // Provide an overridable print method for when we can't provide an operator<<
  // overload for T (e.g. types from libraries)
  template <typename U = T, typename = std::enable_if_t<value_has_print<U>::value>>
  raw_ostream& printValue(raw_ostream& os) const {
    return os << get();
  }

  std::variant<Uninitialized, Overdefined, T> value;
};

// A generalization of mlir::dataflow::Lattice which can be attached to any kind
// of program point, not just mlir::Value.
template <typename ValueT> class PointLattice : public AnalysisState {
public:
  using AnalysisState::AnalysisState;

  // TODO: might need to override AnalysisState::onUpdate to (re)visit users of
  // values if the program point is a mlir::Value.
  // See AbstractSparseLattice::onUpdate

  /// Return the value held by this lattice. This requires that the value is
  /// initialized.
  ValueT& getValue() { return value; }
  const ValueT& getValue() const { return const_cast<PointLattice<ValueT>*>(this)->getValue(); }

  using LatticeT = PointLattice<ValueT>;

  /// Join the information contained in the 'rhs' lattice into this
  /// lattice. Returns if the state of the current lattice changed.
  ChangeResult join(const AnalysisState& rhs) {
    return join(static_cast<const LatticeT&>(rhs).getValue());
  }

  /// Meet (intersect) the information contained in the 'rhs' lattice with
  /// this lattice. Returns if the state of the current lattice changed.
  ChangeResult meet(const AnalysisState& rhs) {
    return meet(static_cast<const LatticeT&>(rhs).getValue());
  }

  /// Join the information contained in the 'rhs' value into this
  /// lattice. Returns if the state of the current lattice changed.
  ChangeResult join(const ValueT& rhs) {
    // Otherwise, join rhs with the current optimistic value.
    ValueT newValue = ValueT::join(value, rhs);
    assert(ValueT::join(newValue, value) == newValue && "expected `join` to be monotonic");
    assert(ValueT::join(newValue, rhs) == newValue && "expected `join` to be monotonic");

    // Update the current optimistic value if something changed.
    if (newValue == value)
      return ChangeResult::NoChange;

    value = newValue;
    return ChangeResult::Change;
  }

  /// Trait to check if `T` provides a `meet` method. Needed since for forward
  /// analysis, lattices will only have a `join`, no `meet`, but we want to use
  /// the same `Lattice` class for both directions.
  template <typename T, typename... Args> using has_meet = decltype(std::declval<T>().meet());
  template <typename T> using lattice_has_meet = llvm::is_detected<has_meet, T>;

  /// Meet (intersect) the information contained in the 'rhs' value with this
  /// lattice. Returns if the state of the current lattice changed.  If the
  /// lattice elements don't have a `meet` method, this is a no-op (see below.)
  template <typename VT, std::enable_if_t<lattice_has_meet<VT>::value>>
  ChangeResult meet(const VT& rhs) {
    ValueT newValue = ValueT::meet(value, rhs);
    assert(ValueT::meet(newValue, value) == newValue && "expected `meet` to be monotonic");
    assert(ValueT::meet(newValue, rhs) == newValue && "expected `meet` to be monotonic");

    // Update the current optimistic value if something changed.
    if (newValue == value)
      return ChangeResult::NoChange;

    value = newValue;
    return ChangeResult::Change;
  }

  template <typename VT> ChangeResult meet(const VT& rhs) { return ChangeResult::NoChange; }

  /// Print the lattice element.
  void print(raw_ostream& os) const override { value.print(os); }

private:
  /// The currently computed value that is optimistically assumed to be true.
  ValueT value;
};

// An abstract base class for data flow analyses that work by traversing the
// operations in the IR. In particular, this is a good choice for analyses where
// the lattice for a value should be updated by (re)visiting the op that defines
// it.
class OperationDataflowAnalysis : public DataFlowAnalysis {
public:
  OperationDataflowAnalysis(DataFlowSolver& solver) : DataFlowAnalysis(solver) {}

  virtual LogicalResult initialize(Operation* top) override {
    WalkResult result = top->walk<WalkOrder::PreOrder>([&](Operation* op) {
      if (failed(visitOperation(op)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return success(!result.wasInterrupted());
  }

  // Delegate to calling visitOperation on the appropriate operation
  LogicalResult visit(ProgramPoint* point) override {
    if (auto* op = point->getOperation())
      return visitOperation(op);
    else if (auto* block = point->getBlock())
      return visitOperation(block->getParentOp());
    else
      return failure();
  }

private:
  virtual LogicalResult visitOperation(Operation* op) = 0;
};

} // namespace dsl
} // namespace zirgen

// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/Dialect/Zll/Analysis/DegreeAnalysis.h"

namespace zirgen::Zll {

void DegreeAnalysis::visitOp(ConstOp op) {
  auto lattice = getOrCreate<Element>(op.getOut());
  propagateIfChanged(lattice, lattice->join(0));
}

void DegreeAnalysis::visitOp(GetOp op) {
  auto lattice = getOrCreate<Element>(op.getOut());
  propagateIfChanged(lattice, lattice->join(1));
}

void DegreeAnalysis::visitOp(GetGlobalOp op) {
  // Globals don't contribute to degree
  auto lattice = getOrCreate<Element>(op.getOut());
  propagateIfChanged(lattice, lattice->join(0));
}

void DegreeAnalysis::visitOp(AddOp op) {
  Degree lhs = getOrCreateFor<Element>(op.getOut(), op.getLhs())->getValue();
  Degree rhs = getOrCreateFor<Element>(op.getOut(), op.getRhs())->getValue();
  if (lhs.isDefined() && rhs.isDefined()) {
    auto lattice = getOrCreate<Element>(op.getOut());
    propagateIfChanged(lattice, lattice->join(std::max(lhs.get(), rhs.get())));
    lattice->addContribution(op.getLhs());
    lattice->addContribution(op.getRhs());
  }
}

void DegreeAnalysis::visitOp(SubOp op) {
  Degree lhs = getOrCreateFor<Element>(op.getOut(), op.getLhs())->getValue();
  Degree rhs = getOrCreateFor<Element>(op.getOut(), op.getRhs())->getValue();
  if (lhs.isDefined() && rhs.isDefined()) {
    auto lattice = getOrCreate<Element>(op.getOut());
    propagateIfChanged(lattice, lattice->join(std::max(lhs.get(), rhs.get())));
    lattice->addContribution(op.getLhs());
    lattice->addContribution(op.getRhs());
  }
}

void DegreeAnalysis::visitOp(NegOp op) {
  Degree in = getOrCreateFor<Element>(op.getOut(), op.getIn())->getValue();
  if (in.isDefined()) {
    auto lattice = getOrCreate<Element>(op.getOut());
    propagateIfChanged(lattice, lattice->join(in.get()));
    lattice->addContribution(op.getIn());
  }
}

void DegreeAnalysis::visitOp(MulOp op) {
  Degree lhs = getOrCreateFor<Element>(op.getOut(), op.getLhs())->getValue();
  Degree rhs = getOrCreateFor<Element>(op.getOut(), op.getRhs())->getValue();
  if (lhs.isDefined() && rhs.isDefined()) {
    auto lattice = getOrCreate<Element>(op.getOut());
    propagateIfChanged(lattice, lattice->join(lhs.get() + rhs.get()));
    lattice->addContribution(op.getLhs());
    lattice->addContribution(op.getRhs());
  }
}

void DegreeAnalysis::visitOp(PowOp op) {
  Degree in = getOrCreateFor<Element>(op.getOut(), op.getIn())->getValue();
  unsigned exp = op.getExponent();
  if (in.isDefined()) {
    auto lattice = getOrCreate<Element>(op.getOut());
    propagateIfChanged(lattice, lattice->join(in.get() * exp));
    lattice->addContribution(op.getIn());
  }
}

void DegreeAnalysis::visitOp(EqualZeroOp op) {
  Degree in = getOrCreateFor<Element>(op, op.getIn())->getValue();
  Degree parent = getOrCreateFor<Element>(op, op->getParentOp())->getValue();
  if (in.isDefined() && parent.isDefined()) {
    auto lattice = getOrCreate<Element>(op);
    propagateIfChanged(lattice, lattice->join(parent.get() + in.get()));
    lattice->addContribution(op.getIn());
  }
}

void DegreeAnalysis::visitOp(TrueOp op) {
  auto lattice = getOrCreate<Element>(op.getOut());
  propagateIfChanged(lattice, lattice->join(0));
}

void DegreeAnalysis::visitOp(AndEqzOp op) {
  Degree in = getOrCreateFor<Element>(op.getOut(), op.getIn())->getValue();
  Degree val = getOrCreateFor<Element>(op.getOut(), op.getVal())->getValue();
  if (in.isDefined() && val.isDefined()) {
    auto lattice = getOrCreate<Element>(op.getOut());
    propagateIfChanged(lattice, lattice->join(std::max(in.get(), val.get())));
    lattice->addContribution(op.getIn());
    lattice->addContribution(op.getVal());
  }
}

void DegreeAnalysis::visitOp(AndCondOp op) {
  Value out = op.getOut();
  Degree in = getOrCreateFor<Element>(out, op.getIn())->getValue();
  Degree cond = getOrCreateFor<Element>(out, op.getCond())->getValue();
  Degree inner = getOrCreateFor<Element>(out, op.getInner())->getValue();
  if (in.isDefined() && cond.isDefined() && inner.isDefined()) {
    auto lattice = getOrCreate<Element>(out);
    propagateIfChanged(lattice, lattice->join(std::max(in.get(), cond.get() + inner.get())));
    lattice->addContribution(op.getIn());
    lattice->addContribution(op.getCond());
    lattice->addContribution(op.getInner());
  }
}

void DegreeAnalysis::visitOp(IfOp op) {
  Degree cond = getOrCreateFor<Element>(op, op.getCond())->getValue();
  Degree parent = getOrCreateFor<Element>(op, op->getParentOp())->getValue();
  if (cond.isDefined() && parent.isDefined()) {
    auto lattice = getOrCreate<Element>(op);
    propagateIfChanged(lattice, lattice->join(parent.get() + cond.get()));
    lattice->addContribution(op.getCond());
  }
}

void DegreeAnalysis::visitOp(CallableOpInterface op) {
  auto lattice = getOrCreate<Element>(op);
  propagateIfChanged(lattice, lattice->join(0));
}

void DegreeAnalysis::visitReturnLikeOp(Operation* op) {
  SmallVector<unsigned, 1> operandDegrees;
  for (Value operand : op->getOperands()) {
    auto operandDegree = getOrCreateFor<Element>(op, operand)->getValue();
    if (operandDegree.isDefined())
      operandDegrees.push_back(operandDegree.get());
    else
      return;
  }
  auto lattice = getOrCreate<Element>(op);
  unsigned maxDegree = *std::max_element(operandDegrees.begin(), operandDegrees.end());
  propagateIfChanged(lattice, lattice->join(maxDegree));

  for (Value operand : op->getOperands())
    lattice->addContribution(operand);
}

} // namespace zirgen::Zll

// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/Dialect/ZStruct/Analysis/DegreeAnalysis.h"

namespace zirgen::ZStruct {

void DegreeAnalysis::visitOp(LookupOp op) {
  Zll::Degree base = getOrCreateFor<Element>(op.getOut(), op.getBase())->getValue();
  if (base.isDefined()) {
    auto lattice = getOrCreate<Element>(op.getOut());
    propagateIfChanged(lattice, lattice->join(base.get()));
    lattice->addContribution(op.getBase());
  }
}

void DegreeAnalysis::visitOp(SubscriptOp op) {
  Zll::Degree base = getOrCreateFor<Element>(op.getOut(), op.getBase())->getValue();
  if (base.isDefined()) {
    auto lattice = getOrCreate<Element>(op.getOut());
    propagateIfChanged(lattice, lattice->join(base.get()));
    lattice->addContribution(op.getBase());
  }
}

void DegreeAnalysis::visitOp(LoadOp op) {
  Zll::Degree ref = getOrCreateFor<Element>(op.getOut(), op.getRef())->getValue();
  if (ref.isDefined()) {
    auto lattice = getOrCreate<Element>(op.getOut());
    propagateIfChanged(lattice, lattice->join(ref.get()));
    lattice->addContribution(op.getRef());
  }
}

void DegreeAnalysis::visitOp(BindLayoutOp op) {
  auto lattice = getOrCreate<Element>(op.getOut());
  if (op.getBuffer().getType().getKind() == Zll::BufferKind::Global)
    propagateIfChanged(lattice, lattice->join(0));
  else
    propagateIfChanged(lattice, lattice->join(1));
}

} // namespace zirgen::ZStruct

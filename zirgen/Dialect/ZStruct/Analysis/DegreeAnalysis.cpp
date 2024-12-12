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

#include "zirgen/Dialect/ZStruct/Analysis/DegreeAnalysis.h"

namespace zirgen::ZStruct {

void DegreeAnalysis::visitOp(LookupOp op) {
  Zll::Degree base = getOrCreateFor<Element>(getProgramPointAfter(op), op.getBase())->getValue();
  if (base.isDefined()) {
    auto lattice = getOrCreate<Element>(op.getOut());
    propagateIfChanged(lattice, lattice->join(base.get()));
    lattice->addContribution(op.getBase());
  }
}

void DegreeAnalysis::visitOp(SubscriptOp op) {
  Zll::Degree base = getOrCreateFor<Element>(getProgramPointAfter(op), op.getBase())->getValue();
  if (base.isDefined()) {
    auto lattice = getOrCreate<Element>(op.getOut());
    propagateIfChanged(lattice, lattice->join(base.get()));
    lattice->addContribution(op.getBase());
  }
}

void DegreeAnalysis::visitOp(LoadOp op) {
  Zll::Degree ref = getOrCreateFor<Element>(getProgramPointAfter(op), op.getRef())->getValue();
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

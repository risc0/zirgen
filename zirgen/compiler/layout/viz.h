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

#include "mlir/IR/BuiltinOps.h"

#include <iostream>

namespace zirgen {
namespace layout {
namespace viz {

void typeRelation(mlir::Type, std::ostream&);
void storageNest(mlir::Type, std::ostream&);
void layoutSizes(mlir::Type, std::ostream&);
void layoutAttrs(mlir::ModuleOp, std::ostream&);
void columnKeyPaths(mlir::ModuleOp, size_t, std::ostream&);

} // namespace viz
} // namespace layout
} // namespace zirgen

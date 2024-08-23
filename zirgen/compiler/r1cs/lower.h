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

#include <memory>

#include "zirgen/compiler/r1cs/r1csfile.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace zirgen::R1CS {

std::optional<mlir::ModuleOp> lower(mlir::MLIRContext&, r1csfile::System&);

} // namespace zirgen::R1CS

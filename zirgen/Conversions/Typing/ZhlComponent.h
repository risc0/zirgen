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

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"

namespace zirgen::Typing {

class ComponentManager;

/// An exception type used to signal that the lowering encountered an error in
/// the IR that it cannot recover from.
struct MalformedIRException : public std::exception {};

// Using the supplied builder, creates a zhlt.component as a
// conversion of the given zhl.component. If errors occur they are
// emitted but do not abort the conversion; a stub component is still
// generated.
Zhlt::ComponentOp generateTypedComponent(mlir::OpBuilder& builder,
                                         ComponentManager* typeManager,
                                         Zhl::ComponentOp component,
                                         mlir::StringAttr mangledName,
                                         llvm::ArrayRef<mlir::Attribute> typeArgs);

} // namespace zirgen::Typing

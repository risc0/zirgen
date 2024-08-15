// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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

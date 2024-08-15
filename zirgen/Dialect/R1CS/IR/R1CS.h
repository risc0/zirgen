// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "zirgen/Dialect/R1CS/IR/Types.h.inc"

#include "mlir/IR/Dialect.h"

#include "zirgen/Dialect/R1CS/IR/Dialect.h.inc"

#define GET_OP_CLASSES
#include "zirgen/Dialect/R1CS/IR/Ops.h.inc"

// Copyright (c) 2023 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "zirgen/Dialect/Zll/IR/IR.h"

#define GET_TYPEDEF_CLASSES
#include "zirgen/Dialect/IOP/IR/Types.h.inc"

#include "zirgen/Dialect/IOP/IR/Dialect.h.inc"

#define GET_OP_CLASSES
#include "zirgen/Dialect/IOP/IR/Ops.h.inc"

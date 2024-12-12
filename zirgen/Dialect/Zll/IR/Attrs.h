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

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {

/// An attribute which holds the coefficients of a compile-time constant
/// polynomial in an extension field.
class PolynomialAttr : public DenseI64ArrayAttr {
public:
  using DenseI64ArrayAttr::DenseI64ArrayAttr;

  operator ArrayRef<uint64_t>() const {
    ArrayRef<char> raw = getRawData();
    return ArrayRef<uint64_t>(reinterpret_cast<const uint64_t*>(raw.data()),
                              raw.size() / sizeof(uint64_t));
  }

  ArrayRef<uint64_t> asArrayRef() const { return ArrayRef<uint64_t>{*this}; }

  uint64_t operator[](std::size_t index) const { return asArrayRef()[index]; }

  static PolynomialAttr get(MLIRContext* context, ArrayRef<uint64_t> content) {
    ArrayRef<int64_t> content_cast(reinterpret_cast<const int64_t*>(content.data()),
                                   content.size());
    return mlir::cast<PolynomialAttr>(DenseI64ArrayAttr::get(context, content_cast));
  }
};

ParseResult parsePolynomialAttr(OpAsmParser& parser, PolynomialAttr& coefficientsAttr);
void printPolynomialAttr(OpAsmPrinter& printer, const OpState&, const PolynomialAttr& coefficients);

} // namespace mlir

namespace zirgen::Zll {

// TODO: Move this into our namespace.
using PolynomialAttr = mlir::PolynomialAttr;

class FieldAttr;
mlir::ParseResult parseField(mlir::AsmParser& parser, FieldAttr& coefficientsAttr);
void printField(mlir::AsmPrinter& printer, const FieldAttr& coefficients);

mlir::ParseResult parseFieldExt(mlir::AsmParser& parser, mlir::UnitAttr& extAttr);
void printFieldExt(mlir::AsmPrinter& printer, const mlir::UnitAttr& extAttr);

} // namespace zirgen::Zll

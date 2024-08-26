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

#ifndef ZSTRUCT_DIALECT
#define ZSTRUCT_DIALECT

include "mlir/IR/DialectBase.td"

def ZStructDialect : Dialect {
  let name = "zstruct";
  let cppNamespace = "::zirgen::ZStruct";
  let summary = "The ZStruct Dialect";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::func::FuncDialect",
    "zirgen::Zll::ZllDialect"
  ];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

#endif // ZSTRUCT_DIALECT

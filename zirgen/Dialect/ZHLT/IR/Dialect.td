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

#ifndef ZHLT_DIALECT
#define ZHLT_DIALECT

include "mlir/IR/DialectBase.td"

def ZhltDialect : Dialect {
  let name = "zhlt";
  let cppNamespace = "::zirgen::Zhlt";
  let summary = "Zirgen high-level dialect (typed)";
  let dependentDialects = ["mlir::func::FuncDialect", "zirgen::Zll::ZllDialect"];
  let hasConstantMaterializer = 1;
}

#endif // ZHLT_DIALECT

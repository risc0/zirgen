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

include "mlir/IR/AttrTypeBase.td"
include "zirgen/Dialect/ZHLT/IR/Dialect.td"
include "zirgen/Dialect/Zll/IR/Interfaces.td"
include "zirgen/Dialect/ZStruct/IR/Interfaces.td"

#ifndef ZHLT_TYPES
#define ZHLT_TYPES

class ZHLTType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ZhltDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def ExecContext : ZHLTType<"ExecContext", "exec_context",
    [CodegenTypeInterface, CodegenOnlyPassByReferenceType, BufferContext]> {
  let summary = "Context argument for execution aspects";
}

def ValidityRegsContext : ZHLTType<"ValidityRegsContext", "validity_regs_context",
    [CodegenTypeInterface, CodegenOnlyPassByReferenceType, BufferContext]> {
  let summary = "Context argument for calculating the validity polynomial from registers";
}

def ValidityTapsContext : ZHLTType<"ValidityTapSContext", "validity_taps_context",
    [CodegenTypeInterface, CodegenOnlyPassByReferenceType, BufferContext]> {
  let summary = "Context argument for calculating the validity polynomial from taps";
}


#endif /* ZHLT_TYPES */

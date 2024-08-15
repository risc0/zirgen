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

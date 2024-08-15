include "mlir/IR/AttrTypeBase.td"
include "zirgen/Dialect/R1CS/IR/Dialect.td"

class R1CSAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<R1CSDialect, name, traits> {
  let mnemonic = attrMnemonic;
}


include "mlir/IR/AttrTypeBase.td"
include "zirgen/Dialect/ZHL/IR/Dialect.td"

class ZhlAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<ZhlDialect, name, traits> {
  let mnemonic = attrMnemonic;
}


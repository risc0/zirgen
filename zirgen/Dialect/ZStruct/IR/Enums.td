include "mlir/IR/EnumAttr.td"

def LayoutKind : I32EnumAttr<"LayoutKind", "Layout Kind", [
     I32EnumAttrCase<"Normal", 0, "normal">,
     I32EnumAttrCase<"Mux", 1, "mux">,
     I32EnumAttrCase<"Argument", 2, "argument">,
]> {
  let cppNamespace = "::zirgen::ZStruct";
}


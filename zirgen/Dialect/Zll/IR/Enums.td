include "mlir/IR/EnumAttr.td"

def BufferKind : I32EnumAttr<"BufferKind", "Buffer Type", [
     I32EnumAttrCase<"Constant", 0, "constant">,
     I32EnumAttrCase<"Mutable", 1, "mutable">,
     I32EnumAttrCase<"Global", 2, "global">,
]> {
  let cppNamespace = "::zirgen::Zll";
}

def DigestKind : I32EnumAttr<"DigestKind", "Digest Type", [
     I32EnumAttrCase<"Default", 0, "default">,
     I32EnumAttrCase<"Sha256", 1, "sha256">,
     I32EnumAttrCase<"Poseidon2", 2, "poseidon2">,
]> {
  let cppNamespace = "::zirgen::Zll";
}


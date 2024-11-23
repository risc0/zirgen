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

#ifndef ZIRGEN_IR_ATTRS
#define ZIRGEN_IR_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "zirgen/Dialect/Zll/IR/Dialect.td"

class ZllAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<ZllDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Tap : ZllAttr<"Tap", "tap"> {
  let summary = "A Tap, which specifies a register and a back.";
  let parameters = (ins
      "uint32_t" : $regGroupId,
      "uint32_t" : $offset,
      "uint32_t" : $back);
  let assemblyFormat = "`<` $regGroupId `,` $offset `,` $back `>`";
}

def TapsAttr : ZllAttr<"Taps", "taps"> {
  let summary = "Set of tap registers and back distances associated with a circuit.   Normally associated with the module operation; use Zll::lookupModuleAttr to access.";
  let genVerifyDecl = 1;
  let parameters = (ins
      ArrayRefParameter<"zirgen::Zll::TapAttr">:$taps
  );

  code extraClassDeclaration = [{
    static constexpr llvm::StringLiteral lookupModuleAttrName() { return "zll.taps"; }

    /// Sort the given tap set and pad to fill the given buffers with no holes.
    static TapsAttr sortAndPad(
                        llvm::SmallVector<zirgen::Zll::TapAttr> taps,
                        zirgen::Zll::BuffersAttr);
  }];

  let assemblyFormat = [{ $taps }];
}


def PolynomialAttr : DenseArrayAttrBase<"PolynomialAttr", "uint64_t", "i64"> {
  let summary = "A compile-time constant element of an extension field";
  let constBuilderCall = "mlir::PolynomialAttr::get($_builder.getContext(), $0)";
}

def Field : ZllAttr<"Field", "field"> {
   let summary = "A description of a finite galois field configuration including a field, extension field, and irriducable polynomial.";
   let parameters = (ins
      StringRefParameter<"Identifying name of this field"> : $name,
      AttrParameter<"uint64_t", "Core prime to this field">  : $prime,
      AttrParameter<"size_t", "Number of elements in extension field"> : $extDegree,
      ArrayRefParameter<"uint64_t", "Irreducible polynomial of the field"> : $polynomial);
  let assemblyFormat = [{ $name `,` $prime `,` $extDegree `,` $polynomial }];
  let extraClassDeclaration = [{
      // Returns the field arithmetic implementation for the base
      // field, operating on individual field elements (u64).
      Field getBaseField() const;

      // Returns the field arithmetic implementation for the base
      // field, operating on arrays of 1 field elements (ArrayRef<u64> with size=1).
      ExtensionField getBaseExtensionField() const;

      // Returns the field arithmetic implementation for the extension
      // field, operating on arrays of extSize field elements (ArrayRef<u64>)
      ExtensionField getExtExtensionField() const;
  }];
  let genVerifyDecl = 1;
}

def FieldExt : ZllAttr<"FieldExt", "fieldExt"> {
  let summary = "Marker to signify a field extension as opposed to the base field.";
  let assemblyFormat = [{ `ext` }];
}

def ProtocolInfoAttr : ZllAttr<"ProtocolInfo", "protocol_info"> {
  let summary = "Define the protocol info string for this circuit.  Normally associated with the module operation; use Zll::lookupModuleAttr to access.";
  let parameters = (ins
      StringRefParameter<"Protocol information string"> : $protocolInfo);
  let builders = [
      AttrBuilder<(ins CArg<"zirgen::ProtocolInfo">: $protocolInfo)>
  ];

  let extraClassDeclaration = [{
    static constexpr llvm::StringLiteral lookupModuleAttrName() { return "zll.protocolInfo"; }

    zirgen::ProtocolInfo getValue();

    std::string str() const { return getProtocolInfo().str(); }
  }];
  let genVerifyDecl = 1;
  let assemblyFormat = [{ $protocolInfo }];
}

def BufferDescAttr : ZllAttr<"BufferDesc", "buffer_desc"> {
  let summary = "Information on a circuit buffer that's global to a circuit";
  let parameters = (ins
      "mlir::StringAttr": $name,
      "zirgen::Zll::BufferType": $type,
      OptionalParameter<"std::optional<size_t>", "Tap group">: $regGroupId
      );
  let assemblyFormat = [{ `(` $name `,` $type ( `,` $regGroupId^ )? `)` }];
  let extraClassDeclaration = [{
      BufferKind getKind() const;
      bool isGlobal() const { return getKind() == BufferKind::Global; }
      size_t getRegCount() const;
  }];
  let extraClassDefinition = [{
    BufferKind $cppClass::getKind() const { return getType().getKind(); }
    size_t $cppClass::getRegCount() const { return getType().getSize(); }
  }];
}

def BuffersAttr : ZllAttr<"Buffers", "buffers"> {
  let summary = "Set of buffers associated with a circuit.  Normally associated with the module operation; use Zll::lookupModuleAttr to access.";
  let parameters = (ins
      ArrayRefParameter<"BufferDescAttr", "">:$buffers
  );
  code extraClassDeclaration = [{
    static constexpr llvm::StringLiteral lookupModuleAttrName() { return "zll.buffers"; }

    // Returns the tap buffer definitions in order
    llvm::SmallVector<BufferDescAttr> getTapBuffers() const;

    // Returns the given named buffer, if present.
    BufferDescAttr getBuffer(llvm::StringRef bufferName) const;
  }];
  let genVerifyDecl = 1;
  let assemblyFormat = [{ $buffers }];
}

def StepsAttr : ZllAttr<"Steps", "steps"> {
  let summary = "List of steps associated with a circuit.  Normally associated with the module operation; use Zll::lookupModuleAttr to access.";
  let parameters = (ins
      ArrayRefParameter<"mlir::StringAttr", "">:$steps
  );

  code extraClassDeclaration = [{
    static constexpr llvm::StringLiteral lookupModuleAttrName() { return "zll.steps"; }
  }];

  let assemblyFormat = [{ $steps }];
}

def CircuitNameAttr : ZllAttr<"CircuitName", "circuit_name"> {
  let summary = "Name of the circuit.  Normally associated with the module operation; use Zll::lookupModuleAttr to access.";
  let parameters = (ins
      StringRefParameter<"name">:$name,
      StringRefParameter<"Base C++ namespace for circuits">:$baseCppNamespace
  );
  let skipDefaultBuilders = 1;
  let builders = [
    AttrBuilder<(ins CArg<"llvm::StringRef", "">:$name,
                     CArg<"llvm::StringRef", "\"risc0::circuit\"">:$baseCppNamespace), [{
      return Base::get($_ctxt, name, baseCppNamespace);
    }]>
  ];

  code extraClassDeclaration = [{
    static constexpr llvm::StringLiteral lookupModuleAttrName() { return "zll.circuit_name"; }
    std::string getCppNamespace();
  }];

  let assemblyFormat = [{ $baseCppNamespace `:` $name }];
}

#endif // ZIRGEN_IR_ATTRS

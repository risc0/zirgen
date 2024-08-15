include "mlir/IR/OpBase.td"

#ifndef ZSTRUCT_INTERFACES
#define ZSTRUCT_INTERFACES

class ZStructNativeTypeTrait<string name,
                    code extraTypeDeclaration = [{}],
                    code extraTypeDefinition = [{}]> :
      NativeTypeTrait<name, extraTypeDeclaration, extraTypeDefinition> {
  let cppNamespace = "::zirgen::ZStruct";
}

// An context argument type which can supply buffers when passed to lookupNearestImplicitArg.
def BufferContext : ZStructNativeTypeTrait<"BufferContextTypeTrait">;

#endif /* ZSTRUCT_INTERFACES */

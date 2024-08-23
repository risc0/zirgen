include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

#ifndef ZHLT_INTERFACES
#define ZHLT_INTERFACES

def ComponentOpInterface : OpInterface<"ComponentOpInterface"> {
  let cppNamespace = "::zirgen::Zhlt";
  let description = [{
    Represents a component that can potentially be constructed.
  }];

  let methods = [
    // See zirgen::Zhlt::ComponentManager for documentation on what these methods do.
    InterfaceMethod<"",
      "void", "inferType",
      (ins "ComponentManager*":$manager, "ComponentTypeAttr&":$name, "mlir::ValueRange":$constructArgs),
      "",
      [{
        // By default, we don't attempt to infer anything.
      }]
    >,

    InterfaceMethod<"",
      "mlir::LogicalResult", "requireComponent",
      (ins "ComponentManager*":$manager, "mlir::Location":$loc,
           "ComponentTypeAttr":$name),
      "",
      [{
          // By default, this is a non-generic component, so we can use it when no type arguments are present.
          if (name.getTypeArgs().empty())
            return mlir::success();
          else
            return mlir::emitError(loc) << name << " does not allow specialization";
      }]
    >,
           
    InterfaceMethod<"",
       "mlir::Type", "getLayoutType",
      (ins "ComponentManager*":$manager,"ComponentTypeAttr":$name),
      "",
      [{
          // By default, has no layout
          return {};
      }]
    >,
    InterfaceMethod<"",
      "mlir::Type", "getValueType",
      (ins "ComponentManager*":$manager, "ComponentTypeAttr":$name),
      "",
      [{
          // By default, has empty value type
          return ZStruct::getComponentType(name.getContext());
      }]
    >,

    InterfaceMethod<"",
      "mlir::Value", "buildConstruct",
      (ins "ComponentManager*":$manager, "mlir::OpBuilder&":$builder,
           "mlir::Location":$loc,
           "ComponentTypeAttr":$name,
           "mlir::ValueRange":$constructArgs,
           "mlir::Value":$layout),
      "",
      [{
          // A builtin componment can't be constructed unless it defines how to construct it.
          mlir::emitError(loc) << name << " can not be constructed";
          return {};
      }]
    >,

    InterfaceMethod<"",
      "mlir::Value", "reconstructFromLayout",
      (ins "ComponentManager*":$manager, "mlir::OpBuilder&":$builder,
           "mlir::Location":$loc,
           "ComponentTypeAttr":$name,
           "mlir::Value":$layout,
           "size_t":$distance),
      "",
      [{
          // By default, we can not reconstruct from a layout.
          mlir::emitError(loc) << name << " can not be reconstructed from a layout";
          return {};
      }]
    >,
           
    InterfaceMethod<"",
      "std::optional<llvm::SmallVector<mlir::Type>>", "getConstructParams",
      (ins "ComponentManager*":$manager, "ComponentTypeAttr":$name),
      "",
      [{
          // By default, we can't tell what constructor parameters are present.
          return std::nullopt;
      }]
    >
  ];
}

#endif

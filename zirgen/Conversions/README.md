# zirgen/Conversions

Dialect conversion passes that lower or transform modules between zirgen's MLIR dialects. The primary conversion here is the ZHL → ZHLT typing pass.

## Typing/

`Typing/` implements the type-checking and generic-instantiation conversion from the ZHL dialect to the ZHLT dialect. Entry point: `zirgen::Typing::typeCheck()` in `ComponentManager.h`.

### What happens during typing

1. **Component discovery** — The `ComponentManager` scans the ZHL module and registers all user-defined components alongside the built-in components defined in `BuiltinComponents.h`.

2. **Generic instantiation (monomorphization)** — ZIR components can be parameterized by types. `ComponentManager::getComponent()` resolves each requested `(name, typeArgs)` pair: if the component is generic and not yet instantiated for these type arguments, it generates a new monomorphic `Zhlt::ComponentOp` with a mangled name. A component stack tracks in-progress instantiations to detect and report illegal recursion.

3. **Output** — The result is a ZHL module fully replaced by a ZHLT module in which all components are monomorphic, fields have concrete types, and the validity and step functions are expressed as `CheckFuncOp` and `StepFuncOp`.

### Key files

| File | Purpose |
|------|---------|
| `ComponentManager.h` | Public interface: `typeCheck()` free function and `ComponentManager` class |
| `ComponentManager.cpp` | Instantiation logic, recursion detection, generic builtin generation |
| `BuiltinComponents.h` | Preamble text and C++ definitions for built-in ZIR types (`Reg`, `Val`, etc.) |
| `TypeCheck.cpp` | Orchestration: constructs `ComponentManager`, calls `gen()`, returns ZHLT module |

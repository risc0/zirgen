# Externs

Externs (external functions) provide an escape hatch in Zirgen for calling code that cannot be expressed within the constraint system. They are essential for nondeterministic computations, witness generation, and integration with external systems during proof generation.

## What Are Externs?

In a zero-knowledge proof system, the circuit defines polynomial constraints that must be satisfied. However, some operations needed for witness generation cannot be expressed as polynomial constraints:

- Complex computations (e.g., division, square roots)
- I/O operations (reading preimages, host system calls)
- Nondeterministic choices (finding witnesses)
- External system integration

Externs allow you to declare functions that will be implemented outside the DSL and called during witness generation.

## Syntax

### Declaring Externs

Externs are declared at the module level (outside of components):

```zirgen
extern FunctionName(param1: Type1, param2: Type2): ReturnType;
```

Components:
- `extern` keyword
- Function name (typically PascalCase)
- Parameter list with types
- Optional return type (defaults to `Component` if omitted)
- Semicolon at the end

### Examples of Declarations

```zirgen
// Extern with return value
extern GetPreimage(idx: Val): Val;

// Extern with multiple parameters
extern HostWrite(fd: Val, addr: ValU32, len: Val): Val;

// Extern with no return value
extern MemoryDelta(addr: Val, cycle: Val, dataLow: Val, dataHigh: Val, count: Val);

// Extern with no parameters
extern NextPreimage(): Val;
```

## Using Externs

### Calling External Functions

Once declared, externs are called like regular functions:

```zirgen
component UseExtern() {
  result := GetPreimage(5);
  data := NextPreimage();
}
```

### Important Constraint

**Externs can only be called in nondeterministic contexts.** This means:

1. The return value cannot be directly used in a constraint equation
2. You must wrap extern results in `NondetReg` before using them in constraints

Example:

```zirgen
component ConstrainExtern() {
  // Call extern
  raw_value := GetPreimage(idx);

  // Wrap in NondetReg to use in constraints
  value := NondetReg(raw_value);

  // Now you can use it in constraints
  value * 2 = other_value;
}
```

From the builtin components documentation:
> Vals computed nondeterministically or returned by externs can't be used in constraints without registerizing, so use NondetReg for such values.

## Real-World Examples

### Keccak Hash Preimage

From the Keccak circuit:

```zirgen
extern GetPreimage(idx: Val): Val;
extern NextPreimage(): Val;

component KeccakTop() {
  // Get preimage data from external source
  first_byte := GetPreimage(0);
  next_byte := NextPreimage();

  // Use in circuit logic...
}
```

### RISC-V Host Calls

From the RISC-V circuit:

```zirgen
extern HostReadPrepare(fd: Val, len: Val): Val;
extern HostWrite(fd: Val, addr: ValU32, len: Val): Val;

component EcallHandler(fd: Val, addr: ValU32, len: Val) {
  // Prepare host read operation
  bytes_read := HostReadPrepare(fd, len);

  // Write to host
  bytes_written := HostWrite(fd, addr, len);
}
```

### Memory System Integration

From the memory subsystem:

```zirgen
extern MemoryDelta(addr: Val, cycle: Val, dataLow: Val, dataHigh: Val, count: Val);
extern GetDiffCount(cycle: Val): Val;
extern GetMemoryTxn(addr: Val): MemoryTxnResult;

component MemoryController(addr: Val, cycle: Val) {
  // Record memory change
  MemoryDelta(addr, cycle, data_low, data_high, 1);

  // Query memory state
  diff_count := GetDiffCount(cycle);
  txn_info := GetMemoryTxn(addr);
}
```

### Division Operation

```zirgen
extern Divide(numer: ValU32, denom: ValU32, sign_type: Val): DivideReturn;

component DivisionCircuit(a: ValU32, b: ValU32) {
  // Call extern to compute quotient and remainder
  div_result := Divide(a, b, UNSIGNED);

  // Extract and register results
  quotient := NondetReg(div_result.quotient);
  remainder := NondetReg(div_result.remainder);

  // Constrain: a = b * quotient + remainder
  a = b * quotient + remainder;
  // Constrain: remainder < b
  InRange(0, remainder, b);
}
```

## How Externs Work

### Compilation and Lowering

When Zirgen code is compiled:

1. **Parsing**: Extern declarations are parsed into AST nodes
2. **Lowering to ZHL IR**: Externs become `zhl.extern` operations with an `extern` attribute
3. **Lowering to ZLL IR**: They become `zll.extern` operations
4. **Code Generation**: Calls emit an `invokeExtern` macro invocation

Example lowering to ZHL IR:

```
zhl.component @MyExtern attributes {extern} {
  %0 = zhl.global "XType"
  %1 = zhl.parameter "x"(0) : %0
  %2 = zhl.global "YType"
  %3 = zhl.parameter "y"(1) : %2
  %4 = zhl.global "RetType"
  %5 = zhl.extern "MyExtern"(%1, %3) : %4
  zhl.super %5
}
```

### Runtime Execution

During witness generation:

1. **Interpreter encounters extern call**: When the interpreter reaches an `extern` operation
2. **ExternHandler invoked**: The interpreter calls the registered `ExternHandler`
3. **External code executes**: The handler runs native code to compute the result
4. **Values returned**: Results are returned as field elements and assigned to outputs

From the interpreter code:

```rust
LogicalResult ExternOp::evaluate(Interpreter& interp,
                                 llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                                 EvalAdaptor& adaptor) {
  ExternHandler* handler = interp.getExternHandler();
  if (!handler) {
    return emitError() << "No extern handler set";
  }
  size_t outCount = getNumResults();
  std::optional<std::vector<uint64_t>> outFp =
      handler->doExtern(getName(), getExtra(), adaptor.getIn(), outCount);
  // ... assign outputs
}
```

### ExternHandler Interface

The `ExternHandler` is a C++ interface that your proving system implements:

```cpp
class ExternHandler {
public:
  virtual std::optional<std::vector<uint64_t>>
    doExtern(StringRef name,
             StringRef extra,
             ArrayRef<uint64_t> args,
             size_t outCount) = 0;
};
```

Your implementation maps extern names to functions:

```cpp
std::optional<std::vector<uint64_t>> MyExternHandler::doExtern(
    StringRef name, StringRef extra, ArrayRef<uint64_t> args, size_t outCount) {
  if (name == "GetPreimage") {
    return {getPreimageData(args[0])};
  } else if (name == "Divide") {
    uint64_t quotient = args[0] / args[1];
    uint64_t remainder = args[0] % args[1];
    return {{quotient, remainder}};
  }
  // ...
}
```

## Testing with Externs

Test files can declare and use externs:

```zirgen
extern ReturnsVal(): Val;
extern TakesArg(x: Val);
extern ReturnsPair(): Pair;
extern TakesPair(p: Pair);

component Top() {
  TakesArg(ReturnsVal());
  TakesPair(ReturnsPair());
}

test example {
  Top();
}
```

During testing, you provide a test-specific `ExternHandler` that supplies appropriate values.

## Common Patterns

### Witness Computation

Compute witness values that satisfy constraints:

```zirgen
extern ComputeWitness(input: Val): Val;

component WitnessConstraint(input: Val) {
  // Compute nondeterministically
  witness := NondetReg(ComputeWitness(input));

  // Constrain the result
  witness * witness = input;  // witness is square root of input
}
```

### Lookup Tables

Access precomputed lookup tables:

```zirgen
extern LookupTable(index: Val): Val;

component UseLookup(idx: Val) {
  result := NondetReg(LookupTable(idx));
  // Constrain that result is correct...
}
```

### State Queries

Query external state during witness generation:

```zirgen
extern GetMemory(addr: Val): Val;
extern SetMemory(addr: Val, value: Val);

component MemoryAccess(addr: Val, is_write: Val, write_val: Val) {
  // Read current value
  old_val := NondetReg(GetMemory(addr));

  // Update if writing
  new_val := [is_write, 1-is_write] -> (write_val, old_val);
  SetMemory(addr, new_val);
}
```

### Complex Computations

Offload expensive computations:

```zirgen
extern Sha256(data: Array<Val, 64>): Array<Val, 32>;

component HashCheck(data: Array<Val, 64>, expected_hash: Array<Val, 32>) {
  // Compute hash externally
  computed_hash := for i : 0..32 { NondetReg(Sha256(data)[i]) };

  // Constrain it matches expected
  for i : 0..32 {
    computed_hash[i] = expected_hash[i];
  }
}
```

## Best Practices

### 1. Minimal Extern Surface

Keep externs minimal. Compute as much as possible within the DSL:

```zirgen
// Good: Extern only for nondeterministic part
extern FindRoot(x: Val): Val;
component CheckRoot(x: Val) {
  root := NondetReg(FindRoot(x));
  root * root = x;  // Constraint in DSL
}

// Less ideal: Entire check as extern
extern CheckRootExtern(x: Val): Val;
```

### 2. Always Register Extern Results

Never use extern results directly in constraints:

```zirgen
// Wrong
extern Bad(x: Val): Val;
component BadExample(x: Val) {
  Bad(x) = 5;  // Error! Cannot constrain extern directly
}

// Correct
extern Good(x: Val): Val;
component GoodExample(x: Val) {
  result := NondetReg(Good(x));
  result = 5;  // OK - constraining a register
}
```

### 3. Type Safety

Use custom return types for complex data:

```zirgen
// Define return type
component DivResult {
  quotient: Val;
  remainder: Val;
}

extern SafeDivide(a: Val, b: Val): DivResult;

component UseSafeDivide(a: Val, b: Val) {
  result := SafeDivide(a, b);
  q := NondetReg(result.quotient);
  r := NondetReg(result.remainder);
}
```

### 4. Document Extern Contracts

Comment externs with their expected behavior:

```zirgen
// Returns the preimage byte at the given index.
// Panics if index is out of bounds.
extern GetPreimage(idx: Val): Val;

// Writes len bytes from addr to file descriptor fd.
// Returns number of bytes actually written.
extern HostWrite(fd: Val, addr: ValU32, len: Val): Val;
```

## Key Takeaways

- Externs enable **nondeterministic computations** and **external system integration**
- Declared at **module level** with `extern` keyword
- Can have **zero or more parameters** and an **optional return type**
- Must be called in **nondeterministic contexts** only
- Return values must be **wrapped in NondetReg** before use in constraints
- Implemented via **ExternHandler** interface in native code
- Used for: witness generation, I/O, complex math, system integration
- Keep extern surface **minimal** - do as much as possible in DSL
- Always **register** extern results before constraining them

Externs are the bridge between Zirgen's constraint-based DSL and the imperative world of witness generation, enabling practical zero-knowledge proof systems.

[Prev](99_Backs.md)

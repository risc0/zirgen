# Backs

Backs (short for "back-references") are a mechanism in Zirgen for referencing register values from previous cycles in the execution trace. They enable you to write constraints and compute values based on historical state, which is essential for expressing sequential logic in circuits.

## Understanding Backs

In Zirgen, a circuit execution progresses through cycles (analogous to clock cycles in a processor), with each cycle populating a row in the execution trace. When you need to reference a value from an earlier cycle, you use a back-reference.

## Syntax

The syntax for backs uses the `@` operator followed by the number of cycles to look back:

```zirgen
register@N
```

Where:
- `register` is a register variable
- `N` is the number of cycles to look back (must be a compile-time constant)
- `@0` refers to the current cycle (usually omitted)
- `@1` refers to the previous cycle
- `@2` refers to two cycles ago, etc.

## Basic Example

From the Fibonacci circuit:

```zirgen
component FibonacciStep(cycle: CycleType) {
  d2 : Reg;
  d3 : Reg;

  // d1 gets f0 on first cycle, otherwise gets previous d2
  d1 := Reg([cycle.is_first_cycle, 1-cycle.is_first_cycle] -> (f0, d2@1));

  // d2 gets f1 on first cycle, otherwise gets previous d3
  d2 := Reg([first, 1-first] -> (f1, d3@1));

  // d3 is always d1 + d2 (current cycle values)
  d3 := Reg(d1 + d2);
}
```

In this example:
- `d2@1` refers to the value that was in `d2` during the previous cycle
- `d3@1` refers to the value that was in `d3` during the previous cycle

## How Backs Work Internally

Backs are implemented using modular arithmetic to compute the correct position in the execution trace:

```
actual_cycle = (current_cycle + total_cycles - back) % total_cycles
```

This formula ensures:
- When `back = 0`: you get the current cycle
- When `back = 1`: you get the previous cycle
- When `back >= current_cycle`: the modulo wraps around to reference later cycles (useful for handling the first cycle)

On the first cycle with `back = 1`, the modulo arithmetic wraps to the last cycle of the trace. This means you must be careful about what constraints you apply on the first cycle.

## Use Cases

### State Transitions

Backs are essential for enforcing state transitions:

```zirgen
component Counter() {
  count : Reg;

  // Enforce that count increments by 1 each cycle
  count := Reg([is_first, 1-is_first] -> (0, count@1 + 1));
}
```

### Memory Consistency

Track memory operations across cycles:

```zirgen
component MemoryCell(addr: Val, cycle: Val) {
  value : Reg;
  prev_addr : Reg;

  // If same address as previous cycle, value must match
  same_addr := (addr = prev_addr@1);
  value := Reg([same_addr, 1-same_addr] -> (value@1, new_value));
}
```

### Pipelined Operations

Implement multi-cycle operations by passing intermediate results:

```zirgen
component Pipeline() {
  stage1 : Reg;
  stage2 : Reg;
  stage3 : Reg;

  // Each stage processes the previous stage's output from last cycle
  stage2 := Reg(ProcessStage2(stage1@1));
  stage3 := Reg(ProcessStage3(stage2@1));
  result := ProcessFinal(stage3@1);
}
```

### Accumulation

Build running sums or products:

```zirgen
component Accumulator(input: Val) {
  sum : Reg;

  // Add current input to previous sum
  sum := Reg([is_first, 1-is_first] -> (input, sum@1 + input));
}
```

## Constraints and Limitations

### Forward Declaration Required

You must declare registers before using their back-references:

```zirgen
component Example() {
  // Declare d2 before using d2@1
  d2 : Reg;
  d1 := Reg(d2@1);  // OK

  // This would fail:
  // d3 := Reg(d4@1);  // Error: d4 not yet declared
  // d4 : Reg;
}
```

### Cyclical Wrapping

Back-references wrap around cyclically. On cycle 0, a reference to `@1` points to the last cycle in the trace. Make sure to handle the first cycle specially if needed:

```zirgen
component SafeBack(is_first: Val) {
  prev : Reg;

  // Use a default value on the first cycle
  curr := Reg([is_first, 1-is_first] -> (default_val, prev@1));
}
```

### Cycle-Based Registers Only

Back-references only work with cycle-based registers. You cannot use backs with global registers (registers that exist outside the per-cycle context):

```zirgen
component Example() {
  // Cycle register - backs are OK
  cycle_reg : Reg;
  val := cycle_reg@1;  // OK

  // Global register - backs not allowed
  global global_reg : Reg;
  // val := global_reg@1;  // Error!
}
```

### Compile-Time Constants

The back amount must be a compile-time constant. You cannot use a runtime value:

```zirgen
component Example() {
  reg : Reg;

  // OK - constant back reference
  val := reg@1;

  // Not allowed - dynamic back reference
  // val := reg@some_variable;  // Error!
}
```

## Advanced: Tap Data Structure

Internally, backs are implemented using "taps" - references to specific positions in the execution trace. Each tap consists of:

- **group**: The register group (categorized as data, code, or accum)
- **pos**: The position/offset within the register layout
- **back**: How many cycles back to look

The prover and verifier use these taps to:
1. Collect the referenced values during witness generation
2. Compute polynomial divisor terms during verification
3. Ensure constraints are satisfied across all cycles

## Example: Constraint with Backs

Here's a complete example showing how to use backs in a constraint:

```zirgen
component SequenceChecker(is_first: Val) {
  prev_val : Reg;
  curr_val : Reg;

  // Constraint: current value must be prev + 1 (except on first cycle)
  constraint := [is_first, 1-is_first] -> (
    curr_val,  // First cycle: any value is OK
    curr_val - prev_val@1 - 1  // Other cycles: must be prev + 1
  );
  constraint = 0;
}
```

## Testing with Backs

When writing tests, remember that backs reference previous cycles:

```zirgen
test fibonacci_sequence {
  // Cycle 0: d1=1, d2=1, d3=2
  // Cycle 1: d1=1 (from d2@1), d2=2 (from d3@1), d3=3
  // Cycle 2: d1=2 (from d2@1), d2=3 (from d3@1), d3=5
  // etc.

  fib := FibonacciStep(cycle_info);
  // Test constraints hold across all cycles
}
```

## Common Patterns

### Initialization Pattern

Always handle the first cycle specially:

```zirgen
[is_first_cycle, 1-is_first_cycle] -> (initial_value, value_from_previous)
```

### State Machine Pattern

Use backs to implement state machines:

```zirgen
component StateMachine() {
  state : Reg;

  next_state := [state@1] -> (
    0 -> State1(),
    1 -> State2(),
    2 -> State3()
  );

  state := Reg(next_state);
}
```

### Sliding Window Pattern

Access multiple previous cycles for complex constraints:

```zirgen
component SlidingWindow() {
  val : Reg;

  // Average of current and two previous values
  avg := (val + val@1 + val@2) / 3;
}
```

## Key Takeaways

- Backs enable **sequential logic** by referencing previous cycles
- Syntax is `register@N` where N is cycles back
- Back-references **wrap around** cyclically (modulo total cycles)
- **Always declare** registers before using their backs
- Handle the **first cycle specially** to avoid unintended wrapping
- Backs work only with **cycle-based registers**, not globals
- The back amount must be a **compile-time constant**
- Common pattern: `[is_first, 1-is_first] -> (init, prev@1)`

Backs are fundamental to writing stateful circuits in Zirgen, enabling you to express constraints that relate the current cycle's values to historical state.

[Prev](99_Arrays_and_Loops.md)
[Next](99_Externs.md)

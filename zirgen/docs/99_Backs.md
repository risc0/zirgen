# Backs

A *back-reference* lets a component read the value that a register (or any component field) held in a previous execution cycle. This is the primary mechanism for expressing recurrences — computations where the current cycle depends on the result of a prior cycle.

## Syntax

```zir
expr@N
```

`N` is a non-negative integer literal specifying how many cycles to look back. `@1` means "the value from the immediately preceding cycle"; `@0` means "the current cycle" (occasionally useful to pass the current cycle's version of a component as an argument).

Back-references can be chained with field lookups and array subscripts:

```zir
x@1              // scalar register from the previous cycle
c@1.field        // field of component c from the previous cycle
arr@1[i]         // element i of array arr from the previous cycle
```

## Basic Example: Counter

```zir
extern GetCycle() : Val;

component Count(first: Val) {
  public a : Reg;
  a := Reg((1 + a@1) * (1 - first));
  // On cycle 0 (first=1): a = 0
  // On cycle N>0 (first=0): a = a@1 + 1
}

test count {
  first := NondetReg(Isz(GetCycle()));
  c := Count(first);
  Output(c.a);  // outputs: 0, 1, 2, 3, …
}
```

## Forward Declarations

A register that is referenced via `@1` *before* it is defined in the current cycle must be **forward-declared** so the compiler can allocate its witness column:

```zir
component Top() {
  // Forward-declare d2 and d3 so they can be back-referenced below
  d2 : Reg;
  d3 : Reg;

  d1 := Reg([first, 1-first] -> (f0, d2@1));  // d2 from previous cycle
  d2 := Reg([first, 1-first] -> (f1, d3@1));  // d3 from previous cycle
  d3 := Reg(d1 + d2);
}
```

Without the `d2 : Reg;` declaration, the compiler cannot resolve `d2@1` because `d2` has not yet been introduced.

## Back-referencing an Entire Component

You can take a back of a component instance to access its state from a prior cycle, including via field lookups:

```zir
component PrevCount(first: Val) {
  public c := Count(first);
  public prev := c@1;     // the Count component from the previous cycle
}

test prev_count {
  first := NondetReg(Isz(GetCycle()));
  c := PrevCount(first);
  Output(c@1.prev.a);   // Count.a two cycles back
}
```

## Back-referencing Parameters

A component can be passed its own previous-cycle instance as a parameter by using `@0` to name the current cycle's value when constructing the next:

```zir
component DoubleBackOne(x: NondetReg) {
  NondetReg(2 * x@1)
}

component Top() {
  first := NondetReg(IsFirstCycle());
  public result : NondetReg;
  result := [first, 1 - first] -> (
    NondetReg(1),
    DoubleBackOne(result@0)   // pass current cycle's result as the argument
  );
}
// cycle 0: 1,  cycle 1: 2,  cycle 2: 4,  cycle 3: 8, …
```

## Back of an Array

Both the whole array and individual elements can be back-referenced:

```zir
// Access element i of arr from the previous cycle
arr@1[i]

// Reduce the previous cycle's array
sum := reduce base@1 init 0 with Add;
```

Example (from `test/back_of_array.zir`):

```zir
test {
  cycle := NondetReg(GetCycle());
  first := NondetReg(Isz(cycle));

  base := [Reg(cycle + 0), Reg(cycle + 1), Reg(cycle + 2), Reg(cycle + 3)];
  result := [first, 1 - first] -> (
    0,
    reduce base@1 init 0 with Add
  );
  Log("result = %u", result);
}
// cycle 0: result = 0
// cycle 1: result = 6   (0+1+2+3)
// cycle 2: result = 10  (1+2+3+4)
```

## Fibonacci with Backs (full example)

From `dsl/examples/fibonacci.zir`:

```zir
component Top() {
  global f0: Reg;
  global f1: Reg;

  cycle := CycleCounter();
  first := cycle.is_first_cycle;

  d2 : Reg;  // forward declaration
  d3 : Reg;  // forward declaration
  d1 := Reg([first, 1-first] -> (f0, d2@1));  // d2 from previous cycle
  d2 := Reg([first, 1-first] -> (f1, d3@1));  // d3 from previous cycle
  d3 := Reg(d1 + d2);
}
```

## Constraints Across Cycles

Back-references can appear in equality constraints, not just in witness generation. For example, to enforce that a counter increments by exactly 1 each cycle:

```zir
cycle = cycle@1 + 1;
```

This polynomial constraint is checked for every pair of consecutive cycles.

## Cycle Boundary Behaviour

On the **first cycle**, reading `expr@1` accesses the value from the *last* cycle (cycles wrap around). If you want to avoid using the back value on the first cycle, guard it with a mux:

```zir
[is_first_cycle, 1 - is_first_cycle] -> (
  initial_value,
  previous_value@1
)
```

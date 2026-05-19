# Arrays and Loops

ZIR provides first-class support for fixed-size arrays and `for` loops. Both are resolved at compile time: the size of every array must be a compile-time constant, and every loop body is unrolled into a sequence of component instantiations.

## Array Literals

An array literal is a comma-separated list of values enclosed in square brackets:

```zir
arr := [1, 2, 3, 4];   // Array<Val, 4>
```

The type of an array literal is `Array<T, N>` where `T` is the element type and `N` is the number of elements (a compile-time constant).

## Range Expressions

A range `start..end` produces an `Array<Val, end-start>` of consecutive integers from `start` (inclusive) to `end` (exclusive):

```zir
r := 0..4;     // [0, 1, 2, 3]  — Array<Val, 4>
r := 4..6;     // [4, 5]        — Array<Val, 2>
```

Ranges are most commonly written directly inside `for` loops.

## For Loops (Map)

A `for` loop iterates over a range or an existing array, executing the body once per element. The result is a new array whose elements are the values produced by each iteration of the body.

### Iterating over a range

```zir
squares := for i : 0..5 {
  i * i
};
// squares = [0, 1, 4, 9, 16]  — Array<Val, 5>
```

### Iterating over an existing array

```zir
component Double(v: Val) { 2 * v }

arr := [3, 4, 5];
doubled := for x : arr { Double(x) };
// doubled = [6, 8, 10]
```

### Creating register arrays

`for` loops are commonly used to allocate arrays of registers in the witness:

```zir
extern IsFirstCycle() : Val;

component Top() {
  first := NondetReg(IsFirstCycle());

  public arr : Array<Reg, 4>;
  arr := for i : 0..4 {
    Reg([first, 1 - first] -> (
      i + 1,              // initial value on cycle 0
      (i + 1) * arr@1[i] // multiply previous value each cycle
    ))
  };
}
```

After 5 cycles this produces: `[1,2,3,4]`, `[1,4,9,16]`, `[1,8,27,64]`, …

## Array Subscripts

Individual elements of an array are accessed with `arr[index]`:

```zir
arr := [10, 20, 30];
x := arr[0];   // 10
y := arr[2];   // 30
```

The index must be a compile-time constant or a `Val` whose range can be verified (e.g. produced by `InRange`).

## Reduce

`reduce` folds an array down to a single value by repeatedly applying a binary component:

```zir
sum := reduce arr init 0 with Add;
```

The general form is:

```
reduce <array> init <seed> with <BinaryComponent>
```

`<BinaryComponent>` must accept two arguments and return a single value. The seed is the starting accumulator; the component is applied left-to-right.

Example — sum array elements from the previous cycle:

```zir
extern GetCycle() : Val;

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

## Generic Array Components

Components can be written to accept arrays of any element type and size using type parameters:

```zir
component Concatenate<T: Type, N: Val, M: Val>(a: Array<T, N>, b: Array<T, M>) {
  for i : 0..(N + M) {
    in_a := InRange(0, i, N);
    [in_a, 1 - in_a] -> (
      a[i],
      b[i - N]
    )
  }
}

component Top() {
  arr := Concatenate<Val, 4, 2>(0..4, 4..6);
  // arr = [0, 1, 2, 3, 4, 5]
}
```

## Constraints Inside Loops

Constraints written inside a `for` body apply independently to every iteration:

```zir
for i : 0..6 {
  arr[i] = i;   // constrains each element equal to its index
}
```

This generates 6 separate polynomial constraints, one per element.

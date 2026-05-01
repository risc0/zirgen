# Arrays and Loops

Zirgen provides first-class support for fixed-size arrays and two higher-order constructs
for working with them: `for` (a map over an array) and `reduce` (a fold). Together these
cover the common patterns of initializing, transforming, and aggregating structured data
inside a circuit.

## Array types and literals

The type of a fixed-size array is written `Array<T, N>` where `T` is the element type
and `N` is a compile-time constant size. For example, `Array<Reg, 4>` is an array of
four `Reg` components.

Array literals are written with square brackets and comma-separated elements:

```
[1, 2, 3]
```

The type of the above literal is inferred from the elements; if all elements are `Val`s
then the result is `Array<Val, 3>`. Elements can be any expression, including component
constructors:

```
component Top() {
  cycle := NondetReg(GetCycle());
  base := [Reg(cycle + 0), Reg(cycle + 1), Reg(cycle + 2), Reg(cycle + 3)];
}
```

Individual elements are accessed with subscript notation:

```
base[0]    // first element
base[i]    // element at index i (must be a compile-time constant)
```

Arrays can be declared without being immediately initialized using a forward declaration,
similar to the forward declarations used with backs:

```
public arr : Array<Reg, 4>;
```

## Range expressions

A compact way to produce an `Array<Val, N>` of consecutive integers is a range literal
`start..end`, which yields the values `start`, `start+1`, ..., `end-1`:

```
0..4      // [0, 1, 2, 3]
1..5      // [1, 2, 3, 4]
1..n+1    // [1, 2, ..., n]  (n must be a compile-time constant)
```

Ranges are most useful as the array argument to `for` or `reduce`.

## `for`: mapping over an array

The `for` construct applies a body expression to each element of an array and collects the
results into a new array of the same size. It is a map, not a loop: it always produces an
array and does not have side effects beyond the component construction that happens inside
its body.

```
for <induction-variable> : <array-expression> { <body> }
```

The induction variable is bound to each element of the array in turn, and the body is
evaluated once per element. The result type is `Array<T, N>` where `T` is the type of the
body expression.

A simple example that builds an array of four registers:

```
arr := for i : 0..4 { Reg(i) };
// arr has type Array<Reg, 4>
// arr[0] = 0, arr[1] = 1, arr[2] = 2, arr[3] = 3
```

A more realistic example that conditionally initializes or advances each element
depending on whether this is the first cycle:

```
extern IsFirstCycle() : Val;

component Top() {
  first := NondetReg(IsFirstCycle());

  public arr : Array<Reg, 4>;
  arr := for i : 0..4 {
    Reg([first, 1-first] -> (
      i + 1,
      (i + 1) * arr@1[i]
    ))
  };
}
```

On the first cycle each element is initialized to `i + 1`. On subsequent cycles each
element is multiplied by `i + 1` relative to its value on the previous cycle
(see [Backs](99_Backs.md) for the `@1` syntax).

`for` can also iterate over an existing array rather than a range:

```
function DoubleAll<N: Val>(arr: Array<Val, N>) {
  for x : arr { x * 2 }
}
```

## `reduce`: folding over an array

The `reduce` construct folds an array into a single value by repeatedly applying a
two-argument component. Its syntax is:

```
reduce <array-expression> init <initial-value> with <reducer>
```

The reducer must be a component (or function) that accepts two arguments: the running
accumulator and the current element. The fold proceeds left-to-right, starting from the
initial value.

Summing an array using the builtin `Add` component:

```
function Sum<N: Val>(arr: Array<Val, N>) {
  reduce arr init 0 with Add
}

component Top() {
  Sum<3>([1, 2, 3]) = 6;
}
```

Computing a factorial by reducing a range:

```
function Factorial<n: Val>() {
  reduce 1..n+1 init 1 with Mul
}

test SimpleFactorial {
  x := Factorial<4>();  // 24
}
```

Custom components can be used as reducers as long as they accept two `Val` arguments and
return something coercible to `Val`. The first argument receives the accumulator and the
second receives the current element.

## Combining `for` and `reduce`

`for` and `reduce` compose naturally. A common pattern is to build a transformed array
inline inside a `reduce`:

```
// Sum the products bits[i] * i to verify which bit is set
reduce for i : 0..N { bits[i] * i } init 0 with Add = v;
```

This is equivalent to mapping the array first and then reducing, but avoids naming an
intermediate variable.

## Taking backs on arrays

The back operator `@` works on arrays just as it does on individual registers. `arr@1`
refers to the entire array from the previous cycle, and `arr@1[i]` accesses a specific
element of that previous-cycle array:

```
result := [first, 1 - first] -> (
  0,
  reduce base@1 init 0 with Add
);
```

See [Backs](99_Backs.md) for a full explanation of the back operator.

[Prev](05_Muxes.md)

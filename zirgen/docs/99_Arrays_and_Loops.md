# Arrays and Loops

Arrays are fixed-size, homogeneous collections of components in Zirgen. Their
size is a compile-time constant (a `Val` type parameter), and their element
type can be any component type. Arrays are pervasive in Zirgen circuits because
they map naturally onto multiple columns of the execution trace and support
compact iteration patterns.

## Array Types

The built-in array type is written `Array<T, N>` where `T` is the element type
and `N` is the size as a `Val` constant. For example, `Array<Reg, 4>` describes
four register columns.

## Array Literals

An array can be created from a list of expressions enclosed in square brackets:

```
seq := [x, x + 1, x + 2];
```

This creates a three-element `Array<Val, 3>` whose elements are the given
values.

A range expression `a..b` is shorthand for the array `[a, a+1, ..., b-1]` and
is especially useful for feeding sequential values:

```
// 0..4 produces [0, 1, 2, 3]
regs := SumRegs<4>(0..4);
```

## Indexing

Individual elements are accessed with `arr[i]`:

```
Output(seq[0]);
Output(seq[1]);
Output(seq[2]);
```

## `for` — Array Comprehensions

The `for` loop creates a new array by evaluating an expression once per index.
The syntax is:

```
for <index> : <range_or_array> { <body> }
```

The `<range_or_array>` can be a range `a..b` (which iterates indices `a` through
`b-1`) or an existing array (which iterates its elements).

**Building an array of registers:**

```
arr := for i : 0..4 { Reg(i) };
// arr has type Array<Reg, 4> with values [0, 1, 2, 3]
```

**Using a declared-then-defined pattern** — declare the type first so that
back-references inside the body can refer to the previous cycle's values:

```
public arr : Array<Reg, 4>;
arr := for i : 0..4 {
  Reg([first, 1-first] -> (
    i + 1,
    (i + 1) * arr@1[i]
  ))
};
```

**Iterating over an existing array** (element `x` is bound to each element in
turn):

```
for x : Func(3) { Output(x) }
```

## `reduce` — Folding an Array

`reduce` folds an array to a single value using a binary component constructor:

```
reduce <array> init <identity> with <BinaryComponent>
```

The binary component takes two arguments (the accumulator and the next element)
and produces the new accumulator.

**Summing an array of values:**

```
function Sum<N: Val>(arr: Array<Val, N>) {
  reduce arr init 0 with Add
}
```

**Computing a product (factorial):**

```
function Factorial<n: Val>() {
  reduce 1..n+1 init 1 with Mul
}
```

**Reducing an array of registers** — here `TriangleSum` is a user-defined
binary component that accumulates across cycles:

```
component SumRegs<numRegs: Val>(vals: Array<Val, numRegs>) {
  public regs := for i : 0..numRegs { AddDoubler(vals[i]) };
  public sum  := reduce regs init 0 with TriangleSum;
}
```

## Parameterizing on Array Size

Components and functions can take the array size as a type parameter of kind
`Val`:

```
component SumRegs<numRegs: Val>(vals: Array<Val, numRegs>) { ... }
```

Call it by supplying the type argument explicitly:

```
regs := SumRegs<4>(0..4);
```

## Combining Arrays and Backs

Because arrays are laid out as contiguous columns in the witness, you can take
a back-reference on an entire array with `arr@1`, and then index into it with
`arr@1[i]`. See [Backs](99_Backs.md) for more on back-reference syntax.

```
// Each element grows by 1 compared to the previous cycle.
for i : 0..4 { arr[i] - arr@1[i] = 1; }
```

[Prev](05_Muxes.md)

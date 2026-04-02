# Arrays and Loops

Arrays and loops are fundamental constructs in Zirgen that enable you to work with collections of values and perform repeated operations. Since Zirgen circuits ultimately compile to polynomial constraints over a fixed witness layout, arrays must have compile-time known sizes, and loops are unrolled during compilation.

## Arrays

### Array Types

Arrays in Zirgen are declared using the `Array` type with a size parameter:

```zirgen
Array<ElementType, Size>
```

For example:
- `Array<Val, 8>` - an array of 8 field elements
- `Array<Reg, 16>` - an array of 16 registers
- `Array<Array<Val, 4>, 3>` - a 2D array (3 rows of 4 elements each)

### Array Literals

You can create arrays using literal syntax:

```zirgen
[1, 2, 3, 4, 5]
```

For multi-dimensional arrays:

```zirgen
[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

### Array Indexing

Access array elements using square bracket notation (zero-indexed):

```zirgen
array[0]        // First element
array[i]        // Element at index i
array[i][j]     // Multi-dimensional access
```

Example:

```zirgen
component AccessExample() {
  arr := [10, 20, 30, 40];
  first := arr[0];   // 10
  third := arr[2];   // 30
}
```

### Array Construction with Loops

Arrays can be constructed using for-loop expressions:

```zirgen
for i : 0..N { expression }
```

Example:

```zirgen
component Squares() {
  // Create array [0, 1, 4, 9, 16, 25, 36, 49]
  for i : 0..8 { i * i }
}
```

## Loops

Zirgen provides several types of loop constructs. All loops are unrolled at compile time, so loop bounds must be known at compile time.

### Indexed For Loops

The most common loop form iterates over a range of indices:

```zirgen
for i : 0..N {
  // loop body
}
```

The loop variable `i` takes values from 0 (inclusive) to N (exclusive).

Example from a rotate operation:

```zirgen
component RotateLeft<SIZE: Val>(in: Array<Val, SIZE>, n: Val) {
  for i : 0..SIZE {
    if (InRange(0, i - n, SIZE)) { in[i - n] } else { in[SIZE + i - n] }
  }
}
```

### Range Expressions

Loop ranges support arithmetic:

```zirgen
for i : 0..(N / P) {
  // Iterates N/P times
}
```

Example:

```zirgen
component Pack<N: Val, P: Val>(in: Array<Val, N>) {
  for i : 0..(N / P) {
    reduce for j : 0..P { Po2(j) * in[i*P + j] } init 0 with Add
  }
}
```

### Element Iteration

You can iterate directly over array elements:

```zirgen
for element : array {
  // process element
}
```

Example:

```zirgen
component ScaleArray(arr: Array<Val, 24>, multiplier: Val) {
  for v : arr { v * multiplier }
}
```

This is particularly useful when you don't need the index, only the values.

### Nested Loops

Loops can be nested for multi-dimensional operations:

```zirgen
component Process2D<ROWS: Val, COLS: Val>(matrix: Array<Array<Val, COLS>, ROWS>) {
  for i : 0..ROWS {
    for j : 0..COLS {
      matrix[i][j] * 2
    }
  }
}
```

Example from Keccak (3D array iteration):

```zirgen
component ThetaP1(a: Array<Array<Array<Val, 64>, 5>, 5>) {
  for j : 0..5 {
    for k : 0..64 {
      Xor5(for i : 0..5 { a[i][j][k] })
    }
  }
}
```

### Reduce Expressions

The `reduce` construct performs a fold/accumulation over an array or loop:

```zirgen
reduce array init initial_value with operation
reduce for i : 0..N { expression } init initial_value with operation
```

Components:
- `array` or `for i : 0..N { expression }` - the values to reduce
- `init` - the initial accumulator value
- `with` - the combining operation (like `Add`, or a custom component)

Example (summing an array):

```zirgen
component Sum5(vals: Array<Val, 5>) {
  reduce vals init 0 with Add
}
```

Example (dot product):

```zirgen
component DotProduct<N: Val>(a: Array<Val, N>, b: Array<Val, N>) {
  reduce for i : 0..N { a[i] * b[i] } init 0 with Add
}
```

The `with` operation can be any binary component that takes two arguments and returns a result. `Add` is the most common, but you can use custom components:

```zirgen
component CustomCombine(a: Val, b: Val) {
  a * 2 + b
}

component ReduceWithCustom(arr: Array<Val, 10>) {
  reduce arr init 0 with CustomCombine
}
```

## Common Patterns

### Array Equality Constraints

Constrain two arrays to be equal element-wise:

```zirgen
component EqArr<SIZE: Val>(a: Array<Val, SIZE>, b: Array<Val, SIZE>) {
  for i : 0..SIZE {
    a[i] = b[i];
  }
}
```

### Array Shifts and Rotations

Shift right (filling with zeros):

```zirgen
component ShiftRight<SIZE: Val>(in: Array<Val, SIZE>, n: Val) {
  for i : 0..SIZE {
    if (InRange(0, i + n, SIZE)) { in[i + n] } else { 0 }
  }
}
```

Rotate left (wrapping around):

```zirgen
component RotateLeft<SIZE: Val>(in: Array<Val, SIZE>, n: Val) {
  for i : 0..SIZE {
    if (InRange(0, i - n, SIZE)) { in[i - n] } else { in[SIZE + i - n] }
  }
}
```

### Conditional Array Processing

Use `if` expressions within loops:

```zirgen
component FilterOdd<SIZE: Val>(arr: Array<Val, SIZE>) {
  for i : 0..SIZE {
    if (i % 2 = 1) { arr[i] } else { 0 }
  }
}
```

### Array Chunking and Packing

Process arrays in chunks:

```zirgen
component Pack<N: Val, P: Val>(in: Array<Val, N>) {
  for i : 0..(N / P) {
    reduce for j : 0..P { Po2(j) * in[i*P + j] } init 0 with Add
  }
}
```

This packs every P elements into a single value using powers of 2.

### One-Hot Encoding

Create a one-hot array where only one element is 1:

```zirgen
component OneHot<N: Val>(v: Val) {
  public bits := for i : 0..N { NondetBitReg(Isz(i - v)) };
  reduce bits init 0 with Add = 1;
  reduce for i : 0..N { bits[i] * i } init 0 with Add = v;
  bits
}
```

## Type Parameters and Generics

Arrays work with type parameters, allowing you to write generic components:

```zirgen
component GenericPair<T: Type, SIZE: Val>(arr1: Array<T, SIZE>, arr2: Array<T, SIZE>) {
  for i : 0..SIZE {
    [arr1[i], arr2[i]]
  }
}
```

The `SIZE` parameter must be a `Val` type parameter, making it a compile-time constant.

## Testing with Arrays

Test cases can use arrays to verify circuit behavior:

```zirgen
test ShiftAndRotate {
  EqArr<8>(ShiftRight<8>([1, 1, 1, 0, 1, 0, 0, 0], 2), [1, 0, 1, 0, 0, 0, 0, 0]);
  EqArr<8>(RotateLeft<8>([1, 2, 3, 4, 5, 6, 7, 8], 3), [4, 5, 6, 7, 8, 1, 2, 3]);
}
```

## Key Takeaways

- Arrays have **compile-time fixed sizes** specified in the type
- Arrays are **zero-indexed**
- Loops are **unrolled at compile time** - there are no runtime loops
- Loop expressions **produce arrays** as their result
- The `reduce` construct **folds/accumulates** array values
- Element iteration (`for v : array`) is convenient when indices aren't needed
- Arrays and loops can be **nested** for multi-dimensional operations
- All loop bounds must be **compile-time constants**

For more information on builtin components used with arrays, see [Builtin Components](A1_Builtin_Components.md).

[Prev](05_Muxes.md)
[Next](99_Backs.md)


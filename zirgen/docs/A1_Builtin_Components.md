# Builtin Components

## Val

Intuitively, a `Val` is a field element. Formally, its _value_ is a
representation of that field element, its _layout_ is trivial since it need not
be stored in a register, its _constraint set_ is empty, and its witness
generation algorithm is a no-op.

`Val`s can be represented syntactically with numeric literals like `42`
(decimal), `0xFF` (hexadecimal), or `0b10101` (binary) notation, and since they
are finite field elements they can be added, subtracted, multiplied, and divided
according to the rules of modular arithmetic. These operations are themselves
components that can be coerced to `Val`, and have "identifier" names (`Add`,
`Sub`, `Mul`, `Div`) as well as the more familiar infix operators (`+`, `-`,
`*`, `/`). For example, `3 * 4` is just syntactic sugar for `Mul(3, 4)`.

## Add

```
component Add(a: Val, b: Val) : Val;
```

The super of `Add` is a `Val` which is the finite field sum of `a` and `b`.
Taking advantage of syntactic sugar, this can also be written as `a + b`.

## Sub

```
component Sub(a: Val, b: Val) : Val;
```

The super of `Sub` is a `Val` which is the sum of `a` and the additive inverse
of `b` in the finite field. Taking advantage of syntactic sugar, this can also
be written as `a - b`.

## Mul

```
component Mul(a: Val, b: Val) : Val;
```

The super of `Mul` is a `Val` which is the finite field product of `a` and `b`.
Taking advantage of syntactic sugar, this can also be written as `a * b`.

## Neg

```
component Neg(v: Val) : Val;
```

The super of `Neg` is a `Val` which is the additive inverse of `v` in the finite
field. Taking advantage of syntactic sugar, this can also be written as `-v`.

## Inv

```
component Inv(v: Val) : Val;
```

The super of `Inv` is a `Val` which is the multiplicative inverse of `v` in the
finite field. This computation is nondeterministic, so to constrain it you might
do the following:
```
vInv := NondetReg(Inv(v));
v * vInv = 1;
```

## Div

```
component Div(a: Val, b: Val) : Val;
```

The super of `Div` is a `Val` which is the result of dividing `a` by `b` in the
finite field. This computation is nondeterministic, so to constrain it you might
do the following:
```
quotient := NondetReg(a / b);
quotient * b = a;
```

## BitAnd

```
component BitAnd(a: Val, b: Val) : Val;
```

The super of `BitAnd` is a `Val` which is the bitwise and of `a` and `b`,
treating them as integers. This computation is nondeterministic.

## InRange

```
component InRange(a: Val, b: Val, c: Val) : Val;
```

The super of `InRange` is a `Val` which is 1 if, treating the arguments as
integers, `a <= b < c`, and 0 otherwise. However, if `a > c` then it causes a
runtime error. This computation is nondeterministic.

## NondetReg

```
component NondetReg(v: Val);
```

Intuitively, a `NondetReg` is a `Val` that is recorded in the witness. Formally,
its _value_ is a structure containing `v`, its _layout_ is a single column, its
_constraint set_ is empty, and its witness generation algorithm writes the given
value into its column.

## Reg

A `Reg` is a wrapper around a `NondetReg`, which additionally adds a constraint
to that column to be equal to the given value. It's definition is exactly:

```
component Reg(v: Val) {
   reg := NondetReg(v);
   v = reg;
   reg
}
```

As a rule of thumb, if `v` is computed as a polynomial of constants and other
registers, it's probably better to use `Reg` than `NondetReg`. `Val`s computed
nondeterministically or returned by externs can't be used in constraints without
registerizing, so use `NondetReg` for such values.

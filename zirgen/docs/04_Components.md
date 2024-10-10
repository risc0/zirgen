# Components

Because the ideas in this section are so central to the language, we already
touched on a lot of them in this section in the [conceptual overview](02_Conceptual_Overview.md).
This section dives deeper into the concrete details, and introduces the syntax
and semantics in greater depth.

Fundamentally, a component is defined by:

* A "value" which describes the logical structure of the component
* A "layout" which describes the structure of columns in the witness
* A set of constraints which must be satisfied by a valid witness
* An algorithm for populating its columns

The ability to compose components is a central design goal of Zirgen, so it
provides a few builtin ones to act as the core building blocks. These are
described in [Builtin Components](A1_Builtin_Components.md), and the examples in
this section assume basic familiarity with these. In terms of how components are
composed, there are two fundamental mechanisms for this: components can be
packed together into structures with named and unnamed fields, which is the
topic discussed in this section; or one can be selected from a set of alternatives
depending on a "selector value" in a mux. These closely mirror the product and
sum types often encountered in functional programming and type theory.

## Structure-like Components

In order to build our own components, we need to define constructors for our new
component types. Such constructors are created with the `component` keyword
followed by an identifier that names the new component type, which should
typically start with a capital letter. It then takes an optional list of one or
more type-annotated type parameters delimited by angle brackets, and a required
parenthesized list of zero or more type-annotated parameters. This is followed
by the constructor body, which is enclosed in curly braces.

Let's break that down a bit. Here's the simplest possible component, which takes
no arguments and has an empty body:

```
component MyTrivialComponent() {}
```

### Constructor Bodies

The body of a component constructor is made out of semicolon-terminated statements,
which are one of:
* a named member definition
* an anonymous member definition
* a constraint

We can define a named member in our constructor by providing an identifier,
followed by a definitional equality symbol `:=`, followed by an expression:

```
component MyRegPair() {
  x := Reg(2);
  y := Reg(5);
}
```

### Super Components

TODO

### Constructor Parameters

We can also parameterize the component constructor like so, where an argument is
denoted as an identifier followed by a colon followed by a type name, and
arguments are separated by commas:

```
component Pair(x: Val, y: Val) {
  x := Reg(x);
  y := Reg(y);
}
```

### Type Parameters

Types themselves can also be parametric. Type parameters are denoted in the same
way as constructor parameters, but are delimited by angle brackets between the
type name and parameter list. These are currently quite restricted in that their
types must be either `Val` or `Type` (which can be instantiated with a type
name). For example:

```
component ConstPair<X: Val, Y: Val>() {
  x := X;
  y := Y;
}
```

### A note about recursion

We've chosen to disallow components from recursively containing subcomponents
with the same constructor --- that is, `MyComponent<X: Val>` cannot contain any
other instance of `MyComponent`, even with a different type parameter. While
such constructions could be quite elegant, arbitrary recursion presents certain
theoretical difficulties for circuits and implementing restricted forms of
recursion complicate compiler implementation. In practice, this turns out to
be just fine for the types of computations typically described in arithmetic
circuits, though this design decision could be revisited in the future. However,
for now simply note that the following DOES NOT WORK and will result in a
compile-time error:

```
component Fib(n: Val) {
  isZeroOrOne := InRange(0, n, 2);
  [isZeroOrOne, 1 - isZeroOrOne] -> (
    1,
    Fib(n - 1) + Fib(n - 2)
  )
}
```

[Prev](03_Building_a_Fibonacci_Circuit.md)
[Next](05_Muxes.md)

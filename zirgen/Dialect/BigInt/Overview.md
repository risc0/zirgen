# Overview of how BigInts are used to prove things

## Prerequisite concepts

### BigInts

BigInts are arbitrary precision nonnegative integers. When we
manipulate them, we typically know the maximum value they can take on.

They're typically (but are not required to be) used to do arithmetic modulo some prime.

### Rank One Constraint System (R1CS)

The term R1CS describes a constraint system where all of the
constraints are of the form $(a*b)-c = 0$ typically modulo a large field.

The `circom` language is designed for implementing circuits to
generate proofs using these types of constraints.

### RSA

We also use BigInts to prove things relating to RSA.  Specifically, we
want to prove that $S^e = M \pmod{N}$, where e = 65537, given bigints
S, M, and N.

### BytePoly

When doing calculations involving BigInts in our proof system, we represent BigInts as a `BytePoly`.

A BytePoly consists of one or more field elements of our native field
(e.g. `BabyBear`), For a BytePoly $(b_0, b_1, b_2, ...)$ The value
of the represented BigInt is $b_0 + 256 b_1 + 256^2 b_2 + ...$ .

When we're processing operations in the BigInt dialect, we keep the
following information about each BigInt we manpulate:

* Number of polynomial coefficients, i.e. the number of native field
  elements that comprise the BytePoly.
* Range of each element (both maximum and minimum)
* Optionally, a power of 2 minimum bound of the represented BigInt.  I.e., a "N" such that the BigInt is >= 2^N


When we initially import a big integer as a BytePoly format, all of
the elements will be in the range $[0, 255]$  However, that range can
expand during calculations; for instance, if we add two initially
imported BytePolys, each element will be in the range $[0, 510]$ .  If
we perform subtraction, each element will be in the range
$[-255, 255]$. If we multiply them, each element will be in the range
$[0, 65025]$.

Under the hood, all coefficients are BabyBear field elements. Negative
numbers are represented as subtracted from the BabyBear prime (that is,
the number `-n` is represented as `P - n`). The type system checks that
the range of possible values never overflows, i.e., that every negative
value has a representation that is larger than every positive value.

### ZKR

ZKR is a control-tree input interpreted by the RISC Zero recursion
circuit.  It provides operations including field arithmetic in the
native field (e.g. BabyBear), and computing cryptographic digests such
as SHA256 and Poseidon2.

Notably, ZKR does not provide any support for control flow; all
operations present are executed in sequence.

## Proving overview

We structure a BigInt proof as containing public witness values,
private witness values, and constant witness values.

A naive attempt to prove BigInt constraints might just be to attempt
to evaluate the constraints within ZKR.  However, some operations
(like BigInt modulo) are problematic to implement with the facilities
that ZKR provides.

Instead, we first add additional witnesses to the BigInt proof that
contains data extracted from intermediate unconstrained operations
like division and modulo.  We then add constraints that those
operations were executed correctly.

For instance to evaluate a bigint modulo $r := a \bmod b$, we would instead use

- $q := a \, \text{div}_\text{unconstrained} \, b$
- $r := a \, \text{mod}_\text{unconstrained} \, b$
- $\text{constrain} (b * q + r = a)$

and add $q$ and $r$ as additional witness values.

After committing to the witnesses of the proof, we use
fiat-shamir to generate a random challenge $Z \in \mathbb{F}_{p^4}$.
This is also known as the `evaluation point`.

Though the BytePoly representation of a BigInt uses base 256,
operations on BytePoly treat it as a polynomial, so any BytePoly that
evaluates to 0 with a base 256 will also evaluate to zero in any other base.

In other words, for a BytePoly $(b_0, b_1, b_2, ...)$, this normally
represents a big integer $(b_0 + 256 b_1 + 256^2 b_2 + ...)$, but if we have
a constraint that it equals zero, we can also evaluate it as
$b_0 + b_1 Z + b_2 Z^2 + ...$ and it will also equal zero.

Evaluating at the evaluation point `Z` prevents a malicious actor from
being able to find a set of witness values where a constraint
evaluates to zero when the constraint isn't actually fulfillfed, since
the evaluation point isn't known until after a commitment has been
made to the witness values.

## Code references

One implementation of generating the BigInt witness is in [eval](IR/Eval.cpp).

The ZKR implementation of verifying the BigInt witness is in [lower](IR/Lower.cpp).

An test of much of the end-to-end functionality of proving a RSA constraint is in [RSA test](IR/test/test.cpp).


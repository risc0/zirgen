# Zirgen Standard Library

This directory contains implementations of the most common components that are
likely to be reused across lots of circuits. The provided components are divided
into three tiers, each larger than the last:

1. Built-ins. These are the most fundamental components of the language, which
   cannot be implemented in the language itself. For example, it includes things
   like finite field operations and `NondetReg`.
2. Preamble. These are slight abstractions around the built-ins that are
   implicitly included in every Zirgen file. For example, it includes `Reg` and
   declarations for common externs like `Log`.
3. The Standard Library. Things that any particular circuit will _likely_ need,
   but not necessarily so. For example, it includes `OneHot` and `BitReg`.

This directory contains the third.

# The Zirgen Circuit Language

## Introduction

The Zirgen circuit language is a domain-specific language for writing arithmetic
circuits for the RISC Zero proof system. For the most part, users of RISC Zero
should be writing their zkApps in Rust and using our RISC-V zkVM rather than
building them directly in Zirgen. However, sometimes it is necessary or
desirable to write parts of an application as an arithmetic circuit to integrate
directly with our proof system and achieve better performance. In fact, our zkVM
itself is a circuit written in Zirgen! Zirgen makes it possible to build
accelerators for important parts of your applications, or even to build entire
other VMs that integrate into the RISC Zero ecosystem through proof composition.
If that sounds like your goal, read on!

So, what kinds of things can you build with Zirgen?
* zk accelerators (hashing, bigint operations, cryptographic primitives)
* zkVMs
* recursion circuits
* arbitrary zkApps (though we recommend doing this in Rust instead!)

Table of Contents:
1. [Getting Started](01_Getting_Started.md)
2. [Basic Concepts](02_Conceptual_Overview.md)

[Next](01_Getting_Started.md)

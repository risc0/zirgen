# Zirgen Circuit Compiler

Zirgen is a compiler for a domain-specific language, also called "zirgen",
which creates arithmetic circuits for the RISC Zero proof system.

For the most part, users of RISC Zero should be writing their zkApps in Rust
and using our RISC-V zkVM rather than building them directly in Zirgen.
Sometimes, however, it is necessary or desirable to write parts of an
application as an arithmetic circuit to integrate directly with our proof system
and achieve better performance. In fact, our zkVM itself is a circuit written
in Zirgen! Zirgen makes it possible to build accelerators for important parts
of your applications, or even to build entire other VMs that integrate into the
RISC Zero ecosystem through proof composition. If that sounds like your goal,
read on!

[Getting Started](zirgen/docs/01_Getting_Started.md)

[Language Overview](zirgen/docs/02_Conceptual_Overview.md)


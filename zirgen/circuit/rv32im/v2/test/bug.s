
.section .text
.global _start
_start:
  li a0, 7
  li a1, 5
  remu a2, a0, a1
  li a3, 2
  beq a2, a3, good
  li a0, 0
  li a7, 0
  ecall
good:
  fence


INST_TESTS = [
    "add",
    "addi",
    "and",
    "andi",
    "auipc",
    "beq",
    "bge",
    "bgeu",
    "blt",
    "bltu",
    "bne",
    "jal",
    "jalr",
    "lb",
    "lbu",
    "lh",
    "lhu",
    "lui",
    "lw",
    "or",
    "ori",
    "sb",
    "sh",
    "simple",
    "sll",
    "slli",
    "slt",
    "slti",
    "sltiu",
    "sltu",
    "sra",
    "srai",
    "srl",
    "srli",
    "sub",
    "sw",
    "xor",
    "xori",
    "div",
    "divu",
    "mul",
    "mulh",
    "mulhsu",
    "mulhu",
    "rem",
    "remu",
]

def riscv_test_suite():
    for test in INST_TESTS:
        native.py_test(
            # tags = ["manual"],
            name = test + "_test",
            srcs = ["run_test.py"],
            main = "run_test.py",
            args = [test],
            data = [
                "//zirgen/circuit/rv32im/shared/test:riscv_test_bins",
                ":risc0-simulate",
            ],
            size = "large",
        )

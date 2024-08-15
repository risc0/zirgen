# Copyright (c) 2023 RISC Zero, Inc.
# All rights reserved.

#!/usr/bin/env python

import os
import subprocess
import pathlib
import sys

EXTENSIONS = [
    ".cpp",
    ".h",
]

ROOT_DIRS = [
    "risc0",
    "zirgen",
]


def main():
    clang_format = pathlib.Path(sys.prefix) / "bin" / "clang-format"
    print(clang_format)
    for root_dir in ROOT_DIRS:
        root = pathlib.Path(os.getenv("BUILD_WORKSPACE_DIRECTORY")) / root_dir
        for root, _, files in os.walk(root):
            for file in files:
                path = pathlib.Path(root) / file
                if path.suffix in EXTENSIONS:
                    print(".", end="", flush=True)
                    subprocess.run([clang_format, "-i", path])
    print()


if __name__ == "__main__":
    main()

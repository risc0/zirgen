#!/usr/bin/env python

# Copyright (c) 2023 RISC Zero, Inc.
# All rights reserved.

import os
import subprocess
import pathlib
import sys
import argparse
import shutil

GENERATED_DIR = "/zirgen-generated/"

parser = argparse.ArgumentParser(
    description='Check that files in the repository match generated files.')
parser.add_argument('--install', action='store_true',
                    help="Install new generated files that don't match the ones in the repository")
parser.add_argument('file', nargs='+',
                    help='Files to process')
args = parser.parse_args()

root = pathlib.Path(os.getenv("BUILD_WORKSPACE_DIRECTORY") or ".")

def read_and_strip(fn):
    with open(fn, 'r', errors="ignore") as f:
        # Ignore any formatting differences by stripping out all spaces and newlines.
        # Rustfmt sometimes likes to put commas at the end of lists, so strip all commas also.
        return "".join([line.strip().replace(" ","").replace(",","") for line in f.readlines()])

all_matched = True

for gen_fn in args.file:
    if GENERATED_DIR not in gen_fn:
        print(f"Unable to find generated directory {GENERATED_DIR} in {gen_fn} ", file=sys.stderr)
    repo_fn = gen_fn.replace(GENERATED_DIR, "/")
    try:
        repo_contents = read_and_strip(root / repo_fn)
    except FileNotFoundError:
        print(f"Unable to open {root} / {repo_fn}")
        repo_contents = ""
    gen_contents = read_and_strip(gen_fn)

    if gen_contents != repo_contents:
        print(f"{gen_fn} does not match {repo_fn}", file=sys.stderr)
        all_matched = False
        if args.install:
            target = root / repo_fn;
            print(f"Copying {gen_fn} to {target}")
            shutil.copyfile(gen_fn, target)
            # Attempt to format if tools are available so generated
            # code is a bit nicer to look at.
            if target.match("*.cpp.inc"):
                os.system(f"clang-format -i {target}")
            elif target.match("*.rs.inc"):
                os.system(f"rustfmt {target}")
        else:
            target = os.getenv("TEST_TARGET")
            if target and "TestFresh" in target:
                generate_target = target.replace("TestFresh", "Generate")
                print("Please regenerate files with")
                print(f"  bazel run {generate_target}")

if (not args.install) and (not all_matched):
    sys.exit(1)

if all_matched and args.install:
    print("All generated files up to date")

sys.exit(0)

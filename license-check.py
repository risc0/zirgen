#!/usr/bin/env python

import sys
import os
from pathlib import Path
from datetime import date
import subprocess

CWD = Path('.').absolute()

PRIVATE_HEADER = Path('LICENSE').read_text().splitlines()

PUBLIC_HEADER = '''
// Copyright {YEAR} RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This code is automatically generated
'''.strip().splitlines()

ALL_HEADERS = [PRIVATE_HEADER, PUBLIC_HEADER]

IGNORED_DIRS = [
    Path('.git').absolute(),
    Path('target').absolute(),
    Path('tmp').absolute(),
    Path('zirgen/bootstrap/target').absolute(),
]

IGNORED_FILES = []

EXTENSIONS = [
    '.cpp',
    '.h',
    '.rs',
]


def check_header(expected_year, lines_actual):
    errs = []
    for lines_expected in ALL_HEADERS:
        for (expected, actual) in zip(lines_expected, lines_actual):
            expected = expected.replace('{YEAR}', expected_year)
            if expected != actual:
                errs.append((expected, actual))
                break
    if len(errs) == len(ALL_HEADERS):
        return errs[0]
    return None


def check_file(file):
    cmd = ['git', 'log', '-1', '--format=%ad', '--date=format:%Y', file]
    expected_year = subprocess.check_output(cmd, encoding='UTF-8').strip()
    rel_path = file.relative_to(CWD)
    lines = file.read_text().splitlines()
    result = check_header(expected_year, lines)
    if result:
        print(f'{rel_path}: invalid header!')
        print(f'  expected: {result[0]}')
        print(f'    actual: {result[1]}')
        return 1
    return 0


def main():
    ret = 0
    for root, dirs, files in os.walk('.'):
        root = Path(root).absolute()
        if root in IGNORED_DIRS:
            dirs[:] = []
            continue
        for file in files:
            file = Path(file)
            path = root / file
            if path not in IGNORED_FILES and file.suffix in EXTENSIONS:
                ret |= check_file(root / file)
    sys.exit(ret)


if __name__ == "__main__":
    main()

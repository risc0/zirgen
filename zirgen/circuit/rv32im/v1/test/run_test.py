# Copyright 2022 Risc0, Inc.
# All rights reserved.

#!/usr/bin/env python

import subprocess
import sys

sys.exit(
    subprocess.run([
        "zirgen/circuit/rv32im/v1/test/risc0-simulate",
        "zirgen/circuit/rv32im/shared/test/" + sys.argv[1], sys.argv[2]
    ]).returncode)

// Copyright 2024 RISC Zero, Inc.
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

#include <cstdint>

inline void fail() {
  asm("fence\n");
}

inline void sys_log(const char* str) {
  unsigned len = 0;
  while (str[len]) {
    len++;
  }

  register uintptr_t t0 asm("t0") = 2; // SOFTWARE
  register uintptr_t a0 asm("a0") = 0; // nullptr
  register uintptr_t a1 asm("a1") = 0; // Read 0 data from host
  register uintptr_t a2 asm("a2") = 1; // SYS_LOG
  register uintptr_t a3 asm("a3") = reinterpret_cast<uintptr_t>(str);
  register uintptr_t a4 asm("a4") = len;
  asm volatile("ecall"
               : "+r"(a0), "+r"(a1)                                   // outputs
               : "r"(t0), "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4) // inputs
               :                                                      // clobbers
  );
}

inline void sys_halt() {
  register uintptr_t t0 asm("t0") = 0;     // HALT
  register uintptr_t a0 asm("a0") = 0;     // Normal
  register uintptr_t a1 asm("a1") = 0x400; // Write whatever is in lowmem
  asm volatile("ecall"
               :                           // outputs
               : "r"(t0), "r"(a0), "r"(a1) // inputs
               :                           // clobbers
  );
}

inline void sys_pause() {
  register uintptr_t t0 asm("t0") = 0;     // HALT
  register uintptr_t a0 asm("a0") = 1;     // Pause
  register uintptr_t a1 asm("a1") = 0x400; // Write whatever is in lowmem
  asm volatile("ecall"
               :                           // outputs
               : "r"(t0), "r"(a0), "r"(a1) // inputs
               :                           // clobbers
  );
}

inline void sys_sha_pair(uint32_t* out, const uint32_t* in, const uint32_t* a, const uint32_t* b) {
  register uintptr_t t0 asm("t0") = 3; // SHA
  register uint32_t* a0 asm("a0") = out;
  register const uint32_t* a1 asm("a1") = in;
  register const uint32_t* a2 asm("a2") = a;
  register const uint32_t* a3 asm("a3") = b;
  register uintptr_t a4 asm("a4") = 1;
  asm volatile("ecall"
               : "+r"(a0), "+r"(a1)                                   // outputs
               : "r"(t0), "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4) // inputs
               :                                                      // clobbers
  );
}

inline void
sys_sha_buffer(uint32_t* state_out, const uint32_t* state_in, const char* buf, uint32_t count) {
  register uintptr_t t0 asm("t0") = 3; // SHA
  register uint32_t* a0 asm("a0") = state_out;
  register const uint32_t* a1 asm("a1") = state_in;
  register const char* a2 asm("a2") = buf;
  register const char* a3 asm("a3") = buf + 32;
  register uintptr_t a4 asm("a4") = count;
  asm volatile("ecall"
               : "+r"(a0), "+r"(a1)                                   // outputs
               : "r"(t0), "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4) // inputs
               :                                                      // clobbers
  );
}

inline uint32_t sys_io(uint32_t* recvPtr,
                       uint32_t recvChunks,
                       const uint32_t* sendPtr,
                       uint32_t sendLen,
                       uint32_t channel) {
  register uintptr_t t0 asm("t0") = 2; // SOFTWARE
  register uintptr_t a0 asm("a0") = reinterpret_cast<uintptr_t>(recvPtr);
  register uintptr_t a1 asm("a1") = recvChunks;
  register uintptr_t a2 asm("a2") = 2; // SYS_IO
  register uintptr_t a3 asm("a3") = reinterpret_cast<uintptr_t>(sendPtr);
  register uintptr_t a4 asm("a4") = sendLen;
  register uintptr_t a5 asm("a5") = channel;
  asm volatile("ecall"
               : "+r"(a0), "+r"(a1)                                            // outputs
               : "r"(t0), "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5) // inputs
               :                                                               // clobbers
  );

  return a0;
}

inline uint32_t sys_input(uint32_t index) {
  register uintptr_t t0 asm("t0") = 1; // INPUT
  register uintptr_t a0 asm("a0") = index;
  asm volatile("ecall"
               : "+r"(a0)         // outputs
               : "r"(t0), "r"(a0) // inputs
               :                  // clobbers
  );
  return a0;
}

inline void
sys_bigint(uint32_t* result, const uint32_t* x, const uint32_t* y, const uint32_t* mod) {
  asm volatile("li t0, 4\n" // BigInt
               "mv a0, %0\n"
               "li a1, 0\n"
               "mv a2, %1\n"
               "mv a3, %2\n"
               "mv a4, %3\n"
               "ecall\n"
               :                                              // outputs
               : "r"(result), "r"(x), "r"(y), "r"(mod)        // inputs
               : "t0", "a0", "a1", "a2", "a3", "a4", "memory" // clobbers
  );
}

inline void sys_bigint2(uint32_t* entry,
                        const uint32_t* a = nullptr,
                        const uint32_t* b = nullptr,
                        const uint32_t* c = nullptr,
                        const uint32_t* d = nullptr) {
  uint32_t* nondetProg = entry + 4;
  uint32_t* verifyProg = nondetProg + entry[0];
  uint32_t* consts = verifyProg + entry[1];
  uint32_t tmpSpace = entry[3] * 4;

  asm volatile("li t0, 6\n" // BigInt
               "mv t1, %0\n"
               "mv t2, %1\n"
               "mv t3, %2\n"
               "sub sp, sp, %3\n"
               "mv a1, %4\n"
               "mv a2, %5\n"
               "mv a3, %6\n"
               "mv a4, %7\n"
               "ecall\n"
               "add sp, sp, %3\n"
               : // outputs
               : "r"(nondetProg),
                 "r"(verifyProg),
                 "r"(consts),
                 "r"(tmpSpace),
                 "r"(a),
                 "r"(b),
                 "r"(c),
                 "r"(d)                                                         // inputs
               : "t0", "t1", "t2", "t3", "a0", "a1", "a2", "a3", "a4", "memory" // clobbers
  );
}

inline void sys_usermode() {
  asm volatile("li t0, 5\n" // Jump to usermode
               "ecall\n"
               :                // outputs
               :                // inputs
               : "t0", "memory" // clobbers
  );
}

asm(R"(
.section .text._start;
.globl _start;
_start:
    .option push;
    .option norelax;
    la gp, __global_pointer$;
    .option pop;
    la sp, __stack_init$;
    jal ra, start
)");

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

#include "zirgen/circuit/rv32im/v1/test/runner.h"

#include <random>

#include "zirgen/circuit/rv32im/v1/platform/opcodes.h"
#include "zirgen/circuit/rv32im/v1/platform/page_table.h"
#include "zirgen/compiler/zkp/sha256.h"
#include "llvm/Support/Format.h"

namespace zirgen::rv32im_v1 {

namespace Syscall {

static constexpr uint32_t kLog = 1;
static constexpr uint32_t kIO = 2;

} // namespace Syscall

using namespace Zll;

void PageFaultInfo::include(uint32_t addr, IncludeDir dir) {
  // if (dir & IncludeDir::Write) {
  //   llvm::errs() << "include: " << llvm::format_hex(addr * kWordSize, 10) << "\n";
  // }
  while (true) {
    uint32_t pageIndex = getPageIndex(addr);
    // uint32_t pageAddr = getPageAddr(pageIndex);
    uint32_t entryAddr = kPageTableAddr + pageIndex * kDigestWords;
    // if (dir & IncludeDir::Write) {
    //   llvm::errs() << "  addr: " << llvm::format_hex(addr * kWordSize, 10)
    //                << ", pageIndex: " << llvm::format_hex(pageIndex, 10)
    //                << ", pageAddr: " << llvm::format_hex(pageAddr * kWordSize, 10)
    //                << ", entryAddr: " << llvm::format_hex(entryAddr * kWordSize, 10) << "\n";
    // }
    if (dir & IncludeDir::Read) {
      reads.insert(pageIndex);
    }
    if (dir & IncludeDir::Write) {
      writes.insert(pageIndex);
    }
    if (pageIndex == info.rootIndex) {
      break;
    }
    addr = entryAddr;
  }
}

void PageFaultInfo::dump() {
  llvm::errs() << "PageFaultInfo\n";
  for (uint32_t pageIndex : reads) {
    llvm::errs() << "  " << llvm::format_hex(pageIndex, 10) << "\n";
  }
}

std::vector<uint64_t> Runner::doExtern(llvm::StringRef name,
                                       llvm::StringRef extra,
                                       llvm::ArrayRef<const InterpVal*> args,
                                       size_t outCount) {
  auto fpArgs = asFpArray(args);
  if (name == "setUserMode") {
    userMode = fpArgs[0];
    return {};
  }
  if (name == "isTrap") {
    // TODO: tests for trap
    return {0};
  }
  if (name == "halt") {
    if (!isHalted) {
      uint32_t exitCode = fpArgs[0];
      uint32_t pc = fpArgs[1];
      switch (exitCode) {
      case HaltType::kTerminate:
        lastPc = pc;
        llvm::errs() << "HALT> pc: " << llvm::format_hex(lastPc, 10) << "\n";
        break;
      case HaltType::kPause:
        lastPc = pc + 4;
        llvm::errs() << "PAUSE> pc: " << llvm::format_hex(lastPc, 10) << "\n";
        break;
      case HaltType::kSystemSplit:
        lastPc = pc;
        llvm::errs() << "SPLIT> pc: " << llvm::format_hex(lastPc, 10) << "\n";
        break;
      }
      isHalted = true;
    }
    return {};
  }
  if (name == "syscallInit") {
    syscallPending.clear();
    syscallA0Out = 0;
    syscallA1Out = 0;
    uint32_t id = loadU32(RegAddr::kA2);
    switch (id) {
    case Syscall::kLog: {
      uint32_t str = loadU32(RegAddr::kA3);
      uint32_t len = loadU32(RegAddr::kA4);
      llvm::errs() << "GUEST_LOG: ptr = " << str << ", len = " << len << "\n";
    } break;
    case Syscall::kIO: {
      uint32_t recvPtr = loadU32(RegAddr::kA0);
      uint32_t recvWords = loadU32(RegAddr::kA1);
      uint32_t sendPtr = loadU32(RegAddr::kA3);
      uint32_t sendLen = loadU32(RegAddr::kA4);
      uint32_t channel = loadU32(RegAddr::kA5);
      llvm::errs() << "  SYS_IO channel: " << channel
                   << ", sendPtr: " << llvm::format_hex(sendPtr, 10) << ", sendLen: " << sendLen
                   << ", recvPtr: " << llvm::format_hex(recvPtr, 10) << ", recvWords: " << recvWords
                   << "\n";
      syscallPending.insert(syscallPending.end(), input.begin(), input.end());
      syscallA0Out = input.size() * kWordSize;
    } break;
    default:
      llvm::errs() << "*** UNKNOWN SYSCALL: Ignoring ***\n";
    }

    llvm::errs() << "syscall id=" << id << "\n";
    return {};
  }
  if (name == "syscallBody") {
    size_t val = 0;
    if (!syscallPending.empty()) {
      val = syscallPending.front();
      syscallPending.pop_front();
    }
    return {(val >> 0) & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF, (val >> 24) & 0xFF};
  }
  if (name == "syscallFini") {
    return {(syscallA0Out >> 0) & 0xFF,
            (syscallA0Out >> 8) & 0xFF,
            (syscallA0Out >> 16) & 0xFF,
            (syscallA0Out >> 24) & 0xFF,
            (syscallA1Out >> 0) & 0xFF,
            (syscallA1Out >> 8) & 0xFF,
            (syscallA1Out >> 16) & 0xFF,
            (syscallA1Out >> 24) & 0xFF};
  }
  if (name == "trace") {
    return {};
  }
  if (name == "pageInfo") {
    uint32_t pc = fpArgs[0];
    uint32_t inst = loadU32(pc / kWordSize);
    auto info = getPageFaultInfo(pc, inst);
    for (auto it = info.reads.rbegin(); it != info.reads.rend(); ++it) {
      uint32_t pageIndex = *it;
      if (!finishedPageReads.count(pageIndex)) {
        finishedPageReads.insert(pageIndex);
        llvm::errs() << "  pageRead> pc: " << llvm::format_hex(pc, 10)
                     << ", inst: " << llvm::format_hex(inst, 10)
                     << ", pageIndex: " << llvm::format_hex(pageIndex, 10) << "\n";
        return {1, pageIndex, 0};
      }
    }
    if (isFlushing) {
      auto it = dirtyPages.begin();
      if (it != dirtyPages.end()) {
        uint32_t pageIndex = *it;
        dirtyPages.erase(it);
        llvm::errs() << "  pageWrite> pc: " << llvm::format_hex(pc, 10)
                     << ", inst: " << llvm::format_hex(inst, 10)
                     << ", pageIndex: " << llvm::format_hex(pageIndex, 10) << "\n";
        return {0, pageIndex, 0};
      }
      return {0, 0, 1};
    }
    return std::vector<uint64_t>(outCount);
  }
  if (name == "ramRead") {
    uint32_t addr = fpArgs[0];
    uint32_t memOp = fpArgs[1];
    uint32_t pageIndex = getPageIndex(addr / kWordSize);
    uint32_t pageAddr = getPageAddr(pageIndex);
    if (memOp == MemoryOpType::kPageIo) {
      residentWords.insert(addr);
    } else {
      if (!residentWords.count(addr)) {
        llvm::errs() << "  ramRead: " << llvm::format_hex(addr * kWordSize, 10) //
                     << ", memOp: " << memOp
                     << ", pageAddr: " << llvm::format_hex(pageAddr * kWordSize, 10) << "\n";
      }
      assert(residentWords.count(addr) && "Memory read before page in");
    }
    return RamExternHandler::doExtern(name, extra, args, outCount);
  }
  if (name == "ramWrite") {
    uint32_t addr = fpArgs[0];
    uint32_t memOp = fpArgs[5];
    if (memOp == MemoryOpType::kPageIo) {
      residentWords.insert(addr);
    } else {
      assert(residentWords.count(addr) && "Memory write before page in");
    }
    return RamExternHandler::doExtern(name, extra, args, outCount);
  }
  if (name == "getMajor") {
    uint32_t cycle = fpArgs[0];
    uint32_t pc = fpArgs[1];
    uint32_t inst = loadU32(pc / kWordSize);
    auto opcode = getOpcodeInfo(inst);
    if (opcode.major == MajorType::kMuxSize) {
      throw std::runtime_error("Invalid major opcode");
    }
    auto info = getPageFaultInfo(pc, inst);
    // info.dump();
    for (uint32_t pageIndex : info.reads) {
      if (!finishedPageReads.count(pageIndex)) {
        return {MajorType::kPageFault};
      }
    }
    if (isFlushing) {
      if (!info.forceFlush || !dirtyPages.empty()) {
        return {MajorType::kPageFault};
      }
    } else {
      // TODO: Only add to dirtyPages if the next instruction will require a flush.
      dirtyPages.insert(info.writes.begin(), info.writes.end());
      if (info.forceFlush || needsFlush(cycle)) {
        isFlushing = true;
        return {MajorType::kPageFault};
      }
    }
    llvm::errs() << "  Mnemonic: " << opcode.mnemonic << "\n";
    return {opcode.major};
  }
  if (name == "getMinor") {
    uint32_t inst = fpArgs[0] | (fpArgs[1] << 8) | (fpArgs[2] << 16) | (fpArgs[3] << 24);
    auto opcode = getOpcodeInfo(inst);
    if (opcode.major == MajorType::kMuxSize) {
      throw std::runtime_error("Invalid major opcode");
    }
    if (opcode.minor == kMinorMuxSize) {
      throw std::runtime_error("Invalid minor opcode");
    }
    return {opcode.minor};
  }
  if (name == "divide") {
    uint32_t numer = fpArgs[0] | (fpArgs[1] << 8) | (fpArgs[2] << 16) | (fpArgs[3] << 24);
    uint32_t denom = fpArgs[4] | (fpArgs[5] << 8) | (fpArgs[6] << 16) | (fpArgs[7] << 24);
    uint32_t signType = fpArgs[8];
    uint32_t onesComp = (signType == 2);
    bool negNumer = signType && int32_t(numer) < 0;
    bool negDenom = signType == 1 && int32_t(denom) < 0;
    if (negNumer) {
      numer = -numer - onesComp;
    }
    if (negDenom) {
      denom = -denom - onesComp;
    }
    uint32_t quot;
    uint32_t rem;
    if (denom == 0) {
      quot = 0xffffffff;
      rem = numer;
    } else {
      quot = numer / denom;
      rem = numer % denom;
    }
    uint32_t quotNegOut = (negNumer ^ negDenom) - ((denom == 0) * negNumer);
    uint32_t remNegOut = negNumer;
    if (quotNegOut) {
      quot = -quot - onesComp;
    }
    if (remNegOut) {
      rem = -rem - onesComp;
    }
    return {(quot >> 0) & 0xff,
            (quot >> 8) & 0xff,
            (quot >> 16) & 0xff,
            (quot >> 24) & 0xff,
            (rem >> 0) & 0xff,
            (rem >> 8) & 0xff,
            (rem >> 16) & 0xff,
            (rem >> 24) & 0xff};
  }
  if (name == "bigintQuotient") {
    // Division of two little-endian positive byte-limbed bigints. a = q * b + r.
    // Assumes a and b are both normalized with limbs in range [0, 255].
    // Throws if the quotient overflows BigInt::kByteWidth. Overflows will not happen if the
    // numberator, a, is the result of a multiplication of two numbers less than the denomintor.
    // The BigInt arithmatic circuit does not accept larger quotients.
    // Returns only the quotient value q as the BigInt circuit does not use the r value.
    std::vector<uint64_t> a = llvm::ArrayRef(fpArgs).slice(0, BigInt::kByteWidth * 2).vec();
    std::vector<uint64_t> b =
        llvm::ArrayRef(fpArgs).slice(BigInt::kByteWidth * 2, BigInt::kByteWidth).vec();
    std::vector<uint64_t> q(BigInt::kByteWidth, 0);

    // This is a variant of school-book multiplication.
    // Reference the Handbook of Elliptic and Hyper-elliptic Cryptography alg. 10.5.1

    // Determine n, the width of the denominator, and check for a denominator of zero.
    size_t n = BigInt::kByteWidth;
    while (n > 0 && b.at(n - 1) == 0) {
      n--;
    }
    if (n == 0) {
      // Divide by zero is strictly undefined, but the BigInt multiplier circuit uses a modulus of
      // zero as a special case to support "checked multiply" of up to 256-bits.
      // Return zero here to facilitate this.
      return q;
    }
    if (n < 2) {
      // TODO(victor): Not an important case. But we should likely handle it anyway.
      throw std::runtime_error("bigint quotient: denominator must be at least 9 bits");
    }
    // Pad the denominator with a zero to avoid edge cases later.
    if (n == BigInt::kByteWidth) {
      b.emplace_back(0);
    }
    size_t m = a.size() - n;

    // Shift (i.e. multiply by two) the inputs a and b until the leading bit of b is 1.
    // Note that shifting both numerator and denominator has no effect on the quotient.
    uint64_t dBits = 0;
    while ((b.at(n - 1) & (0x80 >> dBits)) == 0) {
      dBits++;
    }
    uint64_t carry = 0;
    for (size_t i = 0; i < n; i++) {
      uint64_t tmp = (b.at(i) << dBits) + carry;
      b.at(i) = tmp & 0xFF;
      carry = tmp >> 8;
    }
    if (carry != 0) {
      throw std::runtime_error("implementation error: final carry in input shift");
    }
    for (size_t i = 0; i < a.size() - 1; i++) {
      uint64_t tmp = (a.at(i) << dBits) + carry;
      a.at(i) = tmp & 0xFF;
      carry = tmp >> 8;
    }
    a.emplace_back(carry);

    for (size_t i = m;; i--) {
      // Approximate how many multiples of b can be subtracted. May overestimate by up to one.
      uint64_t qApprox =
          std::min<uint64_t>(((a.at(i + n) << 8) + a.at(i + n - 1)) / b.at(n - 1), 255);
      while ((qApprox * ((b.at(n - 1) << 8) + b.at(n - 2))) >
             ((a.at(i + n) << 16) + (a.at(i + n - 1) << 8) + a.at(i + n - 2))) {
        qApprox--;
      }

      // Subtract multiples of the denominator from a.
      uint64_t borrow = 0;
      for (size_t j = 0; j <= n; j++) {
        uint64_t sub = qApprox * b.at(j) + borrow;
        if (a.at(i + j) < (sub & 0xFF)) {
          a.at(i + j) += 0x100 - (sub & 0xFF);
          borrow = (sub >> 8) + 1;
        } else {
          a.at(i + j) -= sub & 0xFF;
          borrow = sub >> 8;
        }
      }
      if (borrow > 0) {
        // Oops, went negative. Add back one multiple of b.
        qApprox--;
        uint64_t carry = 0;
        for (size_t j = 0; j <= n; j++) {
          uint64_t tmp = a.at(i + j) + b.at(j) + carry;
          a.at(i + j) = tmp & 0xFF;
          carry = tmp >> 8;
        }
        // Adding back one multiple of b should go from negative back to positive.
        if (borrow - carry != 0) {
          throw std::runtime_error("implementation error: underflow in bigint division");
        }
      }

      if (i < q.size()) {
        q.at(i) = qApprox;
      } else if (qApprox != 0) {
        throw std::runtime_error("bigint quotient: quotient exceeds allowed size");
      }

      if (i == 0) {
        break;
      }
    }
    return q;
  }

  return RamExternHandler::doExtern(name, extra, args, outCount);
}

bool Runner::needsFlush(uint32_t cycle) {
  // It takes 1152 cycles to compute a SHA-256 digest for a 1024-byte page.
  return false;
}

int32_t signExtend12(uint32_t in) {
  if (in & 0x800) {
    in |= 0xfffff000;
  }
  return static_cast<int32_t>(in);
}

PageFaultInfo Runner::getPageFaultInfo(uint32_t pc, uint32_t inst) {
  PageFaultInfo info;
  // While it's not technically true that all instructions cause writes to at least one system
  // register, it's safe to do this because the only cost is doing an extra hash of a page that
  // isn't actually dirty. The benefit is that we don't have to identity instructions that don't
  // have any system register mutations.
  uint32_t regOff = userMode ? kUserRegisterOffset : kRegisterOffset;
  info.include(regOff, IncludeDir::Both);
  info.include(pc / kWordSize);

  auto op = getOpcodeInfo(inst);
  switch (op.major) {
  case MajorType::kMuxSize:
    throw std::runtime_error("Invalid major opcode");
    break;
  case MajorType::kMemIO: {
    if (op.minor < 5) {
      // load: I-type
      uint32_t rs1 = (inst >> 15) & 0x1f;
      int32_t imm = signExtend12((inst >> 20) & 0xfff);
      uint32_t base = loadU32(regOff + rs1);
      uint32_t addr = base + imm;
      llvm::errs() << "  load: " << llvm::format_hex(inst, 10);
      llvm::errs() << ", x" << rs1 << ": " << llvm::format_hex(base, 10);
      llvm::errs() << ", M[x" << rs1;
      llvm::errs() << " + " << imm;
      llvm::errs() << "] -> " << llvm::format_hex(addr, 10);
      llvm::errs() << "\n";
      info.include(addr / kWordSize);
    } else {
      // store: S-type
      uint32_t rs1 = (inst >> 15) & 0x1f;
      uint32_t imm_low = (inst >> 7) & 0x1f;
      uint32_t imm_high = (inst >> 25) & 0x7f;
      int32_t imm = signExtend12((imm_high << 5) | imm_low);
      ;
      uint32_t base = loadU32(regOff + rs1);
      uint32_t addr = base + imm;
      llvm::errs() << "  store: " << llvm::format_hex(inst, 10);
      llvm::errs() << ", M[x" << rs1;
      llvm::errs() << " + " << imm;
      llvm::errs() << "] -> " << llvm::format_hex(addr, 10);
      llvm::errs() << "\n";
      info.include(addr / kWordSize, IncludeDir::Both);
    }
  } break;
  case MajorType::kECall: {
    uint32_t minor = loadU32(RegAddr::kT0);
    switch (minor) {
    case ECallType::kHalt: {
      llvm::errs() << "ecall/halt\n";
      uint32_t mode = loadU32(RegAddr::kA0);
      uint32_t addr = loadU32(RegAddr::kA1);
      for (size_t i = 0; i < kDigestWords; i++) {
        info.include((addr / kWordSize) + i);
      }
      if (mode == HaltType::kPause) {
        info.forceFlush = true;
      }
    } break;
    case ECallType::kSoftware: {
      llvm::errs() << "ecall/software\n";
      uint32_t addr = loadU32(RegAddr::kA0);
      uint32_t words = loadU32(RegAddr::kA1);
      for (size_t i = 0; i < words; i++) {
        info.include((addr / kWordSize) + i);
      }
    } break;
    case ECallType::kSha: {
      llvm::errs() << "ecall/sha\n";
      uint32_t stateOutAddr = loadU32(RegAddr::kA0);
      uint32_t stateInAddr = loadU32(RegAddr::kA1);
      uint32_t block1Addr = loadU32(RegAddr::kA2);
      uint32_t block2Addr = loadU32(RegAddr::kA3);
      uint32_t count = loadU32(RegAddr::kA4);
      llvm::errs() << "  stateOut: " << llvm::format_hex(stateOutAddr, 10)
                   << ", stateIn: " << llvm::format_hex(stateInAddr, 10)
                   << ", block1: " << llvm::format_hex(block1Addr, 10)
                   << ", block2: " << llvm::format_hex(block2Addr, 10) //
                   << ", count: " << count << "\n";
      for (size_t i = 0; i < kDigestWords; i++) {
        info.include(stateOutAddr / kWordSize + i);
      }
      for (size_t i = 0; i < kDigestWords; i++) {
        info.include(stateInAddr / kWordSize + i);
      }
      for (size_t i = 0; i < count; i++) {
        uint32_t addr1 = block1Addr / kWordSize + i * kBlockSize;
        uint32_t addr2 = block2Addr / kWordSize + i * kBlockSize;
        for (size_t j = 0; j < kDigestWords; j++) {
          info.include(addr1 + j);
          info.include(addr2 + j);
        }
      }
    } break;
    }
  } break;
  }

  return info;
}

std::vector<Polynomial> initCode(size_t maxCycles) {
  std::vector<uint64_t> code = writeCode(maxCycles);
  std::vector<Polynomial> image(code.size());
  for (size_t i = 0; i < image.size(); i++) {
    image[i] = {code[i]};
  }
  return image;
}

Runner::Runner(size_t maxCycles, std::map<uint32_t, uint32_t> elfImage, uint32_t entryPoint)
    : code(initCode(maxCycles))
    , cycles(code.size() / kCodeSize)
    , out(kInOutSize, Polynomial(1, kFieldInvalid))
    , data(kDataSize * cycles, Polynomial(1, kFieldInvalid))
    , mix(kMixSize, Polynomial(1, kFieldInvalid))
    , accum(kAccumSize * cycles, Polynomial(1, kFieldInvalid))
    , args{code, out, data, mix, accum} {
  generateCircuit();
  module.setExternHandler(this);

  // load the ELF image
  for (auto kvp : elfImage) {
    storeU32(kvp.first, kvp.second);
  }

  size_t nextOut = 0;

  // Initialize 'Input'
  Digest inputId = shaHash("Hello");
  for (size_t i = 0; i < kDigestWords; i++) {
    uint64_t word = inputId.words[i];
    for (size_t j = 0; j < kWordSize; j++) {
      out[nextOut++] = {(word >> (8 * j)) & 0xff};
    }
  }
  // Initialize PC
  for (size_t i = 0; i < kWordSize; i++) {
    out[nextOut++] = {(entryPoint >> (8 * i)) & 0xff};
  }
  // Initialize ImageID
  Digest imageId = initMemoryImage();
  for (size_t i = 0; i < kDigestWords; i++) {
    uint64_t word = imageId.words[i];
    for (size_t j = 0; j < kWordSize; j++) {
      out[nextOut++] = {(word >> (8 * j)) & 0xff};
    }
  }
}

void Runner::generateCircuit() {
  module.addFunc<5>(
      "riscv",
      {cbuf(kCodeSize), gbuf(kInOutSize), mbuf(kDataSize), gbuf(kMixSize), mbuf(kAccumSize)},
      [](Buffer code, Buffer out, Buffer data, Buffer mix, Buffer accum) {
        CompContext::init({"_ram_finalize",
                           "ram_verify",
                           "_bytes_finalize",
                           "bytes_verify",
                           "compute_accum",
                           "verify_accum"});

        CompContext::addBuffer("code", code);
        CompContext::addBuffer("out", out);
        CompContext::addBuffer("data", data);
        CompContext::addBuffer("mix", mix);
        CompContext::addBuffer("accum", accum);

        Top top;
        top->set();

        CompContext::fini();
      });
  module.optimize();

  if (module.computeMaxDegree("riscv") > kMaxDegree) {
    llvm::errs() << "Degree exceeded max degree " << kMaxDegree << "\n";
    module.dumpPoly("riscv");
    throw(std::runtime_error("Maximum degree exceeeded"));
  }

  // module.dump();
  // module.dumpPoly("riscv");
  // exit(1);

  module.optimize(5);
  // module.dumpStage(0);
}

Digest Runner::initMemoryImage() {
  // Populate the page table merkle tree
  llvm::errs() << "Compute page table MT\n";
  PageTableInfo info;

  std::vector<uint32_t> zeroPage(kPageSize / kWordSize);
  Digest zeroDigest = shaHash(zeroPage.data(), zeroPage.size());

  for (size_t i = 0; i < info.numPages; i++) {
    std::vector<uint32_t> page(kPageSize / kWordSize);
    bool isZeroPage = true;
    uint32_t pageAddr = getPageAddr(i);
    for (size_t j = 0; j < page.size(); j++) {
      uint32_t word = loadU32(pageAddr + j);
      if (word) {
        page[j] = word;
        isZeroPage = false;
      }
    }
    if (isZeroPage) {
      storePageEntry(i, zeroDigest);
    } else {
      storePageEntry(i, shaHash(page.data(), page.size()));
    }
  }

  // Compute root of merkle tree (ImageID)
  std::vector<uint32_t> page(roundUp(info.numRootEntries * kDigestWords, kBlockSize));
  for (size_t i = 0; i < page.size(); i++) {
    page[i] = loadU32(info.rootPageAddr + i);
  }
  Digest imageId = shaHash(page.data(), page.size());
  llvm::errs() << "ImageID: " //
               << llvm::format_hex(imageId.words[0], 10) << ", "
               << llvm::format_hex(imageId.words[1], 10) << ", "
               << llvm::format_hex(imageId.words[2], 10) << ", "
               << llvm::format_hex(imageId.words[3], 10) << ", "
               << llvm::format_hex(imageId.words[4], 10) << ", "
               << llvm::format_hex(imageId.words[5], 10) << ", "
               << llvm::format_hex(imageId.words[6], 10) << ", "
               << llvm::format_hex(imageId.words[7], 10) << "\n";

  return imageId;
}

void Runner::setInput(const std::vector<uint32_t>& input) {
  this->input = input;
}

void Runner::storePageEntry(uint32_t pageIndex, const Digest& digest) {
  for (size_t i = 0; i < kDigestWords; i++) {
    storeU32(kPageTableAddr + pageIndex * kDigestWords + i, digest.words[i]);
  }
}

void Runner::run() {
  runStage(0);
  sort("ram");
  runStage(1);
  sort("bytes");
  runStage(2);
  setMix();
  runStage(3);
  calcPrefixProducts(ExtensionField(kFieldPrimeDefault, kExtSize));
  runStage(4);
}

void Runner::runAgain() {
  llvm::errs() << "Run again> lastPc: " << llvm::format_hex(lastPc, 10) << "\n";

  isHalted = false;
  isFlushing = false;
  finishedPageReads.clear();
  dirtyPages.clear();
  residentWords.clear();

  // Copy ImageID from post state of recently completed image.
  std::vector<Polynomial> imageId;
  size_t offset = (2 * kDigestWords + 2) * kWordSize;
  for (size_t i = 0; i < kDigestWords; i++) {
    for (size_t j = 0; j < kWordSize; j++) {
      imageId.push_back(out[offset + i * kWordSize + j]);
    }
  }

  out.assign(out.size(), Polynomial(1, kFieldInvalid));
  data.assign(data.size(), Polynomial(1, kFieldInvalid));
  mix.assign(mix.size(), Polynomial(1, kFieldInvalid));
  accum.assign(accum.size(), Polynomial(1, kFieldInvalid));

  size_t nextOut = 0;

  // Initialize 'Input'
  Digest inputId = shaHash("Hello");
  for (size_t i = 0; i < kDigestWords; i++) {
    uint64_t word = inputId.words[i];
    for (size_t j = 0; j < kWordSize; j++) {
      out[nextOut++] = {(word >> (8 * j)) & 0xff};
    }
  }
  // Initialize PC
  for (size_t i = 0; i < kWordSize; i++) {
    out[nextOut++] = {(lastPc >> (8 * i)) & 0xff};
  }
  // Initialize ImageID
  for (size_t i = 0; i < kDigestWords; i++) {
    for (size_t j = 0; j < kWordSize; j++) {
      out[nextOut++] = imageId[i * kWordSize + j];
    }
  }

  run();
}

bool Runner::done() {
  module.setExternHandler(nullptr);
  code.clear();
  out.clear();
  data.clear();
  mix.clear();
  accum.clear();
  args.clear();
  return isHalted;
}

void Runner::runStage(size_t stage) {
  module.runStage(stage, "riscv", args, 0, cycles);
}

void Runner::setMix() {
  // Not cryptographic randomness, this runner is only for testing
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, kFieldPrimeDefault - 1);
  for (size_t i = 0; i < mix.size(); i++) {
    mix[i] = {static_cast<uint64_t>(distribution(generator))};
  }
}

} // namespace zirgen::rv32im_v1

// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include <deque>

#include "zirgen/circuit/rv32im/v2/emu/image.h"

namespace zirgen::rv32im_v2 {

struct HostIoHandler {
  // Called when the VM writes data to the host
  virtual uint32_t write(uint32_t fd, const uint8_t* data, uint32_t len) = 0;
  // Called when the VM reads data from the host
  virtual uint32_t read(uint32_t fd, uint8_t* data, uint32_t len) = 0;
};

// Implemention used for testing
struct TestIoHandler : public HostIoHandler {
  std::map<uint32_t, std::deque<uint8_t>> input;
  std::map<uint32_t, std::deque<uint8_t>> output;
  uint32_t write(uint32_t fd, const uint8_t* data, uint32_t len) override;
  uint32_t read(uint32_t fd, uint8_t* data, uint32_t len) override;
  void push_u32(uint32_t fd, uint32_t val);
  uint32_t pop_u32(uint32_t fd);
};

struct Segment {
  // Initial sparse memory state for the segment
  MemoryImage image;
  // Recorded host->guest IO, one entry per read
  std::vector<std::vector<uint8_t>> readRecord;
  // Recorded rlen of guest->host IO, one entry per write
  std::vector<uint32_t> writeRecord;
  // The 'input' digest
  Digest input;
  // Is this the terminating segment
  bool isTerminate;
  // Cycle at which we suspend
  size_t suspendCycle;
  // Total physical cycles
  size_t pagingCycles;
};

// Run the executor and returns a set of segments. The memory image passed in
// is updated in place.
std::vector<Segment> execute(MemoryImage& in,
                             HostIoHandler& io,
                             size_t segmentSize,
                             size_t maxCycles,
                             Digest input = Digest::zero());

} // namespace zirgen::rv32im_v2

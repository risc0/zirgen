const RHO: [u32; 24] = [
  1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27,
  41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
];
const PI: [usize; 24] = [
  10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15,
  23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
];
const WORDS: usize = 25;
const EWORDS: usize = 25+3;
const ROUNDS: usize = 24;
const ROUNDLEN: u64 = 111*180;

pub const RC: [u64; ROUNDS] = [
  1u64,
  0x8082u64,
  0x800000000000808au64,
  0x8000000080008000u64,
  0x808bu64,
  0x80000001u64,
  0x8000000080008081u64,
  0x8000000000008009u64,
  0x8au64,
  0x88u64,
  0x80008009u64,
  0x8000000au64,
  0x8000808bu64,
  0x800000000000008bu64,
  0x8000000000008089u64,
  0x8000000000008003u64,
  0x8000000000008002u64,
  0x8000000000000080u64,
  0x800au64,
  0x800000008000000au64,
  0x8000000080008081u64,
  0x8000000000008080u64,
  0x80000001u64,
  0x8000000080008008u64,
];

#[allow(unused_assignments)]
pub fn keccakf(all: &mut [[u64; EWORDS]], offset: usize) {
  use crunchy::unroll;

  let mut a: [u64; EWORDS] = [0; EWORDS];
  a.clone_from_slice(&all[offset]);

  for i in 0..ROUNDS {
  let mut array: [u64; 5] = [0; 5];

  // Theta
  unroll! {
    for x in 0..5 {
    unroll! {
      for y_count in 0..5 {
      let y = y_count * 5;
      array[x] ^= a[x + y];
      }
    }
    }
  }

  unroll! {
    for x in 0..5 {
    unroll! {
      for y_count in 0..5 {
      let y = y_count * 5;
      a[y + x] ^= array[(x + 4) % 5] ^ array[(x + 1) % 5].rotate_left(1);
      }
    }
    }
  }

  // Rho and pi
  let mut last = a[1];
  unroll! {
    for x in 0..24 {
    array[0] = a[PI[x]];
    a[PI[x]] = last.rotate_left(RHO[x]);
    last = array[0];
    }
  }

  // Chi
  unroll! {
    for y_step in 0..5 {
    let y = y_step * 5;

    unroll! {
      for x in 0..5 {
      array[x] = a[y + x];
      }
    }

    unroll! {
      for x in 0..5 {
      a[y + x] = array[x] ^ ((!array[(x + 1) % 5]) & (array[(x + 2) % 5]));
      }
    }
    }
  };

  // Iota
  a[0] ^= RC[i];
  a[WORDS] += 1;
  a[WORDS+1] += ROUNDLEN;
  all[offset+i+1].clone_from_slice(&a);
  }
}

const WORDS: usize = 25;
const EWORDS: usize = 25+3;
const ROUNDLEN: u64 = 68*180;

pub const KRC: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub const INIT: [u32; 8] = [
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

pub fn sha256f(all: &mut [[u64; EWORDS]], idx: usize) {
    let mut w = [0u32; 64];
    for i in 0..8 {
      let curr = all[idx][i];
      w[2*i] = (curr as u32).swap_bytes();
      w[2*i+1] = ((curr>>32) as u32).swap_bytes();
    }
    for i in 16..64 {
        let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^
                 (w[i-15] >> 3);
        let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^
                 (w[i-2] >> 10);
        w[i] = w[i - 16].wrapping_add(s0)
                        .wrapping_add(w[i-7])
                        .wrapping_add(s1);
    }

    let mut aa: [u32; 8] = [all[idx][8+0] as u32, all[idx][8+1] as u32,
                            all[idx][8+2] as u32, all[idx][8+3] as u32,
                            all[idx][8+4] as u32, all[idx][8+5] as u32,
                            all[idx][8+6] as u32, all[idx][8+7] as u32];
    for i in 0..64 {
        let ch = (aa[4] & aa[5]) ^ (!aa[4] & aa[6]);
        let ma = (aa[0] & aa[1]) ^ (aa[0] & aa[2]) ^ (aa[1] & aa[2]);
        let s0 = aa[0].rotate_right(2) ^ aa[0].rotate_right(13) ^
                 aa[0].rotate_right(22);
        let s1 = aa[4].rotate_right(6) ^ aa[4].rotate_right(11) ^
                 aa[4].rotate_right(25);
        let t0 = aa[7].wrapping_add(s1)
                      .wrapping_add(ch)
                      .wrapping_add(KRC[i])
                      .wrapping_add(w[i]);
        let t1 = s0.wrapping_add(ma);

        aa[7] = aa[6];
        aa[6] = aa[5];
        aa[5] = aa[4];
        aa[4] = aa[3].wrapping_add(t0);
        aa[3] = aa[2];
        aa[2] = aa[1];
        aa[1] = aa[0];
        aa[0] = t0.wrapping_add(t1);
    }

    for i in 0..8 {
      let w = all[idx][8+i] as u32;
      all[idx+1][8+i] = (w.wrapping_add(aa[i])) as u64;
    }
    all[idx+1][WORDS+1] += ROUNDLEN;
}

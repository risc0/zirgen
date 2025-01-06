//arr_a: 0-21
//arr_b: 21-43
//arr_c: 44-65
//arr_d: 66-89
//arr_e: 90-111
//arr_f: 112-133
//minor=134, major=135, rnd: 136, blk: 137, midx=138
//auxr: 139-154
//major_arr: 155-176
//pc: 177, invpc: 178
//count: 179

mod tkeccak;
mod tsha256;

type K64 = u64; //keccak 64bit word
type B32 = u32; //babybear 31bit field element
const MAJLEN: usize = 22;
const AUXLEN: usize = 16;
const AUXBIN: usize = 11;
const SLEN: usize = 21;
const BLEN: usize = 22;
const RLEN: usize = 24;
const PERMROWS: usize = 111;
const COLS: usize = 180;
const ROUNDS: usize = 24;
const KINPWORDS: usize = 17;
const KWORDS: usize = 25;
const EWORDS: usize = 25+3;
const MIDXBITS: usize = 29;
const ROUNDBITS: usize = 64 - MIDXBITS*2;
const SHA2ROWS: usize = 68;
const BABYBEAR_MODULUS: u64 = 15 * (1 << 27) + 1;

mod op {
  pub const SKIP: super::K64 = 0;
  pub const PERMUTE: super::K64 = 1;
  pub const ABSORB: super::K64 = 2;
  pub const ENDK: super::K64 = 3;
  pub const MEMZPAD: super::K64 = 4;
  pub const SHA2BLK: super::K64 = 5;
}

const ROTOFF: &'static [u32] = &[0, 1, 62, 28, 27,
                                 36, 44, 6, 55, 20,
                                 3, 10, 43, 25, 39,
                                 41, 45, 15, 21, 8,
                                 18, 2, 61, 56, 14];

fn inv_babybear(x: B32) -> B32 { // inefficient, please replace!
  if 0 == x { return 0; };
  use num_bigint::BigUint;
  let m = BigUint::from((15 * (1 << 27) + 1) as u32);
  let y = BigUint::from(x);
  let z = y.modinv(&m).unwrap();
  z.to_u32_digits().first().unwrap().clone()
}

fn mult_babybear (x: B32, y: B32) -> B32 { //branch on prod==0 slower?
  let bx = x as u64;
  let by = y as u64;
  let product = bx.wrapping_mul(by);
  (product % BABYBEAR_MODULUS) as B32
}

macro_rules! zcopy {
    ($arr: expr, $idx: expr) => {
        #[cfg(feature="nozvec")]
        { $arr[$idx] = 0; }
    }
}

fn write_pc(vals: &mut[B32], pc: B32) {
  let j = (pc as usize)*COLS + 6*BLEN+2+5+AUXLEN+MAJLEN;
  vals[j] = pc;
  vals[j+1] = inv_babybear(pc);
}

#[cfg(not(feature="nozvec"))]
fn write_dummy(vals: &mut[B32], pc: B32) {
  let idx = (pc as usize)*COLS + 6*BLEN+2;
  vals[idx] = 9;
  vals[idx+1] = 16;
  vals[idx+14] = 1;
  vals[idx+37] = 1;
}

#[cfg(feature="nozvec")]
fn write_dummy(vals: &mut[B32], pc: B32) {
  let mut idx = (pc as usize)*COLS;
  for _ in 0..6*BLEN+2 {vals[idx] = 0; idx += 1; }
  vals[idx] = 9; idx += 1;
  vals[idx] = 16; idx += 1;
  for _ in 0..3+9 {vals[idx] = 0; idx += 1; }
  vals[idx] = 1; idx += 1;
  for _ in 0..7+15 {vals[idx] = 0; idx += 1; }
  vals[idx] = 1; idx += 1;
  for _ in 0..5 {vals[idx] = 0; idx += 1; }
  idx += 2;
  vals[idx] = 0;
}

#[inline]
fn onehot<const L: usize>(vals: &mut[B32],
                          idx: &mut usize, v: usize) {
  let end = *idx+L;
  #[cfg(feature="nozvec")]
  for j in *idx..end {
    vals[j] = 0;
  }
  vals[*idx+v] = 1;
  *idx = end; 
}

#[inline]
fn zloop(_vals: &mut[B32], idx: &mut usize, len: usize) {
    #[cfg(feature="nozvec")]
    for _ in 0..len {
      _vals[*idx] = 0; *idx += 1;
    }
    #[cfg(not(feature="nozvec"))]
    { *idx += len; }
}

#[inline]
fn unpack32<const L: usize>(vals: &mut[B32],
                            idx: &mut usize, v: B32) {
  for j in 0..L {
    vals[*idx] = (1 as B32) & (v >> j);
    *idx += 1;
  }
}

#[inline]
fn unpack64<const S: usize, const L: usize>(vals: &mut[B32],
                                            idx: &mut usize, v: K64) {
  for j in S..L+S {
    vals[*idx] = ((1 as K64) & (v >> j)) as B32;
    *idx += 1;
  }
}

#[inline]
fn unpack_low(vals: &mut[B32], idx: &mut usize, v: K64) {
  unpack64::<0,BLEN>(vals, idx, v);
}

#[inline]
fn unpack_mid(vals: &mut[B32], idx: &mut usize, v: K64) {
  unpack64::<BLEN,SLEN>(vals, idx, v);
}

#[inline]
fn unpack_high(vals: &mut[B32], idx: &mut usize, v: K64) {
  const SUMLEN: usize = BLEN+SLEN;
  unpack64::<SUMLEN,SLEN>(vals, idx, v);
}

#[inline]
fn expand_st(vals: &mut[B32], idx: &mut usize,
             st: &[K64], start: usize) {
  for j in start..KWORDS {
    let w = st[j];
    vals[*idx] = (w & ((1 << BLEN)-1)) as B32; *idx +=1;
    vals[*idx] = ((w >> BLEN) & ((1 << SLEN)-1)) as B32; *idx +=1;
    vals[*idx] = (w >> (BLEN+SLEN)) as B32; *idx +=1;
  }
}

#[inline]
fn expand_counters(vals: &mut[B32], idx: &mut usize, last: K64) {
  //round=6bits, block=29bits, midx=29bits
  let rnd_counter = (last as B32) & ((1<<ROUNDBITS)-1);
  let blk_counter = ((last >> ROUNDBITS) as B32) & ((1<<MIDXBITS)-1);
  let midx_counter = (last >> (ROUNDBITS+MIDXBITS)) as B32;
  vals[*idx] = rnd_counter; *idx += 1;
  vals[*idx] = blk_counter; *idx += 1;
  vals[*idx] = midx_counter; *idx += 1;
}

#[inline]
fn inject_counters(vals: &mut[B32], idx: &mut usize) {
  vals[*idx] = vals[*idx-COLS]; *idx += 1;
  vals[*idx] = vals[*idx-COLS]; *idx += 1;
  vals[*idx] = vals[*idx-COLS]; *idx += 1;
}

//#[inline]
fn xor5minor0(xidx: usize, vals: &mut[B32], idx: &mut usize,
              st: &[K64], xor5r_low: B32) {
  unpack_low(vals, idx, st[xidx+5*0]);
  unpack_low(vals, idx, st[xidx+5*1]);
  unpack_low(vals, idx, st[xidx+5*2]);
  unpack_low(vals, idx, st[xidx+5*3]);
  vals[*idx] = xor5r_low; *idx +=1;
  zcopy!(vals, *idx); *idx +=1;
  unpack_low(vals, idx, st[xidx+5*4]);
  unpack32::<BLEN>(vals, idx, xor5r_low);
  vals[*idx] = xidx as B32; *idx +=1; //minor
  vals[*idx] = 1; *idx +=1;
  expand_counters(vals, idx, st[KWORDS]);
  onehot::<AUXLEN>(vals, idx, xidx);
  onehot::<MAJLEN>(vals, idx, 0);
  *idx += 2; //skip pc,invpc
  zcopy!(vals, *idx); *idx +=1; //count of memarg
}

//#[inline]
fn xor5minor1(xidx: usize, vals: &mut[B32], idx: &mut usize,
              st: &[K64], x5v: &[K64]) {
  let xor5r_low = x5v[1+4*xidx] as B32;
  let xor5r_mid = x5v[2+4*xidx] as B32;
  unpack_mid(vals, idx, st[xidx+5*0]);
  vals[*idx] = xor5r_low; *idx += 1;
  unpack_mid(vals, idx, st[xidx+5*1]);
  vals[*idx] = xor5r_mid; *idx += 1;
  unpack_mid(vals, idx, st[xidx+5*2]);
  zcopy!(vals, *idx); *idx +=1;
  unpack_mid(vals, idx, st[xidx+5*3]);
  zloop(vals, idx, 3);
  unpack_mid(vals, idx, st[xidx+5*4]);
  zcopy!(vals, *idx); *idx +=1;
  unpack32::<BLEN>(vals, idx, xor5r_mid);
  vals[*idx] = xidx as B32; *idx +=1;
  vals[*idx] = 2; *idx +=1;
  inject_counters(vals, idx);
  onehot::<AUXLEN>(vals, idx, xidx);
  onehot::<MAJLEN>(vals, idx, 1);
  *idx += 2;
  zcopy!(vals, *idx); *idx +=1;
}

//#[inline]
fn xor5minor2(xidx: usize, vals: &mut[B32], idx: &mut usize,
              st: &[K64], x5v: &[K64]) {
  let xor5r_low  = x5v[1+4*xidx] as B32;
  let xor5r_mid  = x5v[2+4*xidx] as B32;
  let xor5r_high = x5v[3+4*xidx] as B32;
  unpack_high(vals, idx, st[xidx+5*0]);
  vals[*idx] = xor5r_low; *idx += 1;
  unpack_high(vals, idx, st[xidx+5*1]);
  vals[*idx] = xor5r_mid; *idx += 1;
  unpack_high(vals, idx, st[xidx+5*2]);
  vals[*idx] = xor5r_high; *idx += 1;
  unpack_high(vals, idx, st[xidx+5*3]);
  zloop(vals, idx, 3);
  unpack_high(vals, idx, st[xidx+5*4]);
  zcopy!(vals, *idx); *idx +=1;
  unpack32::<BLEN>(vals, idx, xor5r_high);
  vals[*idx] = xidx as B32; *idx +=1;
  vals[*idx] = 3; *idx +=1;
  inject_counters(vals, idx);
  onehot::<AUXLEN>(vals, idx, xidx);
  onehot::<MAJLEN>(vals, idx, 2);
  *idx += 2;
  zcopy!(vals, *idx); *idx +=1;
}

//#[inline]
fn xor5minor3(xidx: usize, vals: &mut[B32], idx: &mut usize,
              st: &[K64], x5v: &[K64]) {
  expand_st(vals, idx, st, 0);
  for j in 0..(xidx+1) {
    vals[*idx] = x5v[4*j + 1] as B32; *idx+=1;
    vals[*idx] = x5v[4*j + 2] as B32; *idx+=1;
    vals[*idx] = x5v[4*j + 3] as B32; *idx+=1;
  }
  zloop(vals, idx, 3*(4-xidx) + 2*BLEN); 
  if xidx < 4 {
    vals[*idx] = (xidx+1) as B32; *idx +=1;
    zcopy!(vals, *idx); *idx +=1;
  } else {
    zcopy!(vals, *idx); *idx +=1;
    vals[*idx] = 4; *idx +=1;
  }
  inject_counters(vals, idx);
  onehot::<AUXLEN>(vals, idx, xidx);
  onehot::<MAJLEN>(vals, idx, 3);
  *idx += 2;
  zcopy!(vals, *idx); *idx +=1;
}

//#[inline]
fn xor5major(xidx: usize, vals: &mut[B32], idx: &mut usize,
             st: &[K64], x5v: &[K64]) {
  xor5minor0(xidx, vals, idx, st, x5v[4*xidx + 1] as B32);
  xor5minor1(xidx, vals, idx, st, x5v);
  xor5minor2(xidx, vals, idx, st, x5v);
  xor5minor3(xidx, vals, idx, st, x5v);
}

//#[inline]
fn rho_minor012(xidx: usize, yidx: usize, vals: &mut[B32], idx: &mut usize,
                st: &[K64], x5v: &[K64]) {
  let xinc_idx  = (xidx+1)%5;
  let xinc_val  = x5v[4*xinc_idx].rotate_left(1);

  let xdec_idx  = (xidx+5-1)%5;
  let xdec_val  = x5v[4*xdec_idx];

  unpack_low(vals, idx, xinc_val);
  unpack_mid(vals, idx, xinc_val); zcopy!(vals, *idx); *idx += 1;
  unpack_high(vals, idx, xinc_val); zcopy!(vals, *idx); *idx += 1;
  unpack_low(vals, idx, xdec_val);
  zcopy!(vals, *idx); *idx +=1;
  zcopy!(vals, *idx); *idx +=1;
  unpack_mid(vals, idx, xdec_val); zcopy!(vals, *idx); *idx +=1;
  unpack_high(vals, idx, xdec_val); zcopy!(vals, *idx); *idx +=1;

  let mut curr_minor = 3*(xidx/2);
  vals[*idx] = (curr_minor+1) as B32; *idx +=1;
  vals[*idx] = (yidx+4) as B32; *idx +=1;
  inject_counters(vals, idx);
  onehot::<AUXLEN>(vals, idx, curr_minor);
  onehot::<MAJLEN>(vals, idx, 4+yidx);
  *idx += 2;
  zcopy!(vals, *idx); *idx +=1;

  curr_minor += 1; //rho_minor1
  let xnxtdec_val = x5v[0 + 4*xidx /* +1 -1 */];
  if 4 == xidx {
    zloop(vals, idx, 3*BLEN);
  } else {
    unpack_low(vals, idx, xnxtdec_val);
    unpack_mid(vals, idx, xnxtdec_val); zcopy!(vals, *idx); *idx += 1;
    unpack_high(vals, idx, xnxtdec_val); zcopy!(vals, *idx); *idx += 1;
  }
  let w = st[xidx+5*yidx];
  let xres = w ^ xinc_val ^ xdec_val;
  let result = xres.rotate_left(ROTOFF[xidx + 5*yidx]);
  let res_low  = (result & ((1 << BLEN)-1)) as B32;
  let res_mid  = ((result >> BLEN) & ((1 << SLEN)-1)) as B32;
  let res_high = (result >> (BLEN+SLEN)) as B32;
  unpack_high(vals, idx, w);
  vals[*idx] = res_low;  *idx += 1;
  vals[*idx] = res_mid;  *idx += 1;
  vals[*idx] = res_high; *idx += 1;
  unpack_mid(vals, idx, w); zcopy!(vals, *idx); *idx += 1;
  unpack_low(vals, idx, w);
  vals[*idx] = (curr_minor+1) as B32; *idx += 1;
  vals[*idx] = (yidx+4) as B32; *idx += 1;
  inject_counters(vals, idx);
  onehot::<AUXLEN>(vals, idx, curr_minor);
  onehot::<MAJLEN>(vals, idx, 4+yidx);
  *idx += 2;
  zcopy!(vals, *idx); *idx +=1;
  if 4 == xidx { return; } //skip rho_minor2 ?
  curr_minor += 1;
  let xnxtinc_idx  = (xidx+2)%5;
  let xnxtinc_val  = x5v[4*xnxtinc_idx];
  let xnxtinc_rotv = xnxtinc_val.rotate_left(1);
  unpack_low(vals, idx, xnxtinc_val);
  unpack_mid(vals, idx, xnxtinc_val); zcopy!(vals, *idx); *idx += 1;
  unpack_high(vals, idx, xnxtinc_val); zcopy!(vals, *idx); *idx += 1;
  let wnxt = st[xidx+1+5*yidx];
  let xnxtres = wnxt ^ xnxtinc_rotv ^ xnxtdec_val;
  let nxtresult = xnxtres.rotate_left(ROTOFF[xidx+1 + 5*yidx]);
  let nxtres_low  = (nxtresult & ((1 << BLEN)-1)) as B32;
  let nxtres_mid  = ((nxtresult >> BLEN) & ((1 << SLEN)-1)) as B32;
  let nxtres_high = (nxtresult >> (BLEN+SLEN)) as B32;
  unpack_high(vals, idx, wnxt);
  vals[*idx] = nxtres_low;  *idx += 1;
  vals[*idx] = nxtres_mid;  *idx += 1;
  vals[*idx] = nxtres_high; *idx += 1;
  unpack_mid(vals, idx, wnxt); zcopy!(vals, *idx); *idx += 1;
  unpack_low(vals, idx, wnxt);
  vals[*idx] = (curr_minor+1) as B32; *idx += 1;
  vals[*idx] = (yidx+4) as B32; *idx += 1;
  inject_counters(vals, idx);
  onehot::<AUXLEN>(vals, idx, curr_minor);
  onehot::<MAJLEN>(vals, idx, 4+yidx);
  *idx += 2;
  zcopy!(vals, *idx); *idx +=1;
}

//#[inline]
fn rho_minor3(yidx: usize, vals: &mut[B32], idx: &mut usize) {
  let pre = 15*yidx;
  for _ in 0..pre {
    vals[*idx] = vals[*idx - COLS*9]; *idx += 1;
  }
  for _ in 0..3 {
    vals[*idx] = vals[*idx - pre - COLS*7 + BLEN*3 + SLEN]; *idx += 1;
  }
  for _ in 0..3 {
    vals[*idx] = vals[*idx - pre - COLS*6 + BLEN*3 + SLEN - 3]; *idx += 1;
  }
  for _ in 0..3 {
    vals[*idx] = vals[*idx - pre - COLS*4 + BLEN*3 + SLEN - 6]; *idx += 1;
  }
  for _ in 0..3 {
    vals[*idx] = vals[*idx - pre - COLS*3 + BLEN*3 + SLEN - 9]; *idx += 1;
  }
  for _ in 0..3 {
    vals[*idx] = vals[*idx - pre - COLS*1 + BLEN*3 + SLEN - 12]; *idx += 1;
  }
  let z = if yidx < 4 {0} else {15};
  for _ in 0..9+BLEN*3-pre-z {
    vals[*idx] = vals[*idx - COLS*9]; *idx += 1;
  }
  zloop(vals, idx, 1+BLEN*2+z); // includes minor
  vals[*idx] = 5+yidx as B32; *idx += 1;
  inject_counters(vals, idx);
  onehot::<AUXLEN>(vals, idx, 8);
  onehot::<MAJLEN>(vals, idx, 4+yidx);
  *idx += 2;
  zcopy!(vals, *idx); *idx +=1;
}

fn pi_inplace(vals: &mut[B32], start: usize) {
  let mut bak: [B32; KWORDS*3] = [0; KWORDS*3];
  bak.copy_from_slice(&vals[start..start+KWORDS*3]);

  for y in 0..5 {
    for x in 0..5 {
      let u = (0 * x + 1 * y) % 5;
      let v = (2 * x + 3 * y) % 5;
      let uv = 3*(v*5 + u);
      let xy = 3*(y*5 + x);
      vals[start+0 + uv] = bak[0 + xy];
      vals[start+1 + uv] = bak[1 + xy];
      vals[start+2 + uv] = bak[2 + xy];
    }
  }
}

//#[inline]
fn chi_minor0(xidx: usize, yidx: usize, vals: &mut[B32], idx: &mut usize) {
  let minor = if (xidx==1) || (xidx==3)
    { 3*(1+(xidx-1)/2) } 
  else
    { 1+3*(xidx/2) };

  let start = *idx - COLS*minor;

  let xinc_idx = start + 3*((xidx+1)%5 + yidx*5);
  let xinc_low = vals[xinc_idx];
  let xinc_mid = vals[xinc_idx+1];
  let xnxtinc_idx = start + 3*((xidx+2)%5 + yidx*5);
  let xnxtinc_low = vals[xnxtinc_idx];
  let xnxtinc_mid = vals[xnxtinc_idx+1];
  let xnon_idx = start + 3*(xidx + yidx*5);
  let xnon_low = vals[xnon_idx];
  let xnon_mid = vals[xnon_idx+1];
  let res_low = xnon_low ^ (!xinc_low & xnxtinc_low);
  let res_mid = xnon_mid ^ (!xinc_mid & xnxtinc_mid);

  unpack32::<BLEN>(vals, idx, xinc_mid);
  unpack32::<BLEN>(vals, idx, xnxtinc_low);
  unpack32::<BLEN>(vals, idx, xnxtinc_mid);
  unpack32::<BLEN>(vals, idx, xinc_low);
  vals[*idx] = res_low; *idx += 1;
  vals[*idx] = res_mid; *idx += 1;
  unpack32::<BLEN>(vals, idx, xnon_low);
  unpack32::<BLEN>(vals, idx, xnon_mid);

  let major = 9+yidx;
  vals[*idx] = minor as B32; *idx += 1;
  vals[*idx] = major as B32; *idx += 1;
  inject_counters(vals, idx);
  onehot::<AUXLEN>(vals, idx, minor-1);
  onehot::<MAJLEN>(vals, idx, major);
  *idx += 2;
  zcopy!(vals, *idx); *idx +=1;
}

//#[inline]
fn chi_minor1(xidx: usize, yidx: usize, vals: &mut[B32], idx: &mut usize) {
  let minor = 2 + 3*(xidx/2);

  let start = *idx - COLS*minor;

  let xp1 = (xidx+1)%5;
  let xinc_idx = start + 3*(xp1 + yidx*5)+2;
  let xinc_high = vals[xinc_idx];
  let xp2 = (xp1+1)%5;
  let xnxtinc_idx = start + 3*(xp2 + yidx*5)+2;
  let xnxtinc_high = vals[xnxtinc_idx];
  let xnon_idx = start + 3*(xidx + yidx*5)+2;
  let xnon_high = vals[xnon_idx];
  let res_high = xnon_high ^ (!xinc_high & xnxtinc_high);
  
  unpack32::<SLEN>(vals, idx, xnon_high);
  vals[*idx] = vals[*idx-COLS+1+BLEN*3]; *idx += 1;
  unpack32::<SLEN>(vals, idx, xinc_high);
  vals[*idx] = vals[*idx-COLS+1+1+BLEN*2]; *idx += 1;
  unpack32::<SLEN>(vals, idx, xnxtinc_high);
  vals[*idx] = res_high; *idx += 1;

  if xidx == 4 {
    zloop(vals, idx, 2+3*BLEN);
  } else {
    let xnninc_idx = start + 3*((xp2+1)%5 + yidx*5)+2;
    let xnninc_high = vals[xnninc_idx];
    let resnn_high = xinc_high ^ (!xnxtinc_high & xnninc_high);
    unpack32::<SLEN>(vals, idx, xnninc_high);
    vals[*idx] = resnn_high; zloop(vals, idx, 3+2*BLEN);
  }
  let major = 9+yidx;
  vals[*idx] = minor as B32; *idx += 1;
  vals[*idx] = major as B32; *idx += 1;
  inject_counters(vals, idx);
  onehot::<AUXLEN>(vals, idx, minor-1);
  onehot::<MAJLEN>(vals, idx, major);
  *idx += 2;
  zcopy!(vals, *idx); *idx +=1;
}

//#[inline]
fn chi_minor2(yidx: usize, vals: &mut[B32], idx: &mut usize) {
  let pre = 15*yidx;
  for _ in 0..pre {
    vals[*idx] = vals[*idx - COLS*9]; *idx += 1;
  }

  for j in 0..3 {
    vals[*idx] = vals[*idx-j - pre - COLS*7 + j*BLEN + SLEN]; *idx += 1;
  }
  vals[*idx] = vals[*idx-3 - pre - COLS*6 + BLEN*4]; *idx += 1;
  vals[*idx] = vals[*idx-4 - pre - COLS*6 + BLEN*4 + 1]; *idx += 1;
  vals[*idx] = vals[*idx-5 - pre - COLS*7 + BLEN*4 - 1]; *idx += 1;
  for j in 0..3 {
    vals[*idx] = vals[*idx-6-j - pre - COLS*4 + j*BLEN + SLEN]; *idx += 1;
  }
  vals[*idx] = vals[*idx-9 - pre - COLS*3 + BLEN*4]; *idx += 1;
  vals[*idx] = vals[*idx-10 - pre - COLS*3 + BLEN*4 + 1]; *idx += 1;
  vals[*idx] = vals[*idx-11 - pre - COLS*4 + BLEN*4 - 1]; *idx += 1;
  for j in 0..3 {
    vals[*idx] = vals[*idx-12-j - pre - COLS*1 + j*BLEN + SLEN]; *idx += 1;
  }

  if yidx < 4 {
    for _ in 0..9+BLEN*3-pre {
      vals[*idx] = vals[*idx - COLS*9]; *idx += 1;
    }
    zloop(vals, idx, 1+BLEN*2); // includes minor
    vals[*idx] = 10+yidx as B32; *idx += 1;
  } else {
    zloop(vals, idx, 13);
    // onehot(rnd) and st[a3]
    onehot::<RLEN>(vals, idx, vals[*idx-COLS+2+2*BLEN+2] as usize);
    unpack32::<BLEN>(vals, idx, vals[*idx - 9*COLS - 5*BLEN/*-2+2*/]);
    vals[*idx] = 9; *idx += 1;
    vals[*idx] = 13 as B32; *idx += 1;
  }
  inject_counters(vals, idx);
  onehot::<AUXLEN>(vals, idx, 8);
  let major = 9+yidx;
  onehot::<MAJLEN>(vals, idx, major);
  *idx += 2;
  zcopy!(vals, *idx); *idx +=1;
}

//#[inline]
fn iota_rc(vals: &mut[B32], idx: &mut usize) {
  let r = vals[*idx - COLS + 6*BLEN + 2 + 2] as B32;
  let w = tkeccak::RC[r as usize];
  let w_low = (w & ((1 << BLEN)-1)) as B32;
  let w_mid = ((w >> BLEN) & ((1 << SLEN)-1)) as B32;
  let w_high = (w >> (BLEN+SLEN)) as B32;

  let a000 = vals[*idx - COLS];
  let r_low = a000 ^ w_low;
  vals[*idx] = r_low; *idx +=1;
  let a001 = vals[*idx - COLS];
  let r_mid = a001 ^ w_mid;
  vals[*idx] = r_mid; *idx +=1;
  let r_high = vals[*idx - COLS] ^ w_high;
  vals[*idx] = r_high; *idx +=1;

  let r_full = (r_low as u64) |
               ((r_mid as u64)<<BLEN) |
               ((r_high as u64)<<(BLEN+SLEN));
  let write1 = (r_full & ((1<<16)-1)) as B32;
  let write2 = ((r_full >> 16) & ((1<<16)-1)) as B32;
  let write3 = ((r_full >> 32) & ((1<<16)-1)) as B32;
  let write4 = (r_full >> 48) as B32;

  for _ in 0..3*BLEN-3+9 {
    vals[*idx] = vals[*idx - COLS]; *idx += 1;
  }
  zloop(vals, idx, RLEN-9-2);
  let blk = vals[*idx-COLS+2+2*BLEN+3];
  let midx = vals[*idx-COLS+2+2*BLEN+4];
  let midxnxt = midx + 1;
  let newblk = blk.wrapping_sub(1); //cannot overlow
  let inv_newblk = inv_babybear(newblk);
  vals[*idx] = inv_newblk; *idx += 1;
  vals[*idx] = mult_babybear(newblk, inv_newblk); *idx += 1; 

  unpack32::<BLEN>(vals, idx, a000);
  unpack32::<BLEN>(vals, idx, a001);
  zcopy!(vals, *idx); *idx +=1;
  let mut count: B32 = 0;
  if r<(ROUNDS as B32)-1 {
    zcopy!(vals, *idx); *idx +=1;
    vals[*idx] = r+1; *idx += 1;
    vals[*idx] = blk; *idx += 1;
    vals[*idx] = midx; *idx += 1;
  } else {
    if 0 == newblk {
      vals[*idx] = 17; *idx += 1;
      zcopy!(vals, *idx); *idx +=1;
      zcopy!(vals, *idx); *idx +=1;
      vals[*idx] = midxnxt; *idx += 1;
      count = 1;
    } else {
      vals[*idx] = 14; *idx += 1;
      zcopy!(vals, *idx); *idx +=1;
      vals[*idx] = newblk; *idx += 1;
      vals[*idx] = midx; *idx += 1;
    }
  }
  onehot::<AUXBIN>(vals, idx, 9);
  vals[*idx] = midxnxt; *idx += 1;
  vals[*idx] = write1; *idx += 1;
  vals[*idx] = write2; *idx += 1;
  vals[*idx] = write3; *idx += 1;
  vals[*idx] = write4; *idx += 1;
  onehot::<MAJLEN>(vals, idx, 13);
  *idx += 2;
  vals[*idx] = count; *idx += 1;
}

fn perm_round(vals: &mut[B32], st: &[K64]) {
  let mut idx = st[KWORDS+1] as usize; 
  let mut x5v: [K64; 20] = [0; 20];
  for xidx in 0..5 {
    let xor5r = st[xidx+5*0] ^ st[xidx+5*1] ^
                st[xidx+5*2] ^ st[xidx+5*3] ^ st[xidx+5*4];
    x5v[0+4*xidx] = xor5r;
    x5v[1+4*xidx] = xor5r & ((1 << BLEN)-1);
    x5v[2+4*xidx] = (xor5r >> BLEN) & ((1 << SLEN)-1);
    x5v[3+4*xidx] = xor5r >> (BLEN+SLEN);
  }
  for xidx in 0..5 {
    xor5major(xidx, vals, &mut idx, st, &x5v);
  }
  for yidx in 0..5 {
    rho_minor012(0, yidx, vals, &mut idx, st, &x5v);
    rho_minor012(2, yidx, vals, &mut idx, st, &x5v);
    rho_minor012(4, yidx, vals, &mut idx, st, &x5v);
    rho_minor3(yidx, vals, &mut idx);
  }
  pi_inplace(vals, idx-COLS);

  for yidx in 0..5 {
    chi_minor0(0, yidx, vals, &mut idx);
    chi_minor1(0, yidx, vals, &mut idx);
    chi_minor0(1, yidx, vals, &mut idx);
    chi_minor0(2, yidx, vals, &mut idx);
    chi_minor1(2, yidx, vals, &mut idx);
    chi_minor0(3, yidx, vals, &mut idx);
    chi_minor0(4, yidx, vals, &mut idx);
    chi_minor1(4, yidx, vals, &mut idx);
    chi_minor2(yidx, vals, &mut idx);
  }
  iota_rc(vals, &mut idx);
}

//#[inline]
fn absorb_word(xyidx: usize, blk: B32, midx: B32,
               vals: &mut[B32], idx: &mut usize,
               prev_st: &[K64], curr_st: &[K64]) {
  let w = prev_st[xyidx];
  let inp = curr_st[xyidx];
  unpack_low(vals, idx, w);
  unpack_low(vals, idx, inp);
  unpack_mid(vals, idx, inp);
  zcopy!(vals, *idx); *idx +=1;
  let new_midx = midx + 1 + (xyidx as B32);
  unpack_high(vals, idx, inp);
  let res = w ^ inp;
  vals[*idx] = (res & ((1 << BLEN)-1)) as B32; *idx += 1;
  vals[*idx] = ((res >> BLEN) & ((1 << SLEN)-1)) as B32; *idx +=1;
  vals[*idx] = (res >> (BLEN+SLEN)) as B32; *idx +=1;
  unpack_mid(vals, idx, w);
  zcopy!(vals, *idx); *idx +=1;
  unpack_high(vals, idx, w);
  zcopy!(vals, *idx); *idx +=1;
  let mut curr_minor = xyidx;
  let mut curr_major = 14;
  if xyidx < 9 {
    vals[*idx] = (xyidx as B32)+1; *idx += 1;
    vals[*idx] = 14; *idx += 1;
  } else {
    vals[*idx] = (xyidx as B32)+1-9; *idx += 1;
    vals[*idx] = 15; *idx += 1;
    curr_minor -= 9;
    curr_major = 15;
  }
  zcopy!(vals, *idx); *idx +=1;
  vals[*idx] = blk; *idx += 1;
  vals[*idx] = midx; *idx += 1;
  onehot::<AUXBIN>(vals, idx, curr_minor);
  vals[*idx] = new_midx; *idx += 1;
  vals[*idx] = (inp & ((1<<16)-1)) as B32; *idx += 1;
  vals[*idx] = ((inp >> 16) & ((1<<16)-1)) as B32; *idx += 1;
  vals[*idx] = ((inp >> 32) & ((1<<16)-1)) as B32; *idx += 1;
  vals[*idx] = (inp >> 48) as B32; *idx += 1;
  onehot::<MAJLEN>(vals, idx, curr_major);
  *idx += 2;
  vals[*idx] = 1; *idx += 1;
}

//#[inline]
fn absorb_copy1(blk: B32, midx: B32,
               vals: &mut[B32], idx: &mut usize,
               prev_st: &[K64]) {
  for _ in 0..3 {
    vals[*idx] = vals[*idx - COLS*9 + BLEN*3 + SLEN]; *idx += 1;
  }
  for _ in 0..3 {
    vals[*idx] = vals[*idx - COLS*8 + BLEN*3 + SLEN - 3]; *idx += 1;
  }
  for _ in 0..3 {
    vals[*idx] = vals[*idx - COLS*7 + BLEN*3 + SLEN - 6]; *idx += 1;
  }
  for _ in 0..3 {
    vals[*idx] = vals[*idx - COLS*6 + BLEN*3 + SLEN - 9]; *idx += 1;
  }
  for _ in 0..3 {
    vals[*idx] = vals[*idx - COLS*5 + BLEN*3 + SLEN - 12]; *idx += 1;
  }
  for _ in 0..3 {
    vals[*idx] = vals[*idx - COLS*4 + BLEN*3 + SLEN - 15]; *idx += 1;
  }
  for _ in 0..3 {
    vals[*idx] = vals[*idx - COLS*3 + BLEN*3 + SLEN - 18]; *idx += 1;
  }
  for _ in 0..3 {
    vals[*idx] = vals[*idx - COLS*2 + BLEN*3 + SLEN - 21]; *idx += 1;
  }
  for _ in 0..3 {
    vals[*idx] = vals[*idx - COLS*1 + BLEN*3 + SLEN - 24]; *idx += 1;
  }
  expand_st(vals, idx, &prev_st, 9);
  zloop(vals, idx, RLEN-9 + BLEN*2 + 1); // includes minor
  vals[*idx] = 15; *idx += 1;
  zcopy!(vals, *idx); *idx +=1;
  vals[*idx] = blk; *idx += 1;
  vals[*idx] = midx; *idx += 1;
  onehot::<AUXLEN>(vals, idx, 9);
  onehot::<MAJLEN>(vals, idx, 14);
  *idx += 2;
  zcopy!(vals, *idx); *idx +=1;
}

//#[inline]
fn absorb_copy2(blk: B32, midx: B32,
                vals: &mut[B32], idx: &mut usize) {
  for _ in 0..BLEN+5 {
    vals[*idx] = vals[*idx - COLS*9]; *idx += 1;
  }
  for _ in 0..3 {
    vals[*idx] = vals[*idx - COLS*8 + BLEN*2 -5 + SLEN]; *idx += 1;
  }
  for _ in 0..3 {
    vals[*idx] = vals[*idx - COLS*7 + BLEN*2 -5 + SLEN - 3]; *idx += 1;
  }
  for _ in 0..3 {
    vals[*idx] = vals[*idx - COLS*6 + BLEN*2 -5 + SLEN - 6]; *idx += 1;
  }
  for _ in 0..3 {
    vals[*idx] = vals[*idx - COLS*5 + BLEN*2 -5 + SLEN - 9]; *idx += 1;
  }
  for _ in 0..3 {
    vals[*idx] = vals[*idx - COLS*4 + BLEN*2 -5 + SLEN - 12]; *idx += 1;
  }
  for _ in 0..3 {
    vals[*idx] = vals[*idx - COLS*3 + BLEN*2 -5 + SLEN - 15]; *idx += 1;
  }
  for _ in 0..3 {
    vals[*idx] = vals[*idx - COLS*2 + BLEN*2 -5 + SLEN - 18]; *idx += 1;
  }
  for _ in 0..3 {
    vals[*idx] = vals[*idx - COLS*1 + BLEN*2 -5 + SLEN - 21]; *idx += 1;
  }
  for _ in 0..(KWORDS-17)*3 {
    vals[*idx] = vals[*idx - COLS*9]; *idx += 1;
  }
  zloop(vals, idx, RLEN-9 + BLEN*2 + 3); // includes minor,major,rnd
  vals[*idx] = blk; *idx += 1;
  vals[*idx] = midx+17; *idx += 1;
  onehot::<AUXLEN>(vals, idx, 8);
  onehot::<MAJLEN>(vals, idx, 15);
  *idx += 2;
  zcopy!(vals, *idx); *idx +=0;
}

fn absorb_all(vals: &mut[B32], all: &[[K64; EWORDS]], stidx: usize) {
  let prev_st = all[stidx-1];
  let curr_st = all[stidx];
  let mut idx = curr_st[KWORDS+1] as usize; 
  let counters = curr_st[KWORDS];
  let midx = (counters >> (ROUNDBITS+MIDXBITS)) as B32;
  let blk = ((counters >> ROUNDBITS) as B32) & ((1<<MIDXBITS)-1);
  //let inp = curr_st[xidx+5*yidx
  for i in 0..9 {
    absorb_word(i, blk, midx, vals, &mut idx, &prev_st, &curr_st);
  }
  absorb_copy1(blk, midx, vals, &mut idx, &prev_st);
  for i in 9..17 {
    absorb_word(i, blk, midx, vals, &mut idx, &prev_st, &curr_st);
  }
  absorb_copy2(blk, midx, vals, &mut idx);
}

#[inline]
fn to_shorts(v: u32) -> (u32,u32) {
  (v&((1<<16)-1), v>>16)
}

#[inline]
fn add_shorts(a: (u32,u32), b: (u32,u32)) -> (u32,u32) {
  ((a.0).wrapping_add(b.0), (a.1).wrapping_add(b.1))
}

#[inline]
fn pack_shorts(v: (u32, u32)) -> u32 {
  (v.0).wrapping_add((v.1) << 16)
}

#[inline]
fn pack_32(vals: &[B32]) -> u32 {
  let mut v = 0;
  for idx in 0..32 {
    v = v | (vals[idx] << idx);
  }
  v
}

#[inline]
fn s2e(vals: &[B32]) -> u32 {
  let mut v = 0;
  for idx in 0..12 {
    v = v | (vals[idx+BLEN+RLEN+BLEN+10] << idx);
  }
  for idx in 0..20 {
    v = v | (vals[idx] << (idx+12));
  }
  v
}

//#[inline]
fn compute_ae(aa: &[u32], w: u32, rnd: usize,
              vals: &mut[B32], idx: &mut usize) -> (u32, u32) {
  let ch = (aa[4] & aa[5]) ^ (!aa[4] & aa[6]);
  let ma = (aa[0] & aa[1]) ^ (aa[0] & aa[2]) ^ (aa[1] & aa[2]);
  let s0 = aa[0].rotate_right(2) ^ aa[0].rotate_right(13) ^
           aa[0].rotate_right(22);
  let s1 = aa[4].rotate_right(6) ^ aa[4].rotate_right(11) ^
           aa[4].rotate_right(25);

  let t0 = add_shorts(
             add_shorts(
               add_shorts(
                 add_shorts(to_shorts(aa[7]), to_shorts(s1)),
                 to_shorts(ch)),
               to_shorts(tsha256::KRC[rnd])), 
             to_shorts(w));
  let t1 = add_shorts(to_shorts(s0), to_shorts(ma));
  let e_out = add_shorts(to_shorts(aa[3]), t0);
  let a_out = add_shorts(t0, t1);

  let ca = (a_out.0)>>16;
  let ce = (e_out.0)>>16;
  let padb = ca |
             (((a_out.1 + ca) & (((1<<5)-1)<<16)) >> (16-3)) |
             ((ce & (1<<3)-1) << 7) |
             (((e_out.1 + ce) & (((1<<2)-1)<<16)) >> (16-7-3));
  unpack32::<12>(vals, idx, padb);

  let packed_a = pack_shorts(a_out);
  let packed_e = pack_shorts(e_out);

  unpack32::<20>(vals, idx, packed_e >> 12);

  vals[*idx] = ((e_out.1) >> 18) & 1; *idx += 1;
  vals[*idx] = (e_out.1) >> 19; *idx += 1;
  (packed_a, packed_e)
}
  
fn sha256_blk(vals: &mut[B32], st: &[K64]) {
  let mut idx = st[KWORDS+1] as usize;

  let counters = st[KWORDS];
  let blk_counter = (counters & ((1<<32)-1)) as B32;
  let midx_counter = (counters >> 32) as B32;
  let mut aa = [st[8+0] as u32, st[8+1] as u32,
                st[8+2] as u32 ,st[8+3] as u32,
                st[8+4] as u32, st[8+5] as u32,
                st[8+6] as u32, st[8+7] as u32];
  let mut rnd = 0 as B32; 
  let (mut w1, mut wblk);
  let mut w2 = 0; 
  let mut inv_counter = 0;
  let mut nxtblk = 0;

  for j in 0..16 {
    let j2 = j / 2;
    nxtblk = blk_counter + j2;
    if 0 == rnd {
      w1 = st[j2 as usize] as u32;
      w2 = (st[j2 as usize] >> 32) as u32;
      wblk = nxtblk;
      rnd = 1;
      inv_counter = inv_babybear(midx_counter + 1 - wblk);
    } else {
      w1 = w2;
      w2 = 0;
      wblk = 0;
      rnd = 0;
      nxtblk += 1;
    }

    let byte1 = (w1<<24) & (((1<<8)-1)<<24); 
    let byte2 = (w1<<8) & (((1<<8)-1)<<16); 
    let byte3 = (w1 & (((1<<8)-1)<<16)) >> 8;
    let byte4 = w1>>24;

    let w = byte1|byte2|byte3|byte4;
    unpack32::<32>(vals, &mut idx, w);

    let (packed_a, packed_e) =
      compute_ae(&aa, w, j as usize, vals, &mut idx);

    zloop(vals, &mut idx, RLEN-6);

    vals[idx] = inv_counter; idx += 1;
    vals[idx] = w1 & ((1<<16)-1); idx += 1;
    vals[idx] = w1 >> 16; idx += 1;
    vals[idx] = w2 & ((1<<16)-1); idx += 1;
    vals[idx] = w2 >> 16; idx += 1;
    vals[idx] = wblk; idx += 1;

    unpack32::<32>(vals, &mut idx, packed_a);
    unpack32::<12>(vals, &mut idx, packed_e);

    vals[idx] = j+1; idx += 1;
    vals[idx] = 18; idx += 1;
    vals[idx] = rnd; idx += 1;
    vals[idx] = nxtblk; idx += 1;
    vals[idx] = midx_counter; idx += 1;

    onehot::<AUXLEN>(vals, &mut idx, j as usize);
    onehot::<MAJLEN>(vals, &mut idx, 18);
    idx += 2;
    let neg_rnd = (!rnd).wrapping_add(1);
    vals[idx] = neg_rnd&((BABYBEAR_MODULUS-1) as B32); idx += 1;

    aa[7] = aa[6];
    aa[6] = aa[5];
    aa[5] = aa[4];
    aa[4] = packed_e as B32;
    aa[3] = aa[2];
    aa[2] = aa[1];
    aa[1] = aa[0];
    aa[0] = packed_a as B32;
  }

  //correct the minor+major of last iteration
  vals[idx-1-2-MAJLEN-AUXLEN-1-1-1-1-1] = 0;
  vals[idx-1-2-MAJLEN-AUXLEN-1-1-1-1] = 19;

  for j in 16..64 {
    let w_2 = pack_32(&vals[idx-2*COLS..]);
    let w_7 = pack_32(&vals[idx-7*COLS..]);
    let w_15 = pack_32(&vals[idx-15*COLS..]);
    let w_16 = pack_32(&vals[idx-16*COLS..]);
    let w_s0 = w_15.rotate_right(7) ^ w_15.rotate_right(18) ^ (w_15 >> 3);
    let w_s1 = w_2.rotate_right(17) ^ w_2.rotate_right(19) ^ (w_2 >> 10);
    let wnew = add_shorts(
                 add_shorts(
                   add_shorts(to_shorts(w_7), to_shorts(w_s1)),
                   to_shorts(w_s0)),
                 to_shorts(w_16));
    let wnew_packed = pack_shorts(wnew);
    unpack32::<32>(vals, &mut idx, wnew_packed);

    let (packed_a, packed_e) =
      compute_ae(&aa, wnew_packed, j as usize, vals, &mut idx);

    let c1 = (wnew.0)>>16;
    let d1 = c1 | (((wnew.1 + c1) >> 16) << 3);
    unpack32::<7>(vals, &mut idx, d1);
    zloop(vals, &mut idx, RLEN-7-2);
    vals[idx] = vals[idx-COLS]; idx += 1;
    vals[idx] = vals[idx-COLS]; idx += 1;

    unpack32::<32>(vals, &mut idx, packed_a);
    unpack32::<12>(vals, &mut idx, packed_e);

    let curr_minor = j as usize & (16-1);
    let curr_major = 18 + (j as usize >> 4);
    let next_minor = (j+1) & (16-1);
    let next_major = 18 + ((j+1) >> 4);
    vals[idx] = next_minor; idx += 1;
    vals[idx] = next_major; idx += 1;
    vals[idx] = 0; idx += 1;
    vals[idx] = nxtblk; idx += 1;
    vals[idx] = midx_counter; idx += 1;
    onehot::<AUXLEN>(vals, &mut idx, curr_minor);
    onehot::<MAJLEN>(vals, &mut idx, curr_major);
    idx += 2;
    vals[idx] = 0; idx += 1;

    aa[7] = aa[6];
    aa[6] = aa[5];
    aa[5] = aa[4];
    aa[4] = packed_e as B32;
    aa[3] = aa[2];
    aa[2] = aa[1];
    aa[1] = aa[0];
    aa[0] = packed_a as B32;
  }

  //correct the minor+major of last iteration
  vals[idx-1-2-MAJLEN-AUXLEN-1-1-1-1-1] = 5;
  vals[idx-1-2-MAJLEN-AUXLEN-1-1-1-1] = 16;

  for j in 0 .. 4 {
    let a_4 = pack_32(&vals[idx-4*COLS+3*BLEN+RLEN..]);
    let a_68 = st[8+3-j] as u32;
    let e_4 = s2e(&vals[idx-4*COLS+2*BLEN..]);
    let e_68 = st[8+7-j] as u32;
    let new_a = a_4.wrapping_add(a_68);
    let new_e = e_4.wrapping_add(e_68);
    let mut carries =
      (a_4 & 0xffff).wrapping_add(a_68 & 0xffff) >> 16;
    carries |=
      (((a_4 as u64).wrapping_add(a_68 as u64) & 0x100000000)
      >> (32-3)) as u32;
    carries |=
      ((e_4 & 0xffff).wrapping_add(e_68 & 0xffff) & 0x10000)
      >> (16-3-4);
    carries |=
      (((e_4 as u64).wrapping_add(e_68 as u64) & 0x100000000)
      >> (32-3-4-3)) as u32;
    unpack32::<14>(vals, &mut idx, carries);
    zloop(vals, &mut idx, 8+BLEN);
    unpack32::<20>(vals, &mut idx, new_e >> 12);
    zloop(vals, &mut idx, 2+22);
    vals[idx] = vals[idx-COLS]; idx += 1;
    vals[idx] = vals[idx-COLS]; idx += 1;
    unpack32::<32>(vals, &mut idx, new_a);
    unpack32::<12>(vals, &mut idx, new_e);
    vals[idx] = 5+1+j as B32; idx += 1;
    vals[idx] = 16; idx += 1;
    vals[idx] = 0; idx += 1;
    vals[idx] = nxtblk; idx += 1;
    vals[idx] = midx_counter; idx += 1;
    onehot::<AUXLEN>(vals, &mut idx, 5+j as usize);
    onehot::<MAJLEN>(vals, &mut idx, 16);
    idx += 2;
    vals[idx] = 0; idx += 1;
  }

  //correct the last iteration
  let contd = midx_counter + 1 - nxtblk;
  if 0 == contd {
    vals[idx-COLS+BLEN+SLEN] = 0;
    vals[idx-COLS+3*BLEN+SLEN] = 0;
    vals[idx-1-2-MAJLEN-AUXLEN-1-1-1-1-1] = 9;
    vals[idx-1-2-MAJLEN-AUXLEN-1-1-1-1] = 16;
    vals[idx-1-2-MAJLEN-AUXLEN-1-1] = 0; //blk
    vals[idx-1-2-MAJLEN-AUXLEN-1] = 0; //midx
  } else {
    vals[idx-COLS+BLEN+SLEN] = inv_babybear(contd);
    vals[idx-COLS+3*BLEN+SLEN] = 1;
    vals[idx-1-2-MAJLEN-AUXLEN-1-1-1-1-1] = 0;
    vals[idx-1-2-MAJLEN-AUXLEN-1-1-1-1] = 18;
  }
  
}

fn print_row(vals: &[B32], row: usize) {
  let idx = row*COLS;
  for j in idx .. idx+COLS {
    print!("{} ", vals[j]);
  }
  println!();
}

//#[inline]
fn output_word(w: K64, midx_nxt: B32, xidx: B32,
               vals: &mut[B32], idx: &mut usize) {
  unpack_mid(vals, idx, w);
  zloop(vals, idx, 2+SLEN+BLEN+RLEN);
  unpack_high(vals, idx, w);
  zcopy!(vals, *idx); *idx +=1;
  unpack_low(vals, idx, w);
  vals[*idx] = xidx; *idx += 1;
  vals[*idx] = 17; *idx += 1;
  zloop(vals, idx, 2);
  vals[*idx] = midx_nxt; *idx += 1;
  onehot::<AUXBIN>(vals, idx, xidx as usize - 1);
  vals[*idx] = midx_nxt; *idx += 1;
  vals[*idx] = (w & ((1<<16)-1)) as B32; *idx += 1;
  vals[*idx] = ((w >> 16) & ((1<<16)-1)) as B32; *idx += 1;
  vals[*idx] = ((w >> 32) & ((1<<16)-1)) as B32; *idx += 1;
  vals[*idx] = (w >> 48) as B32; *idx += 1;
  onehot::<MAJLEN>(vals, idx, 17);
  *idx += 2;
  vals[*idx] = 1; *idx += 1;
}

//#[inline]
fn sha2init(v44: K64, v20: B32, midx: B32,
            minor: usize, new_minor: B32, new_major: B32,
            vals: &mut[B32], idx: &mut usize) {
  zloop(vals, idx, BLEN*2);
  unpack32::<20>(vals, idx, v20);
  zloop(vals, idx, 2+RLEN);
  unpack64::<0,44>(vals, idx, v44);
  vals[*idx] = new_minor; *idx += 1;
  vals[*idx] = new_major; *idx += 1;
  zloop(vals, idx, 2);
  vals[*idx] = midx; *idx += 1;
  onehot::<AUXLEN>(vals, idx, minor);
  onehot::<MAJLEN>(vals, idx, 16);
  *idx += 2;
  zcopy!(vals, *idx); *idx +=1;
}

//#[inline]
fn memzpad(bitlen: B32, midx_new: B32, nskip_new: B32,
           vals: &mut[B32], idx: &mut usize) {
  unpack32::<30>(vals, idx, bitlen);
  if 0 == nskip_new {
    zloop(vals, idx, 3*BLEN-30+2);
    let w1 = (bitlen << 8) & (((1<<8)-1)<<8); 
    let w2 = (bitlen>>8)&((1<<8)-1); 
    let w3 = (bitlen & (((1<<8)-1)<<16)) >> 8;
    let w4 = bitlen>>24;
    vals[*idx] = w4 | w3; *idx += 1;
    vals[*idx] = w2 | w1; *idx += 1;
  } else {
    zloop(vals, idx, 3*BLEN-30+4);
  }
  let invn = inv_babybear(nskip_new);
  vals[*idx] = invn; *idx += 1;
  vals[*idx] = mult_babybear(nskip_new, invn); *idx += 1;
  vals[*idx] = midx_new; *idx += 1;
  zloop(vals, idx, RLEN-7+2*BLEN);
  if 0 == nskip_new {
    vals[*idx] = 1; *idx += 1;
    vals[*idx] = 16; *idx += 1;
    zloop(vals, idx, 2);
  } else {
    vals[*idx] = 0; *idx += 1;
    vals[*idx] = 16; *idx += 1;
    vals[*idx] = bitlen; *idx += 1;
    vals[*idx] = nskip_new - 1; *idx += 1;
  }
  vals[*idx] = midx_new; *idx += 1;
  onehot::<AUXLEN>(vals, idx, 0);
  onehot::<MAJLEN>(vals, idx, 16);
  *idx += 2;
  vals[*idx] = 1; *idx += 1;
}

//#[inline]
fn final_write_and_padding(vals: &mut[B32], st: &[K64]) {
  let mut midxnn = st[KWORDS] as B32;
  let mut idx = st[KWORDS+1] as usize; 
  let mut nskip = 7 - ((midxnn+1)&7);
  zloop(vals, &mut idx, SLEN);
  vals[idx] = nskip; idx += 1;
  zloop(vals, &mut idx, 4*BLEN-3+RLEN);
  vals[idx] = nskip&1; idx += 1;
  vals[idx] = (nskip>>1)&1; idx += 1;
  vals[idx] = (nskip>>2)&1; idx += 1;
  zcopy!(vals, idx); idx +=1;
  vals[idx] = 16; idx += 1;
  let bitlen = 64*midxnn;
  vals[idx] = bitlen; idx += 1;
  vals[idx] = nskip; idx += 1;
  vals[idx] = midxnn; idx += 1;
  onehot::<AUXBIN>(vals, &mut idx, 4);
  vals[idx] = midxnn; idx += 1;
  vals[idx] = 0x80; idx += 1;
  zloop(vals, &mut idx, 3);
  onehot::<MAJLEN>(vals, &mut idx, 17);
  idx += 2;
  vals[idx] = 1; idx += 1;
  loop {
    midxnn += 1;
    memzpad(bitlen, midxnn, nskip, vals, &mut idx);
    if 0 == nskip { break; } 
    nskip -= 1;
  }
  sha2init(0xD19A54FF53A, 0x5BE0C, midxnn, 1, 2, 16, vals, &mut idx);
  sha2init(0x9AB3C6EF372, 0x1F83D, midxnn, 2, 3, 16, vals, &mut idx);
  sha2init(0x88CBB67AE85, 0x9B056, midxnn, 3, 4, 16, vals, &mut idx);
  sha2init(0x27F6A09E667, 0x510E5, midxnn, 4, 0, 18, vals, &mut idx);
}

//#[inline]
fn end_kinstance(vals: &mut[B32], all: &[[K64; EWORDS]], stidx: usize) {
  let prev_st = all[stidx-1];
  let curr_st = all[stidx];
  let mut idx = curr_st[KWORDS+1] as usize; 
  //increment midx because tkeccak didn't
  let midx = 1 + (prev_st[KWORDS] >> (ROUNDBITS+MIDXBITS)) as B32;
  output_word(prev_st[1], midx+1, 1, vals, &mut idx);
  output_word(prev_st[2], midx+2, 2, vals, &mut idx);
  output_word(prev_st[3], midx+3, 3, vals, &mut idx);
  let midxnxt = midx+4;
  let blk = curr_st[KWORDS] as B32; 
  let inv_blk = inv_babybear(blk);
  zloop(vals, &mut idx, BLEN*4);
  vals[idx] = inv_blk; idx += 1;
  vals[idx] = mult_babybear(blk, inv_blk); idx += 1;
  zloop(vals, &mut idx, BLEN*2);
  if 0 == blk {
    vals[idx] = 4; idx += 1;
    vals[idx] = 17; idx += 1;
    zloop(vals, &mut idx, 2);
  } else {
    zcopy!(vals, idx); idx +=1;
    vals[idx] = 14; idx += 1;
    zcopy!(vals, idx); idx +=1;
    vals[idx] = blk; idx += 1;
  }
  vals[idx] = midxnxt; idx += 1;
  onehot::<AUXBIN>(vals, &mut idx, 3);
  vals[idx] = midxnxt; idx += 1;
  vals[idx] = blk; idx += 1;
  zloop(vals, &mut idx, 3);
  onehot::<MAJLEN>(vals, &mut idx, 17);
  idx += 2;
  vals[idx] = 1; //idx += 1;
} 

fn first_cycle(vals: &mut[B32], first_st: &[K64]) {
  let mut idx = 0;
  let blk = first_st[KWORDS] as B32;
  debug_assert!(blk != 0);
  zloop(vals, &mut idx, 5*BLEN+RLEN+1); //includes minor
  vals[idx] = 14; idx += 1; //major
  zcopy!(vals, idx); idx +=1; //rnd
  vals[idx] = blk; idx += 1;
  zcopy!(vals, idx); idx +=1; //midx
  onehot::<AUXBIN>(vals, &mut idx, 5);
  zcopy!(vals, idx); idx +=1; //memarg_idx
  vals[idx] = blk; idx += 1; //memarg_w1
  zloop(vals, &mut idx, 3); //memarg_w234
  onehot::<MAJLEN>(vals, &mut idx, 17);
  idx += 2;
  vals[idx] = 1; //memarg_count
}

fn keccak_witgen(vals: &mut[B32], all: &[[K64; EWORDS]]) {
  first_cycle(vals, &all[0]);
  for (j, st) in all.iter().enumerate().skip(1).rev() {
    match st[KWORDS+2] {
      op::PERMUTE  => perm_round(vals, st),
      op::ABSORB => absorb_all(vals, all, j),
      op::ENDK => end_kinstance(vals, all, j),
      op::MEMZPAD => final_write_and_padding(vals, st),
      op::SHA2BLK => sha256_blk(vals, st),
      _ => continue,
    }
  }
}

fn sequential_preparation(inp: &[K64]) -> (Vec<[K64; EWORDS]>, usize)  {
  let mut sum = 1; // includes last blk=0
  let mut pos = 0;
  let mut last = inp[0];
  while last > 0 {
    sum += last as usize * (2+ROUNDS) + 1;
    pos += last as usize * KINPWORDS + 1;
    last = inp[pos];
  }

  // inexact upper bound for sha256 cycles: 10 > 27/((136+32+1)/64)
  // +1 roundup, +1 endk, +1 memzpad, +1 sha256init
  sum += sum/10 + 4; 

  let mut st = vec![[0 as K64; EWORDS]; sum];
  let mut inject_kout = Vec::new();

  last = inp[0];
  pos = 1;
  st[0][KWORDS] = last; //st for 1st cycle
  let mut idx = 1;
  let mut start = COLS * 1;
  let mut midx = 0;
  let mut total_skip;
  
  loop { //keccak states after each of its 24 rounds
    for j in 0 .. KINPWORDS {
      st[idx][j] = inp[pos]; pos += 1;
    }
    st[idx][KWORDS] = 0 | (last<<ROUNDBITS) | (midx<<(ROUNDBITS+MIDXBITS));
    st[idx][KWORDS+1] = start as K64;
    st[idx][KWORDS+2] = op::ABSORB;

    start += COLS*19;
    midx += 17;
    idx += 1;

    for j in 0 .. KWORDS {
      st[idx][j] = st[idx-1][j] ^ st[idx-2][j];
    }
    st[idx][KWORDS] = 0 | (last<<ROUNDBITS) | (midx<<(ROUNDBITS+MIDXBITS));
    st[idx][KWORDS+1] = start as K64;
    st[idx][KWORDS+2] = op::PERMUTE;

    tkeccak::keccakf(&mut st, idx);
    st[idx+ROUNDS][KWORDS+2] = op::SKIP;

    start += COLS * ROUNDS * PERMROWS;
    idx += ROUNDS + 1;
    last -= 1;
    if 0 == last {
      let endk = &st[idx-1];
      inject_kout.push([pos as K64, endk[0], endk[1], endk[2], endk[3]]);
      last = inp[pos]; pos += 1;
      st[idx][KWORDS] = last;
      st[idx][KWORDS+1] = start as K64;
      st[idx][KWORDS+2] = op::ENDK;
      start += COLS*4;
      midx += 5; // +1 was written by iota of rc=24
      idx += 1;
      if 0 == last {
        break;
      }
    }
  }
  debug_assert!(inject_kout.len() > 0);

  midx += 1;

  let tmplen = (midx*64) as u32;
  let w1 = (tmplen<<24) & (((1<<8)-1)<<24);
  let w2 = (tmplen<<8) & (((1<<8)-1)<<16);
  let w3 = (tmplen>>8) & (((1<<8)-1)<<8);
  let w4 = (tmplen>>24) & ((1<<8)-1);
  let bitlen = w1|w2|w3|w4;

  //mem padding of sha256
  st[idx][KWORDS] = midx;
  st[idx][KWORDS+1] = start as K64;
  st[idx][KWORDS+2] = op::MEMZPAD;
  total_skip = 8-((midx as usize + 1)&7);
  midx += total_skip as K64;
  total_skip += 1;
  start += COLS*(total_skip+4); //+4 for sha2init
  idx += 1;

  let mut s2counters = midx << 32;

  //first sha256 st+inp
  for j in 0..8 {
    st[idx][j] = inp[j];
    st[idx][j+8] = tsha256::INIT[j] as u64; 
  }
  st[idx][KWORDS] = s2counters;
  st[idx][KWORDS+1] = start as K64;
  st[idx][KWORDS+2] = op::SHA2BLK;

  //first run to fill the inputs
  let mut bakidx = idx;
  let mut inner = 0;
  pos = 0;
  for elem in inject_kout {
    while pos < elem[0] as usize {
      st[bakidx][inner] = inp[pos];
      pos += 1;
      if inner < 7 { inner += 1; } else { inner = 0; bakidx += 1; }
    }
    for k in 1..=4 {
      st[bakidx][inner] = elem[k];
      if inner < 7 { inner += 1; } else { inner = 0; bakidx += 1; }
    }
  }
  st[bakidx][inner] = 0; // last blk=0
  if inner < 7 { inner += 1; } else { inner = 0; bakidx += 1; }
  st[bakidx][inner] = 0x80;
  //nskip are already zeros, last input filled later

  //rerun for every sha256 state
  debug_assert!(((midx+1)%8) == 0);
  for _blk in 1..((midx+1)/8) {
    tsha256::sha256f(&mut st, idx);
    idx += 1;
    start += COLS * SHA2ROWS;
    s2counters += 8;
    st[idx][KWORDS] = s2counters;
    st[idx][KWORDS+1] = start as K64;
    st[idx][KWORDS+2] = op::SHA2BLK;
  }
  st[idx][7] = (bitlen as K64) << 32; //fill the last input
  tsha256::sha256f(&mut st, idx);

  (st, total_skip)
}

fn calc_witness_len(inp: &[K64], padskip: usize) -> usize  {
  let mut sum = 1+padskip+4;
  let mut pos = 0;
  let mut last = inp[0] as usize;
  let mut injected = 0;
  loop {
    sum += last*(19+ROUNDS*PERMROWS) + 4;
    injected += 4;
    pos += last * KINPWORDS + 1;
    last = inp[pos] as usize;
    if 0 == last { break; }
  }
  let len = inp.len() + injected + padskip;
  debug_assert!(len % 8 == 0);
  sum + SHA2ROWS*(len/8)
}

fn u8_as_u64(inp: &[u8]) -> &[u64] {
  let ptr = inp.as_ptr();
  let input_len = inp.len() / core::mem::size_of::<u64>();
  debug_assert!(inp.len() % core::mem::size_of::<u64>() == 0);
  unsafe {
      core::slice::from_raw_parts(ptr as *const u64, input_len)
  }
}

fn decode_input(inp: &str) -> Vec<u8> {
  assert!(inp.len() % 16 == 0);
  let Ok(result) = (0..inp.len())
      .step_by(2)
      .map(|i| u8::from_str_radix(&inp[i..i + 2], 16))
      .collect()
    else { panic!() };
  result
}

fn main() {
/*
  let inp1 = [0x0000000000000001 as K64,
              0x6369757120656854, 0x206e776f7262206b,
              0x706d756a20786f66, 0x74207265766f2073,
              0x20797a616c206568, 0x12e676f64,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0x8000000000000000,
              0];
  let inp2 = [0x0000000000000006 as K64,
              0x65646E616D6D6F43, 0x697265646F522072,
              0x6E69616C42206B63, 0x64656B6F6F6C2065,
              0x6369746E61726620, 0x6F726120796C6C61,
              0x2065687420646E75, 0x202E656764697262,
              0x6968206572656877, 0x65636966666F2073,
              0x2065726577207372, 0x6E69746365726964,
              0x7269617065722067, 0x6C20687469772073,
              0x7520646E6120776F, 0x6F7620746E656772,
              0x7573202C73656369,
              0x6120736E6F656772, 0x676E697473697373,
              0x6964206120746120, 0x20746C7563696666,
              0x6F6974617265706F, 0x6720656854202E6E,
              0x6565747320796172, 0x7261706D6F63206C,
              0x617720746E656D74, 0x666E6F6320612073,
              0x666F206E6F697375, 0x7469766974636120,
              0x636165202C736569, 0x6C726564726F2068,
              0x7374692079622079, 0x2074756220666C65,
              0x7265766F20656874,
              0x72706D69206C6C61, 0x77206E6F69737365,
              0x686320666F207361, 0x726353202E736F61,
              0x6F626120736E6565, 0x6820656E6F206576,
              0x276E616D736D6C65, 0x6F69746174732073,
              0x6465776F6873206E, 0x616C702065687420,
              0x6F6C65622074656E, 0x687420646E612077,
              0x2C726568746F2065, 0x6920737069687320,
              0x20746962726F206E, 0x63614D207261656E,
              0x202C727568747241,
              0x7265766520747562, 0x6520657265687779,
              0x206568742065736C, 0x6F63206C656E6170,
              0x6461682073726576, 0x6572206E65656220,
              0x7266206465766F6D, 0x6F736E6F63206D6F,
              0x736574202C73656C, 0x757274736E692074,
              0x65772073746E656D, 0x7070696C63206572,
              0x206F746E69206465, 0x6E69207269656874,
              0x61202C7365646973, 0x6E6863657420646E,
              0x7320736E61696369,
              0x20796220646F6F74, 0x6C6F632068746977,
              0x6465646F632D726F, 0x6F727463656C6520,
              0x657373612063696E, 0x74207365696C626D,
              0x63616C706572206F, 0x7479726576652065,
              0x61687420676E6968, 0x64656D6565732074,
              0x75667462756F6420, 0x706D756854202E6C,
              0x687720646E612073, 0x756F732073656E69,
              0x726874206465646E, 0x656874206867756F,
              0x3938207069687320,
              0x656877656D6F7320, 0x7420746661206572,
              0x6E69676E65206568, 0x6320676E69726565,
              0x6B726F7720776572, 0x6874206E6F206465,
              0x012E6C6C75682065, 0x0000000000000000,
              0x0000000000000000, 0x0000000000000000,
              0x0000000000000000, 0x0000000000000000,
              0x0000000000000000, 0x0000000000000000,
              0x0000000000000000, 0x0000000000000000,
              0x8000000000000000,
              0x0000000000000002,
              0x6F77206573656854, 0x6572657720736472,
              0x6465726574747520, 0x796C754A206E6920,
              0x7962203530383120, 0x615020616E6E4120,
              0x5320616E766F6C76, 0x202C726572656863,
              0x6E69747369642061, 0x2064656873697567,
              0x20666F207964616C, 0x72756F6320656874,
              0x6320646E61202C74, 0x746E656469666E6F,
              0x6469616D206C6169, 0x6F6E6F682D666F2D,
              0x6874206F74207275,
              0x736572706D452065, 0x20617972614D2073,
              0x766F726F646F7946, 0x77207449202E616E,
              0x6720726568207361, 0x20676E6974656572,
              0x636E697250206F74, 0x6C69737361562065,
              0x6E616D2061202C79, 0x6E69206867696820,
              0x6E61206B6E617220, 0x65636966666F2064,
              0x6177206F6877202C, 0x6966206568742073,
              0x61206F7420747372, 0x7461206576697272,
              0x8000000000000001,
              0];
  let inpstr1 =
    "010000000000000054686520717569636B2062726F776E20666F78206A756D70\
     73206F76657220746865206C617A7920646F672E010000000000000000000000\
     0000000000000000000000000000000000000000000000000000000000000000\
     0000000000000000000000000000000000000000000000000000000000000000\
     000000000000000000000000000000800000000000000000";
*/
  let inpstr2 =
    "0600000000000000436F6D6D616E64657220526F64657269636B20426C61696E\
     65206C6F6F6B6564206672616E746963616C6C792061726F756E642074686520\
     6272696467652E20776865726520686973206F66666963657273207765726520\
     646972656374696E6720726570616972732077697468206C6F7720616E642075\
     7267656E7420766F696365732C2073757267656F6E7320617373697374696E67\
     206174206120646966666963756C74206F7065726174696F6E2E205468652067\
     72617920737465656C20636F6D706172746D656E7420776173206120636F6E66\
     7573696F6E206F6620616374697669746965732C2065616368206F726465726C\
     7920627920697473656C662062757420746865206F766572616C6C20696D7072\
     657373696F6E20776173206F66206368616F732E2053637265656E732061626F\
     7665206F6E652068656C6D736D616E27732073746174696F6E2073686F776564\
     2074686520706C616E65742062656C6F7720616E6420746865206F746865722C\
     20736869707320696E206F72626974206E656172204D61634172746875722C20\
     627574206576657279776865726520656C7365207468652070616E656C20636F\
     7665727320686164206265656E2072656D6F7665642066726F6D20636F6E736F\
     6C65732C207465737420696E737472756D656E7473207765726520636C697070\
     656420696E746F20746865697220696E73696465732C20616E6420746563686E\
     696369616E732073746F6F64206279207769746820636F6C6F722D636F646564\
     20656C656374726F6E696320617373656D626C69657320746F207265706C6163\
     652065766572797468696E672074686174207365656D656420646F7562746675\
     6C2E205468756D707320616E64207768696E657320736F756E64656420746872\
     6F75676820746865207368697020383920736F6D657768657265206166742074\
     686520656E67696E656572696E67206372657720776F726B6564206F6E207468\
     652068756C6C2E01000000000000000000000000000000000000000000000000\
     0000000000000000000000000000000000000000000000000000000000000000\
     0000000000000000000000000000000000000000000000800200000000000000\
     546865736520776F7264732077657265207574746572656420696E204A756C79\
     203138303520627920416E6E61205061766C6F766E6120536368657265722C20\
     612064697374696E67756973686564206C616479206F662074686520636F7572\
     742C20616E6420636F6E666964656E7469616C206D6169642D6F662D686F6E6F\
     757220746F2074686520456D7072657373204D617279612046796F646F726F76\
     6E612E2049742077617320686572206772656574696E6720746F205072696E63\
     652056617373696C792C2061206D616E206869676820696E2072616E6B20616E\
     64206F66666963652C2077686F207761732074686520666972737420746F2061\
     727269766520617401000000000000800000000000000000";

  let decoded = decode_input(inpstr2);
  let inp  = u8_as_u64(&decoded);
  //let inp = &inp2;

  let (st, tskip) = sequential_preparation(inp);
  let needed_cycles = calc_witness_len(inp, tskip);
  let dummy_cycles = 99; // any arbitrary #cycles at the end
  let total_cycles = needed_cycles + dummy_cycles;
  let mut values = vec![0; COLS*total_cycles];

  keccak_witgen(&mut values, &st);

  for i in (0 .. total_cycles as u32).rev() {
    write_pc(&mut values, i);
  }

  for i in (needed_cycles as u32 .. total_cycles as u32).rev() {
    write_dummy(&mut values, i);
  }

  for i in 0 .. total_cycles {
    print_row(&values, i);
  }
}

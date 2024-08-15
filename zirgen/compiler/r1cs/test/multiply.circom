pragma circom 2.1.6;

template Multiply() {
  signal input a;
  signal input b;
  signal input c;
  signal s1;
  signal output out;
  
  s1 <== a * b;
  out <== s1 * c;
}

component main = Multiply();

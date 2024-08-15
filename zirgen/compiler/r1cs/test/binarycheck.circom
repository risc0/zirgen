pragma circom 2.0.0;

template binaryCheck () {

   // Declaration of signals.

   signal input in;
   signal output out;

   // Statements.

   in * (in-1) === 0;
   out <== in;
}

component main = binaryCheck();


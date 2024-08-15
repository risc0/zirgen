// RUN: zirgen-opt --compute-taps --make-verify-taps %s | FileCheck %s

func.func @get(%cbuf : !zll.buffer<16, constant>,
               %unused : !zll.val<BabyBear>,
               %mbuf : !zll.buffer<16, mutable>)
               -> !zll.val<BabyBear> {
   // CHECK-LABEL: func @verify_taps
   // TODO: Check for more than function name.
   %0 = zll.get %cbuf[0] back 5 : <16, constant>
   %1 = zll.get %mbuf[0] back 2 : <16, mutable>
   %2 = zll.get %mbuf[0] back 0 : <16, mutable>
   return %2 : !zll.val<BabyBear>
}

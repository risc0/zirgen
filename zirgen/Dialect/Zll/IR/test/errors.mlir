// RUN: zirgen-opt --compute-taps --verify-diagnostics %s

func.func @valmismatch(%buf : !zll.buffer<1, constant>,
                       %unused : !zll.val<BabyBear>,
                       %bufgl : !zll.buffer<1, constant, <Goldilocks>>) {
   %0 = zll.get %buf[0] back 5 : <1, constant>
// expected-error@+1 {{All val types must be the same}} 
   %1 = zll.get %bufgl[0] back 3 : <1, constant, <Goldilocks>>
   return
}

// RUN: zirgen-opt --compute-taps --make-verify-taps %s | FileCheck %s

module attributes {zll.buffers = #zll<buffers ("code", <16, constant>, 1), ("out", <1, global>), ("data", <16, mutable>, 2), ("mix", <1, global>), ("accum", <1, mutable>, 0)>, zll.protocolInfo = #zll<protocol_info "TEST____________">, zll.steps = #zll<steps "exec", "ram_verify", "verify_bytes", "compute_accum", "verify_accum">} {

func.func @get(%cbuf : !zll.buffer<16, constant> {zirgen.argName = "code"},
               %unused : !zll.val<BabyBear>,
               %mbuf : !zll.buffer<16, mutable> {zirgen.argName = "data"})
               -> !zll.val<BabyBear> {
   // CHECK-LABEL: func @verify_taps
   // TODO: Check for more than function name.
   %0 = zll.get %cbuf[0] back 5 : <16, constant>
   %1 = zll.get %mbuf[0] back 2 : <16, mutable>
   %2 = zll.get %mbuf[0] back 0 : <16, mutable>
   return %2 : !zll.val<BabyBear>
}

}

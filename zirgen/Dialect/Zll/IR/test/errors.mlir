// RUN: zirgen-opt --compute-taps --verify-diagnostics %s

module attributes {zll.buffers = #zll<buffers ("code", <16, constant>, 1), ("out", <1, global>), ("data", <16, mutable>, 2), ("mix", <1, global>), ("accum", <1, mutable>, 0)>, zll.protocolInfo = #zll<protocol_info "TEST____________">, zll.steps = #zll<steps "exec", "ram_verify", "verify_bytes", "compute_accum", "verify_accum">} {

func.func @valmismatch(%buf : !zll.buffer<1, constant> {zirgen.argName = "code"},
                       %unused : !zll.val<BabyBear>,
                       %bufgl : !zll.buffer<1, constant, <Goldilocks>> {zirgen.argName = "data"}) {
   %0 = zll.get %buf[0] back 5 : <1, constant>
// expected-error@+1 {{All val types must be the same}} 
   %1 = zll.get %bufgl[0] back 3 : <1, constant, <Goldilocks>>
   return
}

}

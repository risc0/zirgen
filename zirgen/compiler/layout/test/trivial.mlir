// RUN: zirgen-opt --optimize-layout --lower-composites %s | FileCheck %s

!val = !zll.val<BabyBear>
!ref = !zstruct.ref<!val>

!Foo = !zstruct.struct<
  "Foo", <
    a: !ref,
    b: !ref
  >
>

!Bar = !zstruct.struct<
  "Bar", <
    b: !ref,
    a: !ref
  >
>

!Mux = !zstruct.union<
  "Mux", <
    foo: !Foo,
    bar: !Bar
  >
>

!Top = !zstruct.struct<
  "Top", <
    mux: !Mux,
    data: !val
  >
>

func.func @Top(%arg: !Top) -> !Top {
  // CHECK-LABEL: @Top
  %mux = zstruct.lookup %arg["mux"] : (!Top) -> !Mux
  %foo = zstruct.lookup %mux["foo"] : (!Mux) -> !Foo
  %bar = zstruct.lookup %mux["bar"] : (!Mux) -> !Bar
  // CHECK: zll.slice %{{[0-9]+}}, [[OFFSET:[0-9]+]], 1
  %foo_a = zstruct.lookup %foo["a"] : (!Foo) -> !ref
  // CHECK-NEXT: zll.slice %{{[0-9]+}}, [[OFFSET]], 1
  %bar_b = zstruct.lookup %bar["b"] : (!Bar) -> !ref
  // CHECK: zll.slice %{{[0-9]+}}, [[OFFSET:[0-9]+]], 1
  %foo_b = zstruct.lookup %foo["b"] : (!Foo) -> !ref
  // CHECK-NEXT: zll.slice %{{[0-9]+}}, [[OFFSET]], 1
  %bar_a = zstruct.lookup %bar["a"] : (!Bar) -> !ref
  return %arg: !Top
}


// RUN: true
// TODO: Re-enable or delete this?  Otherwise run: zirgen-opt --lower-composites %s | FileCheck %s

!val = !zll.val<BabyBear>
!ref = !zstruct.ref<!val>
!struct_A = !zstruct.struct<"A", <foo:!ref, bar:!ref>>
!union_A = !zstruct.union<"A", <foo:!ref, bar:!ref>>
!struct_B = !zstruct.struct<"B", <baz:!struct_A, qux:!union_A>>
!union_B = !zstruct.union<"B", <baz:!struct_A, qux:!union_A>>
!val_16 = !zstruct.array<!ref, 16>
!struct_A_8 = !zstruct.array<!struct_A, 8>

func.func @pass_array(%arg : !val_16) -> !val_16 {
  // CHECK-LABEL: @pass_array
  // CHECK-NEXT: return %arg0 : !zll.buffer<16, mutable>
  return %arg : !val_16
}

func.func @get_from_array(%arg : !val_16) -> !val {
  // CHECK-LABEL: @get_from_array
  // CHECK: %0 = zll.slice %arg0, 0, 1 : <16, mutable>
  %c0 = arith.constant 0 : index
  %ref = zstruct.subscript %arg[index %c0] : (!val_16) -> !ref
  // CHECK-NEXT: %1 = zll.get %0[0] back 0 : <1, mutable>
  %ret = zstruct.load %ref : (!ref) -> !val
  // CHECK-NEXT: return %1 : !zll.val<BabyBear>
  return %ret : !val
}

func.func @store_to_array(%arg : !val_16, %x : !val) -> !val_16 {
  // CHECK-LABEL: @store_to_array
  // CHECK: %0 = zll.slice %arg0, 0, 1 : <16, mutable>
  %c0 = arith.constant 0 : index
  %ref = zstruct.subscript %arg[index %c0] : (!val_16) -> !ref
  // CHECK-NEXT: zll.set %0 : <1, mutable>[0] = %arg1 : <BabyBear>
  zstruct.store %ref, %x : (!val) -> !ref
  // CHECK-NEXT: return %arg0 : !zll.buffer<16, mutable>
  return %arg : !val_16
}

func.func @pass_struct(%arg : !struct_A) -> !struct_A {
  // CHECK-LABEL: @pass_struct
  // CHECK-NEXT: return %arg0 : !zll.buffer<2, mutable>
  return %arg : !struct_A
}

func.func @get_from_struct(%arg : !struct_A) -> !val {
  // CHECK-LABEL: @get_from_struct
  // CHECK-NEXT: %0 = zll.slice %arg0, 0, 1 : <2, mutable>
  %ref = zstruct.lookup %arg ["foo"] : (!struct_A) -> !ref
  // CHECK-NEXT: %1 = zll.get %0[0] back 0 : <1, mutable>
  %val = zstruct.load %ref : (!ref) -> !val
  // CHECK-NEXT: return %1 : !zll.val<BabyBear>
  return %val : !val
}

func.func @pass_through_struct(%arg : !struct_A) -> !val {
  // CHECK-LABEL: @pass_through_struct
  // CHECK-NEXT: %0 = call @get_from_struct(%arg0) : (!zll.buffer<2, mutable>) -> !zll.val<BabyBear>
  %ret = func.call @get_from_struct(%arg) : (!struct_A) -> !val
  return %ret : !val
}

func.func @store_to_struct(%arg : !struct_A, %x : !val) -> !struct_A {
  // CHECK-LABEL: @store_to_struct
  // CHECK-NEXT:  %0 = zll.slice %arg0, 0, 1 : <2, mutable>
  %ref = zstruct.lookup %arg ["foo"] : (!struct_A) -> !ref
  // CHECK-NEXT: zll.set %0 : <1, mutable>[0] = %arg1 : <BabyBear>
  zstruct.store %ref, %x : (!val) -> !ref
  // CHECK-NEXT: return %arg0 : !zll.buffer<2, mutable>
  return %arg : !struct_A
}

func.func @pass_union(%arg : !union_A) -> !union_A {
  // CHECK-LABEL: @pass_union
  // CHECK-NEXT: return %arg0 : !zll.buffer<1, mutable>
  return %arg : !union_A
}

func.func @get_from_union(%arg : !union_A) -> !val {
  // CHECK-LABEL: @get_from_union
  // CHECK-NEXT: %0 = zll.slice %arg0, 0, 1 : <1, mutable>
  %ref = zstruct.lookup %arg ["foo"] : (!union_A) -> !ref
  // CHECK-NEXT: %1 = zll.get %0[0] back 0 : <1, mutable>
  %val = zstruct.load %ref : (!ref) -> !val
  // CHECK-NEXT: return %1 : !zll.val<BabyBear>
  return %val : !val
}

func.func @store_to_union(%arg : !union_A, %x : !val) -> !union_A {
  // CHECK-LABEL: @store_to_union
  // CHECK-NEXT: %0 = zll.slice %arg0, 0, 1 : <1, mutable>
  %ref = zstruct.lookup %arg ["foo"] : (!union_A) -> !ref
  // CHECK-NEXT: zll.set %0 : <1, mutable>[0] = %arg1 : <BabyBear>
  zstruct.store %ref, %x : (!val) -> !ref
  // CHECK-NEXT: return %arg0 : !zll.buffer<1, mutable>
  return %arg : !union_A
}

func.func @get_2level(%arg : !struct_A_8) -> !val {
  // CHECK-LABEL: @get_2level
  // CHECK: %0 = zll.slice %arg0, 6, 2 : <16, mutable>
  %c3 = arith.constant 3 : index
  %ref3 = zstruct.subscript %arg[index %c3] : (!struct_A_8) -> !struct_A
  // CHECK-NEXT: %1 = zll.slice %0, 0, 1 : <2, mutable>
  %ref3foo = zstruct.lookup %ref3 ["foo"] : (!struct_A) -> !ref
  // CHECK-NEXT: %2 = zll.get %1[0] back 0 : <1, mutable>
  %val3foo = zstruct.load %ref3foo : (!ref) -> !val
  // CHECK-NEXT: return %2 : !zll.val<BabyBear>
  return %val3foo : !val
}

func.func @sum_2level(%arg : !struct_A_8) -> !val {
  // CHECK-LABEL: @sum_2level
  // CHECK: %0 = zll.slice %arg0, 6, 2 : <16, mutable>
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %ref3 = zstruct.subscript %arg[index %c3] : (!struct_A_8) -> !struct_A
  // CHECK-NEXT:  %1 = zll.slice %0, 0, 1 : <2, mutable>
  %ref3foo = zstruct.lookup %ref3 ["foo"] : (!struct_A) -> !ref
  // CHECK-NEXT:  %2 = zll.get %1[0] back 0 : <1, mutable>
  %val3foo = zstruct.load %ref3foo : (!ref) -> !val
  // CHECK-NEXT:  %3 = zll.slice %0, 1, 1 : <2, mutable>
  %ref3bar = zstruct.lookup %ref3 ["bar"] : (!struct_A) -> !ref
  // CHECK-NEXT:  %4 = zll.get %3[0] back 0 : <1, mutable>
  %val3bar = zstruct.load %ref3bar : (!ref) -> !val
  // CHECK-NEXT:  %5 = zll.add %2 : <BabyBear>, %4 : <BabyBear>
  %sum3 = zll.add %val3foo : !val, %val3bar : !val
  // CHECK-NEXT:  %6 = zll.slice %arg0, 10, 2 : <16, mutable>
  %ref5 = zstruct.subscript %arg[index %c5] : (!struct_A_8) -> !struct_A
  // CHECK-NEXT:  %7 = zll.slice %6, 0, 1 : <2, mutable>
  %ref5foo = zstruct.lookup %ref5 ["foo"] : (!struct_A) -> !ref
  // CHECK-NEXT:  %8 = zll.get %7[0] back 0 : <1, mutable>
  %val5foo = zstruct.load %ref5foo : (!ref) -> !val
  // CHECK-NEXT:  %9 = zll.slice %6, 1, 1 : <2, mutable>
  %ref5bar = zstruct.lookup %ref5 ["bar"] : (!struct_A) -> !ref
  // CHECK-NEXT:  %10 = zll.get %9[0] back 0 : <1, mutable>
  %val5bar = zstruct.load %ref5bar : (!ref) -> !val
  // CHECK-NEXT:  %11 = zll.add %8 : <BabyBear>, %10 : <BabyBear>
  %sum5 = zll.add %val5foo : !val, %val5bar : !val
  // CHECK-NEXT:  %12 = zll.add %5 : <BabyBear>, %11 : <BabyBear>
  %ret = zll.add %sum3 : !val, %sum5 : !val
  // CHECK-NEXT:  return %12 : !zll.val<BabyBear>
  return %ret : !val
}

// ensure that non-default field prime types pass through, intact
!xval = !zll.val<Goldilocks>
!xref = !zstruct.ref<!xval>
!struct_XA = !zstruct.struct<"XA", <foo:!xref, bar:!xref>>
!struct_XA_8 = !zstruct.array<!struct_XA, 8>

func.func @get_xfoo(%arg : !struct_XA_8) -> !xval {
  // CHECK-LABEL: @get_xfoo
  // CHECK: %0 = zll.slice %arg0, 2, 2 : <16, mutable, <Goldilocks>>
  %c1 = arith.constant 1 : index
  %elementref = zstruct.subscript %arg[index %c1] : (!struct_XA_8) -> !struct_XA
  // CHECK-NEXT: %1 = zll.slice %0, 1, 1 : <2, mutable, <Goldilocks>>
  %fieldref = zstruct.lookup %elementref ["bar"] : (!struct_XA) -> !xref
  // CHECK-NEXT: %2 = zll.get %1[0] back 0 : <1, mutable, <Goldilocks>>
  %ret = zstruct.load %fieldref : (!xref) -> !xval
  // CHECK-NEXT: return %2 : !zll.val<Goldilocks>
  return %ret : !xval
}

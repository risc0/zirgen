// RUN: zirgen-opt --canonicalize %s | FileCheck %s

//CHECK-DAG: !zstruct$A = !zstruct.struct<A, <foo: !zstruct.ref>>
//CHECK-DAG: !zstruct$Q = !zstruct.struct<Q, <foo: f32, bar: f32>>
//CHECK-DAG: !zstruct$X = !zstruct.struct<X, <a: f32>>
//CHECK-DAG: !zstruct$Y = !zstruct.struct<Y, <foo: f32, bar: f32>>
//CHECK-DAG: !zstruct$Z = !zstruct.struct<Z, <foo: f32, bar: f32>>
//CHECK-DAG: !zstruct$B = !zstruct.struct<B, <bar: !zstruct$A, baz: !zunion$A>>
//CHECK-DAG: !zunion$A = !zstruct.union<A, <foo: !zstruct.ref>>

func.func @struct_syntax_1(%arg : !zstruct.struct<X, <a:f32>>) -> !zstruct.struct<X, <a:f32>> {
  // CHECK-LABEL: @struct_syntax_1
  // CHECK-NEXT: return %arg0 : !zstruct$X
  return %arg: !zstruct.struct<"X", <a:f32>>
}

func.func @struct_syntax_2(%arg : !zstruct.struct<Y, <foo:f32, bar:f32>>) -> !zstruct.struct<"Y", <foo:f32, bar:f32>> {
  // CHECK-LABEL: @struct_syntax_2
  // CHECK-NEXT: return %arg0 : !zstruct$Y
  return %arg: !zstruct.struct<Y, <foo:f32, bar: f32>>
}

func.func @struct_syntax_3(%arg : !zstruct.struct<Z, <foo: f32, bar:f32>>) -> !zstruct.struct<Z, <foo: f32, bar:f32>> {
  // CHECK-LABEL: @struct_syntax_3
  // CHECK-NEXT: return %arg0 : !zstruct$Z
  return %arg: !zstruct.struct<Z, <foo: f32, bar: f32>>
}

func.func @struct_syntax_4(%arg: !zstruct.struct<Q, <foo:f32, bar: f32>>) -> !zstruct.struct<Q, <foo: f32, bar: f32>> {
  // CHECK-LABEL: @struct_syntax_4
  // CHECK-NEXT: return %arg0 : !zstruct$Q
  return %arg: !zstruct.struct<Q, <foo:f32, bar: f32>>
}

func.func @union_syntax_1(%arg : !zstruct.union<"A0A", <foo:f32>>) -> !zstruct.union<"A0A", <foo:f32>> {
  // CHECK-LABEL: @union_syntax_1
  // CHECK-NEXT: return %arg0 : !zunion$A0A
  return %arg: !zstruct.union<"A0A", <foo:f32>>
}

func.func @union_syntax_2(%arg : !zstruct.union<"v c r", <foo:f32, bar:f32>>) -> !zstruct.union<"v c r", <foo:f32, bar:f32>> {
  // CHECK-LABEL: @union_syntax_2
  // CHECK-NEXT: return %arg0 : !zunion$v_c_r
  return %arg: !zstruct.union<"v c r", <foo:f32, bar: f32>>
}

!val = !zll.val<BabyBear>
!ref_to_val = !zstruct.ref<!val>
!struct_A = !zstruct.struct<"A", <foo:!ref_to_val>>
!union_A = !zstruct.union<"A", <foo:!ref_to_val>>
!struct_B = !zstruct.struct<"B", <bar:!struct_A, baz:!union_A>>
!union_B = !zstruct.union<"B", <bar:!struct_A, baz:!union_A>>

// load ref -> val
func.func @load_ref(%arg : !ref_to_val) -> !val {
  // CHECK-LABEL: @load_ref
  // CHECK: %0 = zstruct.load %arg0 back %c0 : (!zstruct.ref) -> !zll.val<BabyBear>
  // CHECK-NEXT: return %0 : !zll.val<BabyBear>
  %c0 = arith.constant 0 : index
  %ret = zstruct.load %arg back %c0 : (!ref_to_val)  -> !val
  return %ret : !val
}

// store val -> ref
func.func @store_ref(%val : !val, %ref : !ref_to_val) {
  // CHECK-LABEL: @store_ref
  // CHECK-NEXT: zstruct.store %arg1, %arg0 : (!zll.val<BabyBear>) -> !zstruct.ref
  zstruct.store %ref, %val : (!val) -> !ref_to_val
  return
}

// load ref -> val, load ref -> val, add vals, return val
func.func @load_and_sum(%l_ref : !ref_to_val, %r_ref : !ref_to_val) -> !val {
  // CHECK-LABEL: @load_and_sum
  // CHECK: %0 = zstruct.load %arg0 back %c0 : (!zstruct.ref) -> !zll.val<BabyBear>
  // CHECK-NEXT: %1 = zstruct.load %arg1 back %c0 : (!zstruct.ref) -> !zll.val<BabyBear>
  // CHECK-NEXT: %2 = zll.add %0 : <BabyBear>, %1 : <BabyBear>
  // CHECK-NEXT: return %2 : !zll.val<BabyBear>
  %c0 = arith.constant 0 : index
  %l_val = zstruct.load %l_ref back %c0 : (!ref_to_val) -> !val
  %r_val = zstruct.load %r_ref back %c0 : (!ref_to_val) -> !val
  %sum = zll.add %l_val : !val, %r_val : !val
  return %sum : !val
}

// load & store (copy)
func.func @load_and_store(%src : !ref_to_val, %dst : !ref_to_val) {
  // CHECK-LABEL: @load_and_store
  // CHECK: %0 = zstruct.load %arg0 back %c0 : (!zstruct.ref) -> !zll.val<BabyBear>
  // CHECK-NEXT: zstruct.store %arg1, %0 : (!zll.val<BabyBear>) -> !zstruct.ref
  %c0 = arith.constant 0 : index
  %val = zstruct.load %src back %c0 : (!ref_to_val) -> !val
  zstruct.store %dst, %val : (!val) -> !ref_to_val
  return
}

// lookup struct -> ref
func.func @lookup_struct_ref(%arg : !struct_A) -> !ref_to_val {
  // CHECK-LABEL: @lookup_struct_ref
  // CHECK-NEXT: %0 = zstruct.lookup %arg0["foo"] : (!zstruct$A) -> !zstruct.ref
  // CHECK-NEXT: return %0 : !zstruct.ref
  %ret = zstruct.lookup %arg ["foo"] : (!struct_A) -> !ref_to_val
  return %ret : !ref_to_val
}

// lookup struct -> val
func.func @lookup_struct_val(%arg : !struct_A) -> !val {
  // CHECK-LABEL: @lookup_struct_val
  // CHECK: %0 = zstruct.lookup %arg0["foo"] : (!zstruct$A) -> !zstruct.ref
  // CHECK-NEXT: %1 = zstruct.load %0 back %c0 : (!zstruct.ref) -> !zll.val<BabyBear>
  // CHECK-NEXT: return %1 : !zll.val<BabyBear>
  %c0 = arith.constant 0 : index
  %ref = zstruct.lookup %arg ["foo"] : (!struct_A) -> !ref_to_val
  %val = zstruct.load %ref back %c0 : (!ref_to_val) -> !val
  return %val : !val
}

// lookup union -> ref
func.func @lookup_union_ref(%arg : !union_A) -> !ref_to_val {
  // CHECK-LABEL: @lookup_union_ref
  // CHECK-NEXT: %0 = zstruct.lookup %arg0["foo"] : (!zunion$A) -> !zstruct.ref
  // CHECK-NEXT: return %0 : !zstruct.ref
  %ret = zstruct.lookup %arg ["foo"] : (!union_A) -> !ref_to_val
  return %ret : !ref_to_val
}

// lookup union -> val
func.func @lookup_union_val(%arg : !union_A) -> !val {
  // CHECK-LABEL: @lookup_union_val
  // CHECK: %0 = zstruct.lookup %arg0["foo"] : (!zunion$A) -> !zstruct.ref
  // CHECK-NEXT: %1 = zstruct.load %0 back %c0 : (!zstruct.ref) -> !zll.val<BabyBear>
  // CHECK-NEXT: return %1 : !zll.val<BabyBear>
  %c0 = arith.constant 0 : index
  %ref = zstruct.lookup %arg ["foo"] : (!union_A) -> !ref_to_val
  %val = zstruct.load %ref back %c0 : (!ref_to_val) -> !val
  return %val : !val
}

func.func @lookup_struct_struct(%arg : !struct_B) -> !struct_A {
  // CHECK-LABEL: @lookup_struct_struct
  // CHECK-NEXT: %0 = zstruct.lookup %arg0["bar"] : (!zstruct$B) -> !zstruct$A
  // CHECK-NEXT: return %0 : !zstruct$A
  %ret = zstruct.lookup %arg ["bar"] : (!struct_B) -> !struct_A
  return %ret : !struct_A
}

func.func @lookup_struct_union(%arg : !struct_B) -> !union_A {
  // CHECK-LABEL: @lookup_struct_union
  // CHECK-NEXT: %0 = zstruct.lookup %arg0["baz"] : (!zstruct$B) -> !zunion$A
  // CHECK-NEXT: return %0 : !zunion$A
  %ret = zstruct.lookup %arg ["baz"] : (!struct_B) -> !union_A
  return %ret : !union_A
}

func.func @lookup_union_struct(%arg : !union_B) -> !struct_A {
  // CHECK-LABEL: @lookup_union_struct
  // CHECK-NEXT: %0 = zstruct.lookup %arg0["bar"] : (!zunion$B) -> !zstruct$A
  // CHECK-NEXT: return %0 : !zstruct$A
  %ret = zstruct.lookup %arg ["bar"] : (!union_B) -> !struct_A
  return %ret : !struct_A
}

func.func @lookup_union_union(%arg : !union_B) -> !union_A {
  // CHECK-LABEL: @lookup_union_union
  // CHECK-NEXT: %0 = zstruct.lookup %arg0["baz"] : (!zunion$B) -> !zunion$A
  // CHECK-NEXT: return %0 : !zunion$A
  %ret = zstruct.lookup %arg ["baz"] : (!union_B) -> !union_A
  return %ret : !union_A
}

!val_16 = !zstruct.array<!val, 16>
!struct_A_8 = !zstruct.array<!struct_A, 8>

func.func @lookup_array_head(%arg : !val_16) -> !val {
  // CHECK-LABEL: @lookup_array_head
  // CHECK-NEXT: %c0 = arith.constant 0 : index
  // CHECK-NEXT: %0 = zstruct.subscript %arg0[index %c0] : (!zstruct.array<!zll.val<BabyBear>, 16>) -> !zll.val<BabyBear>
  // CHECK-NEXT: return %0 : !zll.val<BabyBear>
  %c0 = arith.constant 0 : index
  %val = zstruct.subscript %arg[index %c0] : (!val_16) -> !val
  return %val : !val
}

func.func @sum_array_head_tail(%arg : !val_16) -> !val {
  // CHECK-LABEL: @sum_array_head_tail
  // CHECK-NEXT: %c0 = arith.constant 0 : index
  // CHECK-NEXT: %c15 = arith.constant 15 : index
  // CHECK-NEXT: %0 = zstruct.subscript %arg0[index %c0] : (!zstruct.array<!zll.val<BabyBear>, 16>) -> !zll.val<BabyBear>
  // CHECK-NEXT: %1 = zstruct.subscript %arg0[index %c15] : (!zstruct.array<!zll.val<BabyBear>, 16>) -> !zll.val<BabyBear>
  // CHECK-NEXT: %2 = zll.add %0 : <BabyBear>, %1 : <BabyBear>
  // CHECK-NEXT: return %2 : !zll.val<BabyBear>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 15 : index
  %head = zstruct.subscript %arg[index %c0]: (!val_16) -> !val
  %tail = zstruct.subscript %arg[index %c1]: (!val_16) -> !val
  %sum = zll.add %head : !val, %tail : !val
  return %sum : !val
}

func.func @lookup_array_struct_val(%arg : !struct_A_8) -> !val {
  // CHECK-LABEL: @lookup_array_struct_val
  // CHECK: %c2 = arith.constant 2 : index
  // CHECK-NEXT: %0 = zstruct.subscript %arg0[index %c2] : (!zstruct.array<!zstruct$A, 8>) -> !zstruct$A
  // CHECK-NEXT: %1 = zstruct.lookup %0["foo"] : (!zstruct$A) -> !zstruct.ref
  // CHECK-NEXT: %2 = zstruct.load %1 back %c0 : (!zstruct.ref) -> !zll.val<BabyBear>
  // CHECK-NEXT: return %2 : !zll.val<BabyBear>
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %struct = zstruct.subscript %arg[index %c2]: (!struct_A_8) -> !struct_A
  %ref = zstruct.lookup %struct["foo"]: (!struct_A) -> !ref_to_val
  %ret = zstruct.load %ref back %c0: (!ref_to_val) -> !val
  return %ret: !val
}

// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

include "mlir/IR/Constraints.td"

// "name", except with the first letter upercase
class detail_UcFirst<string name> {
  string ret = !toupper(!substr(name, 0, 1)) # !substr(name, 1);
}

// Snippets for one parameter, i.e. one element in either of the (ins) or (outs) dags.
class detail_ParamSnippets<
   TypeConstraint type,
   string name,
   int index> {
   string ucName = detail_UcFirst<name>.ret;

   // 1 if this parameter is variadic, 0 otherwise.  (Optional does not count as variadic for this purpose)
   bit variadic = !isa<Variadic>(type);

   // If non-empty, builderCall specifies how to create this type.  Consequently,
   // we will not need to pass the type as an argument since we can construct it ourselves.
   string builderCall = !if(!isa<Type>(type), !cast<Type>(type).builderCall, "");

   defvar paramTypes = "zirgen::Zhlt::detail::snippetGetIndex($rawTypes, $segmentSizes, "
             # index # ")";
   defvar paramValues = "zirgen::Zhlt::detail::snippetGetIndex($rawValues, $segmentSizes, "
             # index # ")";

   defvar getterName = "get" # ucName;

   // Declare and define accessors for the types of this param.
   code typeGetter = !if(variadic, [{
      ::llvm::ArrayRef<::mlir::Type> }] # getterName # "Types" # [{ () {
         return  }] # paramTypes # [{ ;
      }
    }], [{
      ::mlir::Type }] # getterName # "Type" # [{ () {
         auto types = }] # paramTypes # [{ ;
         if (types.empty()) return ::mlir::Type();
         assert(types.size() == 1);
         return types[0];
      }
    }]);

   // Declare and define accessors for the values of this param
   code valueGetter = !if(variadic, [{
       $cppValueRangeType }] # getterName # [{ () {
         return }] # paramValues # [{ ;
       }
     }], [{
       $cppValueType }] # getterName # [{ () {
         auto values = }] # paramValues # [{ ;
         if (values.empty()) return $cppValueType();
         assert(values.size() == 1);
         return values[0];
      }
    }]);

  // Generate code to call "setNameFn" to label values used, e.g.
  // by getAsm*Arguments.
  code setValueNames = !if(variadic, [{
     {
       size_t idx=0;
       for (::mlir::Value value : }] # paramValues # [{ )
          setNameFn(value, }] # "\"" # name # "\"" # [{ + std::to_string(idx++) );
     }
  }], [{
     for (::mlir::Value value : }] # paramValues # [{ )
       setNameFn(value, }] # "\"" # name # "\"" # [{ );
  }]) ;

  Pred pred = Concat<
     "llvm::all_of(" # paramTypes # ", [&](auto typePart) {\n  return ",
     SubstLeaves<"$_self", "typePart", type.predicate>,
     ";\n})">;

  dag builderTypeArg =
    !if(!empty(builderCall),
      !dag(ins,
        [CArg<!if(variadic, "::mlir::TypeRange",
            !if(!isa<TypeDef>(type),
                !cast<TypeDef>(type).cppType,
                type.cppType))>],
        [name]),
        // No builder call - don't make user specify type.
        (ins));
  code builderBuildType = !if(variadic, [{
      llvm::append_range($buildRawTypes, }] # name # [{);
      $buildSegmentSizes.push_back( }] # name # [{ .size());
  }],
      !if(!empty(builderCall), [{
        if ( }] # name # [{ ) {
          $buildRawTypes.push_back( }] # name # [{ );
          $buildSegmentSizes.push_back(1);
        } else {
          $buildSegmentSizes.push_back(0);
        }
      }], [{
        $buildRawTypes.push_back( }] # builderCall # [{ );
        $buildSegmentSizes.push_back(1);
      }])
  );

  dag builderValueArg = !dag(ins,
      [CArg<!if(variadic, "::mlir::ValueRange", "::mlir::Value")>],
      [name]);
  code builderBuildValue = !if(variadic, [{
    llvm::append_range($buildRawValues, }] # name # [{);
    $buildSegmentSizes.push_back( }] # name # [{ .size());
  }],
  [{
    if ( }] # name # [{ ) {
      $buildRawValues.push_back( }] # name # [{ );
      $buildSegmentSizes.push_back(1);
    } else {
      $buildSegmentSizes.push_back(0);
    }
  }]);
}

// Generated code snippets to generate accessors in the style of mlir's Operations
// for ZFuncOps and ZCallOps which want to name the inputs and results on
// their function types.
class NamedVariadicSnippets<
  // Parameters in the style that operations use, i.e. (ins ...) or (outs .... )
  dag params> {
  defvar paramSnips  = !foreach(idx, !range(!size(params)),
    detail_ParamSnippets<
      !getdagarg<TypeConstraint>(params, idx),
      !getdagname(params, idx),
      idx>);

  // Declare getters for the types of this param, typically for inclusion in extraClassDeclaration.
  //
  // Callers must substitute in the following:
  //   $rawTypes: The flattened TypeRange of raw types
  //   $segmentSizes: A DenseI32ArrayAttr of segment sizes in the style of AttrSizedOperandSegments.
  code typeGetters = !interleave(!foreach(snip, paramSnips, snip.typeGetter), "\n");

  // Declare getters for the values passed this param, typically for inclusion in extraClassDeclaration.
  //
  // Callers must substitute in the following:
  //   $rawValues: The flattened range of raw values, e.g. an OperandRange.
  //   $segmentSizes: A DenseI32ArrayAttr of segment sizes in the style of AttrSizedOperandSegments.
  //   $cppValueType: The type of values in $rawValue, e.g. "::mlir::OpOperand"
  //   $cppValueRangeType The type of range of values in $rawValue, e.g. " ::mlir::OperandRange"
  code valueGetters = !interleave(!foreach(snip, paramSnips, snip.valueGetter), "\n");

  // Emit code to provide names for all values, typically for inclusion in a OpAsmInterface
  // method like "getAsmBlockArgumentNames".   Assumes the OpAsmSetValueNameFn argument
  // is named "setNameFn".
  //
  // Callers must substitute in the following:
  //   $rawValues: The flattened ValueRange of raw values.
  //   $segmentSizes: A DenseI32ArrayAttr of segment sizes in the style of AttrSizedOperandSegments.
  code setValueNames = !interleave(!foreach(snip, paramSnips, snip.setValueNames), "\n");

  // Predicates to ensure all type constraints are satisfied.
  //
  // Callers must substitute in the following:
  //   $rawTypes: The flattened TypeRange of raw types
  //   $segmentSizes: A DenseI32ArrayAttr of segment sizes in the style of AttrSizedOperandSegments.
  Pred pred = And<
    !listconcat([
      CPred<"::zirgen::Zhlt::detail::isValidSegmentSizes($segmentSizes, $rawTypes.size(), "
            # !size(params) # ")">],
    !foreach(snip, paramSnips, snip.pred))>;

  // Argument declarations for a builder function taking types for these parameters.
  dag builderTypeArgs = !foldl(
      (ins),
      !foreach(snip, paramSnips, snip.builderTypeArg),
      total, rec, !con(total, rec));
  // Code for body of a builder function to gather the types for all the parameters.
  //
  // Callers must substitute in the following:
  //   $buildRawTypes: SmallVector<TypeRange> containing the list of types being built.
  //   $buildSegmentSizes: SmallVector<int32_t> containing the segment sizes for each argument.
  code builderBuildTypes = !interleave(!foreach(snip, paramSnips, snip.builderBuildType), "\n");

  // Argument declarations for a builder function taking values for these parameters.
  dag builderValueArgs = !foldl(
      (ins),
      !foreach(snip, paramSnips, snip.builderValueArg),
      total, rec, !con(total, rec));
  // Code for body of a builder function to gather the values for all the parameters.
  //
  // Callers must substitute in the following:
  //   $buildRawValues: SmallVector<Value> containing the list of types being built.
  //   $buildSegmentSizes: SmallVector<int32_t> containing the segment sizes for each argument.
  code builderBuildValues = !interleave(!foreach(snip, paramSnips, snip.builderBuildValue), "\n");
}

// Performs the given replacements on a NamedVariadicSnippets
class NamedVariadicSubstSnippets<NamedVariadicSnippets orig,
  string rawTypes,
  string rawValues,
  string segmentSizes,
  string buildRawTypes,
  string buildRawValues,
  string buildSegmentSizes,
  string cppValueType,
  string cppValueRangeType
> {
  code typeGetters =
       !subst("$rawTypes", rawTypes,
       !subst("$segmentSizes", segmentSizes,
       orig.typeGetters));
  code valueGetters =
       !subst("$rawValues", rawValues,
       !subst("$segmentSizes", segmentSizes,
       !subst("$cppValueType", cppValueType,
       !subst("$cppValueRangeType", cppValueRangeType,
       orig.valueGetters))));
  code setValueNames =
       !subst("$rawValues", rawValues,
       !subst("$segmentSizes", segmentSizes,
       orig.setValueNames));
  code builderBuildTypes =
       !subst("$buildRawTypes", buildRawTypes,
       !subst("$buildSegmentSizes", buildSegmentSizes,
       orig.builderBuildTypes));
  code builderBuildValues =
       !subst("$buildRawValues", buildRawValues,
       !subst("$buildSegmentSizes", buildSegmentSizes,
       orig.builderBuildValues));
  Pred pred =
       SubstLeaves<"$rawTypes", rawTypes,
       SubstLeaves<"$segmentSizes",  segmentSizes,
       orig.pred>>;
}


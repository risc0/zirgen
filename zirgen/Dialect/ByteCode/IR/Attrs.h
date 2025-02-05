

class IntKindAttr : public Attribute {
public:
  using Attribute::Attribute;
  using ValueType = bool;

  static BoolAttr get(MLIRContext* context, bool value);

  /// Enable conversion to IntegerAttr and its interfaces. This uses conversion
  /// vs. inheritance to avoid bringing in all of IntegerAttrs methods.
  operator IntegerAttr() const { return IntegerAttr(impl); }
  operator TypedAttr() const { return IntegerAttr(impl); }

  /// Return the boolean value of this attribute.
  bool getValue() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Attribute attr);
};

// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

// Circuit-agnostic code for toy example circuits

#include <functional>

// In a real zero-knowledge proof, "Val" would be a field element.
// But "int" is easier to deal with when testing the circuit code generation.
#define SET_FIELD(FIELD) /* */
using Val4 = std::array<int, 4>;
using Fp4 = Val4;
using Val = int;
using ExtVal = Val4;
using String = std::string;
using PolyMix = Val4;
using TapGroupName = const char*;

size_t to_size_t(Val x) {
  return x;
}

Val4 operator+(const Val4& lhs, const Val4& rhs) {
  Val4 result;
  for (size_t i = 0; i != 4; i++) {
    result[i] = lhs[i] + rhs[i];
  }
  return result;
}
Val4 operator-(const Val4& lhs, const Val4& rhs) {
  Val4 result;
  for (size_t i = 0; i != 4; i++) {
    result[i] = lhs[i] + rhs[i];
  }
  return result;
}
Val4 operator*(const Val4& lhs, const Val4& rhs) {
  Val4 result;
  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      if (i + j < 4) {
        result[i + j] += lhs[i] * rhs[j];
      } else {
        assert(lhs[i] * rhs[j] == 0 && "extension field doesn't wrap multiplication");
      }
    }
  }
  return result;
}

// In a real zero-knowledge proof, "Reg" would be a reference of some
// sort into an execution trace.  However, for testing code generation
// we don't have to supply a full implementation.
struct Reg {
  size_t index;
};

bool operator==(const Reg& lhs, const Reg& rhs) {
  return lhs.index == rhs.index;
}

#define MAKE_REF(INDEX)                                                                            \
  Reg { .index = INDEX }
#define MAKE_VAL(VAL) Val(VAL)
#define MAKE_VAL_EXT(...) Val4({__VA_ARGS__})
#define LOAD(REF, BACK) Val(REF.buffer.at(REF.layout->index))
#define LOAD_AS_EXT(REF, BACK)                                                                     \
  Val4 { LOAD(REF, BACK), 0, 0, 0 }
#define LOAD_EXT(REF, BACK)                                                                        \
  Val4 {                                                                                           \
    REF.buffer.at(REF.layout->index + 0), REF.buffer.at(REF.layout->index + 1),                    \
        REF.buffer.at(REF.layout->index + 2), REF.buffer.at(REF.layout->index + 3),                \
  }
#define STORE(REF, VAL) REF.buffer.at(REF.layout->index) = VAL
#define DEFINE_LAYOUT_BUFFER(CONST, BUFFER) /* */
#define LAYOUT_LOOKUP(LAYOUT, FIELD) LAYOUT.map([](auto layout) { return &layout->FIELD; })
#define LAYOUT_SUBSCRIPT(LAYOUT, INDEX) LAYOUT.map([](auto layout) { return &layout->at(INDEX); })

#define GET_BUFFER(CTX, BUFFER) CTX.BUFFER
#define BIND_LAYOUT(LAYOUT, BUFFER) BoundLayout(&LAYOUT, BUFFER)

template <typename Layout> struct BoundLayout {
  const Layout* layout;
  std::vector<uint32_t>& buffer;

  BoundLayout(const Layout* layout, std::vector<uint32_t>& buffer)
      : layout(layout), buffer(buffer) {}

  template <typename F> auto map(F f) const {
    using NewLayoutType = typename std::remove_cv_t<
        typename std::remove_pointer_t<typename std::invoke_result<F, const Layout*>::type>>;
    return BoundLayout<NewLayoutType>(f(layout), buffer);
  }
};

template <typename T> auto load(T& buf, Reg ref, size_t distance = 0) {
  // TODO: Implement backs
  assert(distance == 0);
  return buf.data.at(ref.index);
}

#define EQZ(VAL, LOC) eqz(VAL, LOC)

void eqz(int x, const char* loc = nullptr) {
  assert(x == 0);
}
void eqz(Val4 x, const char* loc = nullptr) {
  assert(std::all_of(x.begin(), x.end(), [](auto elem) { return elem == 0; }));
}

template <typename T, size_t N> T& subscript(std::array<T, N>& array, size_t index) {
  return array[index];
}

// TODO: Implement stub mixstate a little better
struct MixState {};
MixState trivialConstraint() {
  return {};
}
MixState andEqz(Val4 mix, MixState in, Val val) {
  eqz(val);
  return in;
}
MixState andEqzExt(Val4 mix, MixState in, Val4 val) {
  eqz(val);
  return in;
}
MixState andCond(MixState in, Val cond, MixState inner) {
  // TODO: implement actual mixing
  return in;
}
MixState andCondExt(MixState in, Val4 cond, MixState inner) {
  // TODO: implement actual mixing
  return in;
}

template <typename C> auto subscript(size_t index, C container) -> typename C::value_type {
  return container[index];
}

template <typename T, typename U, size_t N>
std::array<U, N> map(std::array<T, N>& array, std::function<U(T)> f) {
  std::array<U, N> output;
  for (size_t i = 0; i < N; i++) {
    output[i] = f(array[i]);
  }
  return output;
}

template <typename U, typename T, typename Layout, size_t N>
std::array<U, N> map(std::array<T, N>& valueArray,
                     const BoundLayout<std::array<Layout, N>>& layoutArray,
                     std::function<U(T, BoundLayout<Layout>)> f) {
  std::array<U, N> output;
  for (size_t i = 0; i < N; i++) {
    output[i] =
        f(valueArray[i], layoutArray.map([i](auto layoutArray) { return &layoutArray->at(i); }));
  }
  return output;
}

template <typename T, typename U, size_t N>
U reduce(std::array<T, N>& array, U init, std::function<U(U, T)> f) {
  U output = init;
  for (T& element : array) {
    output = f(output, element);
  }
  return output;
}

Val isz(Val x) {
  return (x == 0) ? 1 : 0;
}

Val inRange(Val low, Val mid, Val high) {
  assert(low <= high);
  return (low <= mid && mid < high) ? 1 : 0;
}

struct Tap {
  const char* buffer;
  size_t index;
  size_t back;
};

#define MAKE_TAP(BUF, INDEX, BACK)                                                                 \
  Tap { .buffer = #BUF, .index = INDEX, .back = BACK }

#define INVOKE_EXTERN(CTX, NAME, ...) externs.NAME(__VA_ARGS__)

static void log_impl(std::string& format, const Val* x) {
  size_t argNum = 0;
  auto nextArg = [&]() { return x[argNum++]; };
  const char* p = format.c_str();
  while (*p) {
    if (*p == '%') {
      p++;
      int len = 0;
      while (*p >= '0' && *p <= '9') {
        len *= 10;
        len += *p - '0';
        p++;
      }
      if (*p == '%') {
        printf("%%");
        p++;
      } else if (*p == 'x') {
        printf("%x", nextArg());
        p++;
      } else if (*p == 'u') {
        printf("%u", nextArg());
        p++;
      } else if (*p == 'w' || *p == 'e') {
        uint64_t vals[4];
        bool isU32 = true;
        uint32_t u32val = 0;
        for (auto& val : vals) {
          val = nextArg();
          if (val > 255) {
            isU32 = false;
          } else {
            u32val >>= 8;
            u32val |= val << 24;
          }
        }
        if (*p == 'e') {
          isU32 = false;
        }

        if (isU32) {
          printf("%*x", len, u32val);
        } else {
          printf("[%llu, %llu, %llu, %llu]", vals[0], vals[1], vals[2], vals[3]);
        }
        p++;
      }
    } else {
      printf("%c", *p);
      p++;
    }
  }
  printf("\n");
}

using Index = size_t;

// This code is automatically generated

#include "impl.h"

using namespace risc0;

namespace circuit::keccak {

void CircuitImpl::add_taps() {
  tapSet = {{
                // groups
                {// group 0
                 {{0, 1, 0, {0, 1}}, {1, 1, 2, {0, 1}}, {2, 1, 4, {0, 1}}, {3, 1, 6, {0, 1}}}}
                // group
                ,
                {                  // group 1
                 {{0, 0, 8, {0}}}} // group
                ,
                {// group 2
                 {{0, 0, 9, {0}},
                  {1, 0, 10, {0}},
                  {2, 0, 11, {0}},
                  {3, 0, 12, {0}},
                  {4, 0, 13, {0}},
                  {5, 0, 14, {0}},
                  {6, 0, 15, {0}},
                  {7, 0, 16, {0}},
                  {8, 0, 17, {0}},
                  {9, 0, 18, {0}},
                  {10, 0, 19, {0}},
                  {11, 0, 20, {0}},
                  {12, 0, 21, {0}}}} // group
            },                       // groups
            {                        // combos
             {0, {0}},
             {1, {0, 1}}}, // combos
            22};
}
} // namespace circuit::keccak

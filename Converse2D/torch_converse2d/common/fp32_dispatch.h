#pragma once
#include <ATen/Dispatch.h>
// Instantiate only FP32 kernels, preserving their original arithmetic.
#define CONVERSE_DISPATCH_FP32(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__))

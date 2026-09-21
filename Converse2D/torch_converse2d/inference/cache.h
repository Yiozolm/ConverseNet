#pragma once
#include <ATen/ATen.h>
#include <vector>
void clear_fb_cache();
bool supports_cuda_graphs();
void begin_graph_cache();
std::vector<at::Tensor> end_graph_cache();

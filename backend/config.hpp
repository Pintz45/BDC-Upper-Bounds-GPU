#pragma once
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>
#include <fstream>
#include <iostream>

constexpr int MAX_N = 33; 

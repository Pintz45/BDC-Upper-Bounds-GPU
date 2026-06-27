#pragma once


typedef double Float;
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
typedef uint32_t Int;

std::vector<std::vector<Float> > load_array_from_file(FILE* in_file);
void write_array_to_file(FILE* out_file, const std::vector<std::vector<Float> >& array);

std::vector<Float> load_1d_array_from_file(FILE* in_file);
void write_1d_array_to_file(FILE* out_file, const std::vector<Float>& array);

/*
Tries to open the requested file with the requested mode.
On failure prints a message to the stderr and exits.
*/
FILE* try_to_open_file(const char* filename, const char* mode);
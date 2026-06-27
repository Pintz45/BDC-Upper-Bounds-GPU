#include "utils.h"
#include <cassert>

std::vector<std::vector<Float>> load_array_from_file(FILE* in_file){
	uint32_t shape[2];
	assert(fread(shape, sizeof(uint32_t), 2, in_file) == 2);
	printf("shape = %u, %u\n", shape[0], shape[1]);

	std::vector<std::vector<Float>> res; res.resize(shape[0]);
	for(auto riter = res.begin(); riter != res.end(); ++riter){
		riter -> resize(shape[1]);
		size_t len = fread(riter -> data(), sizeof(Float), shape[1], in_file);
		assert(len == shape[1]);
	}
	return res;
}

void write_array_to_file(FILE* out_file, const std::vector<std::vector<Float>>& array){
	uint32_t shape[2];
	shape[0] = (uint32_t) array.size();
	shape[1] = (uint32_t) array[0].size();
	fwrite(shape, sizeof(*shape), 2, out_file);

	for(auto iter = array.begin(); iter != array.end(); ++iter){
		assert(fwrite(iter -> data(), sizeof(Float), shape[1], out_file) == shape[1]);
	}
}


std::vector<Float> load_1d_array_from_file(FILE* in_file){
	uint32_t shape;
	assert(fread(&shape, sizeof(uint32_t), 1, in_file) == 1);

	std::vector<Float> res; res.resize(shape);
	assert(fread(res.data(), sizeof(Float), shape, in_file) == shape);
	return res;
}

void write_1d_array_to_file(FILE* out_file, const std::vector<Float>& array){
	uint32_t shape = (uint32_t) array.size();
	fwrite(&shape, sizeof(shape), 1, out_file);
	assert(fwrite(array.data(), sizeof(Float), shape, out_file) == shape);
}


FILE* try_to_open_file(const char* filename, const char* mode){
	FILE* fptr = fopen(filename, mode);
	if (fptr == NULL)
	{
		fprintf(stderr, "Error: failed to open file %s.\n", filename);
		exit(2);
	}
	return fptr;
}
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include "utils.h"

inline std::string get_cache_filename(size_t n, size_t k){
	const std::string BASE_PATH = ".";
	const std::string PATH_SEPARATOR = "/";
	const std::string CACHE_FOLDER_NAME = "transition_counts";
	const std::string FORMAT = "cache_";
	const std::string FORMAT_SEPARATOR = "_";
	return BASE_PATH + PATH_SEPARATOR + CACHE_FOLDER_NAME + PATH_SEPARATOR + FORMAT + std::to_string(n) + FORMAT_SEPARATOR + std::to_string(k);
}


inline void _save_data_to_cache_file(FILE* cache_file, const std::vector<uint32_t>& data){
	fwrite(data.data(), sizeof(uint32_t), data.size(), cache_file);
}

inline void save_data_to_cache_file(size_t n, size_t k, const std::vector<uint32_t>& data){
	std::string filename = get_cache_filename(n, k);
	FILE* cache_file = try_to_open_file(filename.data(), "wb");
	_save_data_to_cache_file(cache_file, data);
	fclose(cache_file);
}


inline std::vector<uint32_t> _load_data_from_cache_file(FILE* cache_file){
	std::vector<uint32_t> v;
	uint32_t buf[1024];
	while (size_t len = fread(buf, sizeof(uint32_t), 1024, cache_file)) {
		v.insert(v.end(), buf, buf + len);
	}
	return v;
}

inline std::vector<uint32_t> load_data_from_cache_file(size_t n, size_t k){
	std::string filename = get_cache_filename(n, k);
	FILE* cache_file = try_to_open_file(filename.data(), "rb");
	auto res = _load_data_from_cache_file(cache_file);
	fclose(cache_file);
	return res;
}



#include <bits/stdc++.h>
#include "cache_io.h"
using namespace std;

// Generate all binary codewords of length n
vector<string> get_all_bit_codewords(size_t n) {
    vector<string> result;
    size_t total = 1ULL << n;  // 2^n codewords
    result.reserve(total);

    for (size_t mask = 0; mask < total; ++mask) {
        string s;
        s.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            s.push_back(((mask >> (n - i - 1)) & 1) ? '1' : '0');
        }
        result.push_back(s);
    }
    return result;
}

// number of subsequences of x equal to y
uint32_t get_num_transition_possibilities(const string &x, const string &y) {
    size_t n = x.size();
    size_t k = y.size();

    // dp[i][j] = number of ways to form prefix of length j of y from prefix of length i of x
    vector<vector<uint32_t>> dp(n + 1, vector<uint32_t>(k + 1, 0));

    // Base case: y empty → 1 way
    for (size_t i = 0; i <= n; ++i) dp[i][0] = 1;

    // Fill table
    for (size_t i = 1; i <= n; ++i) {
        for (size_t j = 1; j <= k; ++j) {
            dp[i][j] = dp[i - 1][j];  // skip x[i-1]
            if (x[i - 1] == y[j - 1]) {
                dp[i][j] += dp[i - 1][j - 1];  // match
            }
        }
    }

    return dp[n][k];
}

int main(int argc, char const *argv[])
{
	if (argc != 2)
	{
		fprintf(stderr, "Usage %s n k\n", argv[0]);
		return 1;
	}

	size_t n = atol(argv[1]);

	for(int n_ = 0; n_ <= n; n_++) {
        for(int k_ = 0; k_ < n_; k_++) {
            vector<string> transmitted_codewords = get_all_bit_codewords(n_);
            vector<string> receveived_codewords = get_all_bit_codewords(k_);


            std::vector<uint32_t> data; data.reserve(transmitted_codewords.size() * receveived_codewords.size());

            for (int i = 0; i < transmitted_codewords.size(); ++i)
            {
                for (int j = 0; j < receveived_codewords.size(); ++j)
                {

                    data.push_back(get_num_transition_possibilities(transmitted_codewords[i], receveived_codewords[j]));
                }
            }
            save_data_to_cache_file(n_, k_, data);
        }
	}

	return 0;
}

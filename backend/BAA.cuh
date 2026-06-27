#pragma once
#include "config.hpp"
#include "cache_io.h"

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err = call;                                                     \
        if (err != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(err));                                       \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while (0)


__host__ __device__ uint32_t nCk(size_t n, size_t k);

// Allocate GPU memory for transition probability lookup tables.
// Loads cached transition probabilities from disk and transfers them to device memory.
// Depending on hardware loading may be successful or not
void load_gpu_cache(size_t in_len, size_t out_len);

// Precompute binomial normalizers and their logarithms, and upload to GPU.
void compute_normalizers(size_t in_len, size_t out_len);

// Precompute log values [0..total_logs) and upload to GPU
void load_log_counts(size_t total_logs);

void load_cache(size_t in_len, size_t out_len);

// =================== inline helpers ===================
__device__   double device_log(double x);

/*
Fast reversing for 32-bit type integers
*/
__host__ __device__ uint32_t reverse32(uint32_t x);

/*
Complement the first k bits (right to left)
*/
__host__ __device__ uint32_t complement_k(uint32_t y, int k);

/*
Reverse only the first k bits (right to left)
*/
__host__ __device__ uint32_t reverse_k(uint32_t y, int k);

/*
Compute the canonical representative as the smallest element of its respective orbit (reversal and complement transformations).
*/

__host__ __device__ uint32_t rep(uint32_t x, int k);

/*
Compute orbit(x) size
*/
__host__ __device__ double orbits(uint32_t x, int k);

/*
Compute the canonical representative of x and store the orbit(x) size in "orbit_sz"
*/
__host__ __device__ uint32_t rep_and_orbit(uint32_t x, int k, double* orbit_sz);

// =================== thrust functor ===================
struct index_dependent_transform {
    double* data;
    uint32_t* trans_canonicals;
    const size_t k;
    __host__ __device__
    index_dependent_transform(double* d, uint32_t* trans, const size_t k_)
        : data(d), trans_canonicals(trans), k(k_) {}
    __host__ __device__
    double operator()(const int& i) const {
        double val = data[i];
        double multiplier = orbits(trans_canonicals[i], k);
        return multiplier * val;
    }
};

// =================== Shared memory struct ===================
// 1024 is the block_size. If we change the block_size to other power of two, sdata's size may also be changed
struct SharedMem {
    alignas(8) double sdata[1024];
    uint8_t nextpos[33][2];
    uint8_t _padding[2];
    alignas(4) uint32_t cnt[33][33];
};

// =================== DP helpers ===================
__device__ void build_next_pos(uint32_t trans_num, uint8_t n, uint32_t b, SharedMem* shmem_ptr);

__device__ void build_cnt(uint32_t trans_num, uint8_t n, uint8_t k, SharedMem* shmem_ptr);

// Builds the three needed tables: nextpos for 0 and 1, then cnt
__device__ void fill_DP_tables(uint32_t trans_num, uint8_t n, uint8_t k, SharedMem* shmem_ptr);

// Unranking: reconstruct the j-th subsequence of length k according to an enumeration
__device__ uint32_t unrank_k(uint32_t trans_num, uint8_t n, uint8_t k, uint32_t j, SharedMem* shmem_ptr);

// =================== probability helpers ===================

/*
 This function computes the total number of subsequences of length "k"
 from a given transition word "trans_num" of length "n"
 that equal a target subsequence "rec_num".

 Instead of recomputing everything via DP, we split the problem into two halves
 (divide-and-conquer) and use smaller precomputed lookup tables (d_lookup_table)
 to quickly get the number of valid subsequences in each half.

 It returns the total number of ways subsequence rec_num appears in trans_num
*/
 __device__ uint32_t total_transitions(uint32_t trans_num, uint32_t rec_num, uint8_t n, uint8_t k);


 // ============================================================================
// 1) Contribution to denominator D(rec)
// ============================================================================
//
// Used when building the denominators of the BAA update:
//      D(rec) = Σ_x Q(x) P(y|x)
//
// This function contributes Q(x) * P(y|x) for a given (trans, rec) pair.
//
__device__ double denom_term(uint32_t trans_num,uint32_t rec_num,  double Q_x, uint8_t n, uint8_t k);


// =======================================================
// Log_W contribution for full BAA denominator update
// ========================================================
//
// Computes the contribution of a single (trans_num, rec_num) pair of the D(y) terms
// 
//
// Formula:
//    contrib = P(rec | trans) * ( log P(rec | trans) + log Q_(trans) - log_D(rec) )
__device__ double log_W_term(uint32_t trans_num, uint32_t rec_num, double log_Q_k, double* d_log_W_jk_den,
                                        uint8_t n, uint8_t k);


__device__ double KL_term(uint32_t trans_num, uint32_t rec_num,
                                  const double* d_log_W_jk_den, uint8_t n, uint8_t k);


// =================== BAA kernels ===================

/*
 ============================================================================
 CUDA kernel: compute_Wjk_den_kernel
 ============================================================================

 This kernel computes the denominator terms D(rec) for the Blahut–Arimoto
 algorithm:

     D(rec) = Σ_x Q(x) P(y=rec | x=trans)

 - Each block corresponds to one transmitted canonical word "trans_num".
 - Threads within the block iterate over possible received words "rec_num".
 - Results are accumulated into the global denominator array "d_denoms".

 INPUTS:
  d_Qi          : device array of input distribution Q(x)
  d_denoms      : device array [num_received] for accumulated denominators
   d_trans_nums  : list of transmitted canonical words (one per block)
   n             : transmitted word length
   k             : received word length

 NOTES:
   - Uses shared memory to hold DP tables for transitions.
   - Orbit size factors are used to account for equivalence under symmetries.
   - Atomic adds ensure safe accumulation across threads.

 ============================================================================
 */
__global__ void compute_den_kernel(
    double*      d_Qi,          // Q distribution on device
    double*            d_denoms,      // output array [num_received]
    uint32_t* d_trans_nums,
    uint8_t           n,             // length of transmitted codewords in bits
    uint8_t           k             // length of received codewords in bits
);

/**
 * ============================================================================
 * CUDA kernel: compute_log_alpha_gpu
 * ============================================================================
 *
 * 
 *   Computes the per-transmitted-codeword log(W(x)) values used in the
 *   Blahut–Arimoto algorithm.
 *
 *   Formula:
 *       log(W(x))= Σ_y  P(y|x) * [ log P(y|x) + log Q(x) - log W(y) ]
 *
 *   where:
 *     • Q(x) is the input distribution (logarithmic form provided here),
 *     • W(y) is the marginal probability of y (log form in d_log_den),
 *     • P(y|x) is the conditional channel probability.
 *
 *   • Each CUDA block corresponds to one transmitted canonical word x.
 *   • Threads inside the block enumerate received words y using
 *     dynamic-programming (DP) unranking tables in shared memory.
 *   • Each thread accumulates a partial sum for its strided subset of y’s.
 *   • A block-wide reduction aggregates results into α_x.
 *
 *   log_d_Q_k    : log(Q(x)) distribution on device
 *   d_trans_nums : transmitted canonical words
 *   d_log_den    : log D(y) denominators
 *   log_W   : output array, one log(W(x)) per transmitted word
 *   n            : input word length (bits)
 *   k            : output word length (bits)
 *
 *   • log_W_term() computes the per-(x,y) contribution:
 *         log_W_term = P(y|x) * ( log P(y|x) + log Q(x) - log W(y) )
 *   • Reduction pattern assumes blockDim.x is a power of two.
 *
 * ============================================================================
 */
__global__ void compute_log_W_gpu(
    double*    log_d_Q_k,
    uint32_t* d_trans_nums,
    double*    d_log_den,
    double*    log_W,
    uint8_t    n, uint8_t k
);

/**
 * ============================================================================
 * CUDA kernel: compute_KL_kernel
 * ============================================================================
 *
 * 
 *   Computes the per-transmitted-codeword KL divergence contributions in the
 *   Blahut–Arimoto algorithm. Specifically:
 *
 *       KL(x) = Σ_y  P(y|x) * ( log P(y|x) - log W(y) )
 *
 *   where:
 *     • P(y|x) is the conditional channel probability,
 *     • W(y)   is the marginal output distribution (in log form here).
 *
 * 
 *   • Each CUDA block corresponds to one transmitted canonical word x.
 *   • Threads enumerate subsets of possible received words y.
 *   • Each thread computes a partial sum of the KL contribution for its strided
 *     subset of y’s.
 *   • Shared-memory reduction accumulates the per-thread partials into KL(x).
 *
 * 
 *   d_trans_nums : [n_I] transmitted canonical words
 *   d_log_den    : [N_j] log W(y) denominators
 *   n            : input word length (bits)
 *   k            : output word length (bits)
 *
 * OUTPUTS
 *   d_KL         : [n_I] per-word KL(x) values
 *
 * NOTES
 *   • KL_term() computes the per-(x,y) contribution:
 *         KL_term = P(y|x) * ( log P(y|x) - log W(y) )
 *   • Reduction pattern assumes blockDim.x is a power of two.
 *
 * ============================================================================
 */
__global__ void compute_KL_kernel(
    const uint32_t* d_trans_nums,      // [n_I] device array of canonical x indices
    const double*   d_log_W_jk_den,     // [N_j] device: log W(y) for each y
    double*         d_KL,               // [n_I] output: KL(x) for each x
    uint8_t        n,                  // bit-length of x
    uint8_t        k                   // bit-length of y
);

__global__ void apply_log_kernel(double* d_array, size_t size);


/**
 * Perform one full Blahut–Arimoto iteration step on GPU.
 *
 * Inputs:
 *   n, k                 - Input/output codeword lengths
 *   Q_i                  - Current input distribution Q(x) [host]
 *   trans_canonicals     - Canonical transmitted codewords [host]
 *   received_canonicals  - Canonical received codewords [host] (unused here but kept for consistency)
 *
 * Output:
 *   new_Q - Updated input distribution after one BAA step [host]
 *
 * Steps:
 *   1) Compute log-denominators log D(y) for all received codewords
 *   2) Compute log W(x) for each transmitted codeword
 *   3) Normalize W(x) using log-sum-exp trick (to prevent underflow)
 *   4) Return normalized distribution
 */
std::vector<double> baa_iteration(std::vector<double>& Q_i,std::vector<uint32_t>& trans_canonicals, size_t n, size_t k);

/**
 * Compute all log denominators log W_jk(y) for the BAA step.
 *
 * For each received word y of length k:
 *   D(y) = ∑_x Q(x) P(y|x)
 *
 * This kernel accumulates denominators in parallel over transmitted
 * codewords and applies log() on the GPU.
 *
 * Inputs:
 *   Q_i              - Input distribution Q(x) 
 *   trans_canonicals - Canonical transmitted codewords 
 *   n, k             - Input/output word lengths
 *
 * Output:
 *   log_den      - Vector of log_ D(y) values 
 */
std::vector<double> compute_all_log_D(
    std::vector<double>& Q_i,
    std::vector<uint32_t>& trans_canonicals,
    size_t n, size_t k);



    /**
 * Compute all log(W) values for the BAA update step.
 *
 * For each transmitted codeword x:
 *   log W(x) = ∑_y P(y|x) * [ log(Q(x)) - log(D(y)) + ... ]
 *
 * This kernel:
 *   1. Uploads log D(y) denominators, Q(x), and trans_canonicals to GPU
 *   2. Converts Q(x) into log Q(x) on the device
 *   3. Launches compute_log_W_gpu to compute W(x) in log domain
 *   4. Copies log W(x) back to the host
 *
 * Inputs:
 *   n, k              - Input/output word lengths
 *   Q_i               - Current input distribution Q(x) [host]
 *   log_den      - Precomputed log denominators log D(y) [host]
 *   trans_canonical   - Canonical transmitted codewords [host]
 *
 * Output:
 *   h_log_W      - Vector of log W(x) values [host]
 */
std::vector<double> compute_all_log_W(
    std::vector<double>& Q_i,
    std::vector<double>& log_den,
    std::vector<uint32_t>& trans_canonical, size_t n, size_t k);


double compute_rate(
    std::vector<double>& log_den,
    std::vector<double>& Q_i,
    std::vector<uint32_t>& trans_canonical, uint8_t n, uint8_t k);

/**
 * Compute the maximum deviation between successive distributions Q and Q_before.
 *
 * The deviation is defined as:
 *     max_i [ log2(Q_i) - log2(Q_before_i) ]
 *
 * This is used as the convergence criterion in the Blahut–Arimoto iterations.
 */
double compute_max_deviation(const std::vector<double>& Q,
                             const std::vector<double>& Q_before);


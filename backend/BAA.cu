#include "BAA.cuh"

__device__ uint32_t*  d_flat_data        = nullptr;
__device__ uint32_t** d_lookup_table     = nullptr;
__device__ double**   d_normalizers      = nullptr;
__device__ double**   d_log_normalizers  = nullptr;
__device__ double*    d_log_counts       = nullptr ;

static double*   g_h_norm_blob     = nullptr;
static double*   g_h_log_norm_blob = nullptr;
static double**  g_d_norm_ptr      = nullptr;
static double**  g_d_log_norm_ptr  = nullptr;


__host__ __device__ uint32_t nCk(size_t n, size_t k) {
    if (k > n) return 0;
    if (k > n - k) k = n - k; // Take advantage of symmetry

    uint64_t res = 1; // Use 64-bit to avoid overflow
    for (size_t i = 1; i <= k; ++i) {
        res = res * (n - i + 1) / i;
    }
    return static_cast<uint32_t>(res);
}


void load_gpu_cache(size_t in_len, size_t out_len) {
    std::vector<uint32_t> flat_data;
    std::vector<size_t> offsets(MAX_N * MAX_N);

    // Check free memory
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    // Only allow up to 95% of total GPU memory
    size_t usable_mem = static_cast<size_t>(0.95 * free_mem);

    // Estimate required memory footprint
    size_t total_cache_bytes = 0;
    for (size_t n = 0; n <= (in_len + 1) / 2; ++n) {
        for (size_t k = 0; k < n; ++k) {
            total_cache_bytes += 1ull << (n + k);
        }
    }
    assert(usable_mem > total_cache_bytes && "Not enough GPU memory for cache");

    // Flatten all cached transition probabilities into one big array
    size_t offset = 0;
    for (size_t n = 0; n <= (in_len + 1) / 2; n++) {
        for (size_t k = 0; k < n; k++) {
            size_t idx = n * MAX_N + k;
            std::vector<uint32_t> trans_probs = load_data_from_cache_file(n, k);

            offsets[idx] = offset;
            flat_data.insert(flat_data.end(), trans_probs.begin(), trans_probs.end());
            offset += trans_probs.size();
        }
    }

    // Copy flattened transition data to GPU
    uint32_t* d_flat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flat, flat_data.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_flat, flat_data.data(),
                          flat_data.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // Build host lookup table (pointers into the flat array)
    uint32_t** h_lookup_table = new uint32_t*[MAX_N * MAX_N];
    for (size_t n = 0; n <= (in_len + 1) / 2; ++n) {
        for (size_t k = 0; k < n; ++k) {
            size_t idx = n * MAX_N + k;
            h_lookup_table[idx] = d_flat + offsets[idx];
        }
    }

    // Copy lookup table to GPU
    uint32_t** d_lookup = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lookup, MAX_N * MAX_N * sizeof(uint32_t*)));
    CUDA_CHECK(cudaMemcpy(d_lookup, h_lookup_table,
                          MAX_N * MAX_N * sizeof(uint32_t*),
                          cudaMemcpyHostToDevice));

    // Publish device pointers into device symbols
    CUDA_CHECK(cudaMemcpyToSymbol(d_flat_data, &d_flat, sizeof(d_flat)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_lookup_table, &d_lookup, sizeof(d_lookup)));

    delete[] h_lookup_table;

    // Final sanity check on remaining memory
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    printf("GPU cache loaded: %.2f MB free / %.2f MB total\n",
           free_mem / (1024.0 * 1024.0),
           total_mem / (1024.0 * 1024.0));
}


void compute_normalizers(size_t in_len, size_t out_len) {
    const size_t total_elems = (in_len + 1) * (out_len + 1);

    // Build host-side normalizers
    std::vector<double> h_norm_blob(total_elems);
    std::vector<double> h_log_norm_blob(total_elems);

    for (size_t n = 0; n <= in_len; ++n) {
        for (size_t k = 1; k <= out_len; ++k) {
            double nck = nCk(n, k);
            size_t idx = n * (out_len + 1) + k;

            h_norm_blob[idx]     = 1.0 / nck;
            h_log_norm_blob[idx] = -std::log(nck);
        }
    }

    // Allocate GPU arrays
    CUDA_CHECK(cudaMalloc(&g_h_norm_blob,     total_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_h_log_norm_blob, total_elems * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(g_h_norm_blob, h_norm_blob.data(),
                          total_elems * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_h_log_norm_blob, h_log_norm_blob.data(),
                          total_elems * sizeof(double), cudaMemcpyHostToDevice));

    // Build row-pointer lookup tables on host
    double** h_norm_ptrs     = new double*[in_len + 1];
    double** h_log_norm_ptrs = new double*[in_len + 1];
    for (size_t n = 0; n <= in_len; ++n) {
        h_norm_ptrs[n]     = g_h_norm_blob     + n * (out_len + 1);
        h_log_norm_ptrs[n] = g_h_log_norm_blob + n * (out_len + 1);
    }

    // Copy pointer tables to GPU
    CUDA_CHECK(cudaMalloc(&g_d_norm_ptr,     (in_len + 1) * sizeof(double*)));
    CUDA_CHECK(cudaMalloc(&g_d_log_norm_ptr, (in_len + 1) * sizeof(double*)));

    CUDA_CHECK(cudaMemcpy(g_d_norm_ptr, h_norm_ptrs,
                          (in_len + 1) * sizeof(double*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_d_log_norm_ptr, h_log_norm_ptrs,
                          (in_len + 1) * sizeof(double*), cudaMemcpyHostToDevice));

    // Publish pointers to device globals
    CUDA_CHECK(cudaMemcpyToSymbol(d_normalizers, &g_d_norm_ptr, sizeof(g_d_norm_ptr)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_log_normalizers, &g_d_log_norm_ptr, sizeof(g_d_log_norm_ptr)));

    delete[] h_norm_ptrs;
    delete[] h_log_norm_ptrs;

    printf("Normalizers computed for n=%zu, k=%zu\n", in_len, out_len);
}

// recomputing logarithms on gpu extremely affects performance
// compute them once, loading into the GPU
void load_log_counts(size_t total_logs) {
    // 1) Build host-side table of logs
    std::vector<double> h_log_counts(total_logs);
    h_log_counts[0] = 0.0; // log(0) undefined → set conventionally to 0
    for (size_t i = 1; i < total_logs; ++i) {
        h_log_counts[i] = std::log(static_cast<double>(i));
    }

    // 2) Allocate device array
    double* d_log_counts_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_log_counts_tmp, total_logs * sizeof(double)));

    // 3) Copy host → device
    CUDA_CHECK(cudaMemcpy(d_log_counts_tmp,
                          h_log_counts.data(),
                          total_logs * sizeof(double),
                          cudaMemcpyHostToDevice));

    // 4) Publish device pointer into device global symbol
    CUDA_CHECK(cudaMemcpyToSymbol(d_log_counts,
                                  &d_log_counts_tmp,
                                  sizeof(double*)));

    // 5) Debug info
    printf("Log-counts uploaded: %zu entries (%.2f KB)\n",
           total_logs,
           total_logs * sizeof(double) / 1024.0);
}




void load_cache(size_t in_len, size_t out_len) {
    load_gpu_cache(in_len, out_len);
    compute_normalizers(in_len, out_len);
    // nCk(in_len,out_len) is an upper bound for the number of subsequences of length "out_len" of an input of size "in_len"
    load_log_counts(nCk(in_len,out_len) + 1);
}

// =================== inline helpers ===================
__device__ double device_log(double x) {
    return log(x);
}

__host__ __device__ uint32_t reverse32(uint32_t x) {
    x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1);
    x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2);
    x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4);
    x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >> 8);
    x = ((x & 0x0000FFFF) << 16) | ((x & 0xFFFF0000) >> 16);
    return x;
}


__host__ __device__ uint32_t complement_k(uint32_t y, int k) {
    uint32_t mask = (k == 32) ? 0xFFFFFFFFu : ((1u << k) - 1);
    return y ^ mask;
}


__host__ __device__ uint32_t reverse_k(uint32_t y, int k) {
    return reverse32(y << (32 - k));
}


__host__ __device__ uint32_t rep(uint32_t x, int k)
{
    uint32_t r  = reverse_k(x, k);
    uint32_t c  = complement_k(x, k);
    uint32_t rc = reverse_k(c, k);

    uint32_t m1 = (r < x ? r : x);
    uint32_t m2 = (c < rc ? c : rc);
    uint32_t rep = (m2 < m1 ? m2 : m1);

    return rep;
}


__host__ __device__ double orbits(uint32_t x, int k)
{
    // 1) Compute the three nontrivial transforms
    uint32_t r  = reverse_k(x, k);
    uint32_t c  = complement_k(x, k);
    uint32_t rc = reverse_k(c, k);  // same as complement_k(r,k)

    bool small_orbit = (x == r) || (x == rc);
    double orbit_sz = small_orbit ? 2.0 : 4.0;

    return orbit_sz;
}


__host__ __device__ uint32_t rep_and_orbit(uint32_t x, int k, double* orbit_sz)
{
    uint32_t r  = reverse_k(x, k);
    uint32_t c  = complement_k(x, k);
    uint32_t rc = reverse_k(c, k);

    uint32_t m1 = (r < x ? r : x);
    uint32_t m2 = (c < rc ? c : rc);
    uint32_t rep = (m2 < m1 ? m2 : m1);

    bool small = (x == r) || (x == rc);
    *orbit_sz = small ? 2.0 : 4.0;

    return rep;
}

// =================== DP helpers ===================
__device__ void build_next_pos(uint32_t trans_num, uint8_t n, uint32_t b, SharedMem* shmem_ptr) {
    // Initialize with sentinel "n" (no next occurrence)
    for(int8_t i = 0; i <= n; i++){ 
        shmem_ptr->nextpos[i][b] = n;
    }

    int8_t last_b = n;
    // Traverse backwards, recording the next position of b
    for(int8_t i = n-1; i >= 0; i--)  {
        // Extract bit x_i, bit (n - i - 1) of trans_num
        uint32_t x_i = (trans_num >> (n - i - 1)) & 1;
        if(x_i == b) last_b = i;
        shmem_ptr->nextpos[i][b] = last_b; 
    }
}
__device__ void build_cnt(uint32_t trans_num, uint8_t n, uint8_t k, SharedMem* shmem_ptr) {
    // Base case:
    // cnt[i][0] = 1 (there is exactly 1 subsequence of length 0 — the empty one)
    // cnt[i][r>0] = 0 initially
    for (int i = 0; i <= n; ++i) {
        shmem_ptr->cnt[i][0] = 1;
        for (int r = 1; r <= k; ++r) shmem_ptr->cnt[i][r] = 0;
    }

    // From position i, the number of subsequences of length r
    // is the sum of those starting at the next 0 and at the next 1 (if they exist),
    // moving to index j+1 each time.
    for (int i = n - 1; i >= 0; --i) {
        for (int r = 1; r <= k; ++r) {
            uint32_t total = 0;
            uint8_t j0 = shmem_ptr->nextpos[i][0];
            if (j0 < n) total += shmem_ptr->cnt[j0 + 1][r - 1];
            uint8_t j1 = shmem_ptr->nextpos[i][1];
            if (j1 < n) total += shmem_ptr->cnt[j1 + 1][r - 1];
            shmem_ptr->cnt[i][r] = total;
        }
    }
}
__device__ void fill_DP_tables(uint32_t trans_num, uint8_t n, uint8_t k, SharedMem* shmem_ptr) {
    build_next_pos(trans_num, n, 0, shmem_ptr);
    build_next_pos(trans_num, n, 1, shmem_ptr);
    build_cnt(trans_num, n, k, shmem_ptr);
}
__device__ uint32_t unrank_k(uint32_t trans_num, uint8_t n, uint8_t k, uint32_t j, SharedMem* shmem_ptr) {
    
    uint8_t i = 0, rem = k;
    uint32_t result = 0;
    while(rem > 0) {
        // Try choosing 0 first
        uint8_t j0 = shmem_ptr->nextpos[i][0];
        uint32_t c0 = (j0 < n) ? shmem_ptr->cnt[j0 + 1][rem - 1] : 0;

        if(j < c0){

            // Choose 0
            i = j0 + 1;
            rem--;
            continue;
        }
        // Otherwise, skip the c0 subsequences that start with 0
        j -= c0;

        // Try choosing 1
        uint8_t j1 = shmem_ptr->nextpos[i][1];
        uint32_t c1 = shmem_ptr->cnt[j1 + 1][rem - 1];

        // Choose 1: set the corresponding bit (position rem-1)
        if(j < c1){
            result += (1 << (rem - 1));
            i = j1 + 1;
            rem--;
        }
    }
    return result;
}

// =================== probability helpers ===================


__device__ uint32_t total_transitions(uint32_t trans_num, uint32_t rec_num, uint8_t n, uint8_t k) {
    uint16_t n1 = (n+1) / 2;
    uint16_t n2 = n - n1;
    uint32_t total = 0;

    // Split trans_num into left (n1 bits) and right (n2 bits)
    // trans_num1 = higher n1 bits
    // trans_num2 = lower n2 bits
    uint32_t trans_num1 = trans_num >> n2;
    uint32_t trans_num2 = (trans_num1 << n2) ^ trans_num;
    uint32_t* trans_mat;

    // Loop over possible splits of subsequence rec_num:
    // j = length assigned to left half (rec_num1)
    // k-j = length assigned to right half (rec_num2)
    for (uint32_t j = 0; j <= k; ++j) {
        // If split is impossible (too many symbols for one half), skip
        if ((j > n1) || ((k - j) > n2)) continue;

        // Split rec_num accordingly into rec_num1 (j bits) and rec_num2 (k-j bits)
        uint32_t rec_num1 = rec_num >> (k - j);
        uint32_t rec_num2 = (rec_num1 << (k - j)) ^ rec_num;

        // Build indices for lookup table
        // Index encodes (transition_half + subsequence_half) relationship
        uint32_t ind_t1 = (trans_num1 << j) ^ rec_num1;
        uint32_t ind_t2 = (trans_num2 << (k - j)) ^ rec_num2;

        uint32_t count1 = 0, count2 = 0;

        // === LEFT HALF ===
        if (n1 == j) {
            // If subsequence exactly fills left half, only one possible match
            count1 = (trans_num1 == rec_num1);
        } else {
            // Otherwise, query precomputed lookup table
            trans_mat = d_lookup_table[n1 * MAX_N + j];
            count1 = trans_mat[ind_t1];
        }

        // === RIGHT HALF ===
        if (n2 == (k - j)) {
            // If subsequence exactly fills right half
            count2 = (trans_num2 == rec_num2);
        } else {
            // Otherwise, use lookup
            trans_mat = d_lookup_table[n2 * MAX_N + (k - j)];
            count2 = trans_mat[ind_t2];
        }

        // Combine: total number of ways = product of left and right counts
        total += count1 * count2;
    }

    return total;
}

__device__ double denom_term(uint32_t trans_num, uint32_t rec_num, double Q_x, uint8_t n, uint8_t k) {
    return Q_x * total_transitions(trans_num, rec_num, n, k) * d_normalizers[n][k];
}

__device__ double log_W_term(uint32_t trans_num, uint32_t rec_num, double log_Q_k, double* d_log_W_jk_den,
                                        uint8_t n, uint8_t k) {
    uint32_t total = total_transitions(trans_num, rec_num, n, k);
    
    double d_Pjk = total * d_normalizers[n][k];

    // Log P(y|x) = log(counts) + logNormalizer
    double log_Pjk = d_log_counts[total] + d_log_normalizers[n][k];
    uint32_t repr = rep(rec_num,k);
    return d_Pjk * (log_Pjk + log_Q_k - d_log_W_jk_den[repr]); 
}

__device__ double KL_term(uint32_t trans_num, uint32_t rec_num,
                                  const double* d_log_W_jk_den, uint8_t n, uint8_t k) {
    uint32_t total = total_transitions(trans_num, rec_num, n, k);
    double d_Pjk = total * d_normalizers[n][k];


    // Log P(y|x) = log(counts) + logNormalizer
    double log_Pjk = d_log_counts[total] + d_log_normalizers[n][k];

    uint32_t repr = rep(rec_num,k);
    double logW = d_log_W_jk_den[repr];
    return d_Pjk * (log_Pjk - logW);
}



__global__ void compute_den_kernel(
    double*      d_Qi,          // Q distribution on device
    double*            d_denoms,      // output array [num_received]
    uint32_t* d_trans_canonicals,
    uint8_t           n,             // length of transmitted codewords in bits
    uint8_t           k             // length of received codewords in bits
)
{
    
	uint32_t tid = threadIdx.x, bid = blockIdx.x;

	uint32_t trans_num = d_trans_canonicals[bid];
    double Q_x     = d_Qi[bid];

	__shared__ SharedMem shmem;

    if (tid == 0) {
        fill_DP_tables(trans_num, n, k, &shmem);
    }
    
    __syncthreads();  // Ensure cnt is ready before other threads use it

    // Compute orbit size of transmitted word (symmetry group size)
    double trans_orbit_size = orbits(trans_num, n);

    // ========================================================================
    // Each thread handles a *strided subset* of received words
    // - unrank_k enumerates received subsequences
    // - Each received word is mapped to its canonical representative
    // - Factor accounts for orbit size ratio (symmetry correction)
    // ========================================================================
    for (uint32_t j = tid; j < shmem.cnt[0][k]; j += blockDim.x) {


        // Decode received subsequence by unranking index j
        uint32_t rec_num = unrank_k(trans_num, n, k, j, &shmem);

        // Compute canonical representative and orbit size of received word
        double rec_orbit_size;
        uint32_t rec_canonical = rep_and_orbit(rec_num,k,&rec_orbit_size);

        // Orbit correction factor: transmitter orbit / receiver orbit
        double factor = trans_orbit_size / rec_orbit_size;


        // compute your increment
        double incr = factor * denom_term(trans_num, rec_num, Q_x, n, k);

        // There is a low probability (not zero) that different threads generate the same "rec_canonical"
        // In that case we have to ensure d_denoms[rec_canonical] is not modified at the same time by more than one thread
        // Accumulate safely into d_denoms using atomicAdd
        atomicAdd(&d_denoms[rec_canonical], incr);
            
    }

	
}

__global__ void compute_log_W_gpu(
    double*    log_d_Q_k,
    uint32_t* d_trans_canonicals,
    double*    d_log_den,
    double*    log_W,     
    uint8_t    n, uint8_t k
) {
    double partial = 0;
	uint32_t tid = threadIdx.x,bid = blockIdx.x;
	uint32_t trans_num = d_trans_canonicals[bid];
    double    log_Qx     = log_d_Q_k[bid];

	__shared__ SharedMem shmem;



    if (tid == 0) {
        fill_DP_tables(trans_num, n, k, &shmem);
    }
    __syncthreads();  // Ensure cnt is ready before other threads use it

    for (uint32_t j = tid; j < shmem.cnt[0][k]; j += blockDim.x) {
        uint32_t rec_num = unrank_k(trans_num, n, k, j, &shmem);

        partial += log_W_term(trans_num, rec_num, log_Qx, d_log_den, n, k);
    }

    shmem.sdata[tid] = partial;
    __syncthreads();

    // Standard power‐of‐two parallel reduction across blockDim.x threads:
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem.sdata[tid] += shmem.sdata[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 now has the full sum for this block:
    if (tid == 0) {
        log_W[bid] = shmem.sdata[0];
    }

}


__global__ void compute_KL_kernel(
    const uint32_t* d_trans_nums,      // [n_I] device array of canonical x indices
    const double*   d_log_den,     // [N_j] device: log W(y) for each y
    double*         d_KL,               // [n_I] output: KL(x) for each x
    uint8_t        n,                  // bit-length of x
    uint8_t        k                   // bit-length of y
) {
    double partial = 0;
	uint32_t tid = threadIdx.x,bid = blockIdx.x;
	uint32_t trans_num = d_trans_nums[bid];

	__shared__ SharedMem shmem;


    if (tid == 0) {
        fill_DP_tables(trans_num, n, k, &shmem);
    }
    __syncthreads();  // Ensure cnt is ready before other threads use it


    for (uint32_t j = tid; j < shmem.cnt[0][k]; j += blockDim.x) {
        uint32_t rec_num = unrank_k(trans_num, n, k, j, &shmem);
        partial += KL_term(trans_num, rec_num, d_log_den, n, k);
    }

    shmem.sdata[tid] = partial;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem.sdata[tid] += shmem.sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_KL[bid] = shmem.sdata[0];
    }
}

__global__ void apply_log_kernel(double* d_array, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (d_array[idx] != 0.0) {
            d_array[idx] = log(d_array[idx]); 
        }
    }
}

__global__ void fill_log_counts(double* d_array, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && idx > 0) {
        d_array[idx] = device_log(idx);
    }
}


std::vector<double> baa_iteration (std::vector<double>& Q_i,std::vector<uint32_t>& trans_canonicals, size_t n, size_t k) 
{
    // Compute denominator terms log W_jk(y) for all received words */
    std::vector<double> log_den = compute_all_log_D(Q_i, trans_canonicals, n, k);

    // Compute log α_x for each transmitted word */
    std::vector<double> log_W =
        compute_all_log_W(Q_i, log_den, trans_canonicals, n, k);

    // Upload log W(x) to device 
    thrust::device_vector<double> d_log_W = log_W;

    // Find maximum log W(x) 
    double max_log_W = *thrust::max_element(
        d_log_W.begin(), d_log_W.end());

    // Convert log W(x) → exp(log W(x) - max) in parallel 
    thrust::transform(
        d_log_W.begin(), d_log_W.end(),
        d_log_W.begin(),
        [max_log_W] __device__(double v) {
            return exp(v - max_log_W);
        });

    // Compute sum of W(x) 
    thrust::device_vector<uint32_t> d_trans_canonicals(
        trans_canonicals.begin(), trans_canonicals.end());

    double W_sum = thrust::transform_reduce(
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(d_log_W.size()),
        index_dependent_transform(
            thrust::raw_pointer_cast(d_log_W.data()),
            thrust::raw_pointer_cast(d_trans_canonicals.data()),
            n),
        0.0,
        thrust::plus<double>());

    // Normalize W(x) in parallel 
    thrust::transform(
        d_log_W.begin(), d_log_W.end(),
        d_log_W.begin(),
        [W_sum] __device__(double v) { return v / W_sum; });

    // Copy back to host 
    std::vector<double> new_Q(d_log_W.size());
    thrust::copy(d_log_W.begin(), d_log_W.end(), new_Q.begin());

    return new_Q;
}


std::vector<double> compute_all_log_D(
    std::vector<double>& Q_i,
    std::vector<uint32_t>& trans_canonicals,
    size_t n, size_t k)
{
    auto t1 = std::chrono::high_resolution_clock::now();

    const size_t N_r = 1ull << k;  // number of received words
    
    // all representatives must be < 2^{k-1}
    const size_t num_received = N_r / 2;  // only canonical representatives

    std::vector<double> log_den(num_received);

    /* Allocate GPU buffers */
    double* d_Qi        = nullptr;
    double* d_denoms    = nullptr;
    uint32_t* d_trans_nums = nullptr;

    CUDA_CHECK(cudaMalloc(&d_Qi, trans_canonicals.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_denoms, num_received * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_trans_nums, trans_canonicals.size() * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_Qi, Q_i.data(),
                          trans_canonicals.size() * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_trans_nums, trans_canonicals.data(),
                          trans_canonicals.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_denoms, 0, num_received * sizeof(double)));

    // Choose kernel launch configuration 

    uint32_t block_size = 1024;
    const size_t gridSize = trans_canonicals.size();


    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = t2 - t1;
    printf("Initialized denominator calculation in %.6f sec\n", diff.count());

    // Launch denominator accumulation kernel 
    compute_den_kernel<<<gridSize, block_size>>>(
        d_Qi,        // input Q(x) on device
        d_denoms,    // output denominators W(y)
        d_trans_nums,
        n, k
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Apply log() elementwise to denominators 
    {
        size_t threadsPerBlock = 1024; // block_size
        size_t blocksPerGrid = (num_received + threadsPerBlock - 1) / threadsPerBlock;

        apply_log_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_denoms, num_received);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    diff = t3 - t2;
    printf("Denominator calculation done in %.6f sec\n", diff.count());

    // Copy result back to host 
    CUDA_CHECK(cudaMemcpy(log_den.data(),
                          d_denoms,
                          num_received * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // Free GPU memory 
    cudaFree(d_Qi);
    cudaFree(d_trans_nums);
    cudaFree(d_denoms);

    auto t4 = std::chrono::high_resolution_clock::now();
    diff = t4 - t3;
    printf("Cleanup completed in %.6f sec\n", diff.count());

    return log_den;
}


std::vector<double> compute_all_log_W(
    std::vector<double>& Q_i,
    std::vector<double>& log_den,
    std::vector<uint32_t>& trans_canonical, size_t n, size_t k)
{
    auto t1 = std::chrono::high_resolution_clock::now();

    const size_t num_x = trans_canonical.size();

    std::vector<double> h_log_W(num_x);

    // Allocate GPU buffers 
    double*   d_log_den = nullptr;
    double*   d_log_W  = nullptr;
    double*   d_Q_i         = nullptr;
    uint32_t* d_trans_nums  = nullptr;

    CUDA_CHECK(cudaMalloc(&d_log_den, log_den.size() * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_log_den, log_den.data(),
                          log_den.size() * sizeof(double),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_log_W, num_x * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_trans_nums, num_x * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_trans_nums, trans_canonical.data(),
                          num_x * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_Q_i, Q_i.size() * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_Q_i, Q_i.data(),
                          Q_i.size() * sizeof(double),
                          cudaMemcpyHostToDevice));

    // Convert Q(x) → log Q(x) on the GPU 
    {
        size_t threadsPerBlock = 1024;
        size_t blocksPerGrid = (num_x + threadsPerBlock - 1) / threadsPerBlock;

        apply_log_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Q_i, num_x);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Choose kernel launch configuration 
    
    uint32_t block_size = 1024;
    block_size = std::clamp(block_size, 32u, 1024u);

    const size_t gridSize = num_x;

    auto t2 = std::chrono::high_resolution_clock::now();
    printf("Initialized log(W) calculation in %.6f sec\n",
           std::chrono::duration<double>(t2 - t1).count());

    // Launch kernel to compute log W 
    compute_log_W_gpu<<<gridSize, block_size>>>(
        d_Q_i,          // log Q(x)
        d_trans_nums,   // canonical transmitted codewords
        d_log_den,  // denominators log D(y)
        d_log_W,   // output log W(x)
        n, k
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t3 = std::chrono::high_resolution_clock::now();
    printf("log(W) calculation done in %.6f sec\n",
           std::chrono::duration<double>(t3 - t2).count());

    // Copy result back to host 
    CUDA_CHECK(cudaMemcpy(h_log_W.data(), d_log_W,
                          num_x * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // Free GPU memory 
    cudaFree(d_log_den);
    cudaFree(d_Q_i);
    cudaFree(d_trans_nums);
    cudaFree(d_log_W);

    auto t4 = std::chrono::high_resolution_clock::now();
    printf("Cleanup completed in %.6f sec\n",
           std::chrono::duration<double>(t4 - t3).count());

    return h_log_W;
}

/**
 * Kernel to compute per-input contributions:
 *   contrib[i] = orbit_size(x_i) * Q(x_i) * KL(x_i)
 */
__global__ void compute_rate_terms(
    const uint32_t* trans_canonicals,  // [num_x]
    const double*   Q_i,               // [num_x]
    const double*   KL,                // [num_x]
    double*         contrib,           // [num_x]
    uint8_t n,                         // input length
    size_t num_x)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_x) return;

    uint32_t x = trans_canonicals[i];
    double orbit_size = orbits(x, n);  // assumes __device__ orbits()
    contrib[i] = orbit_size * Q_i[i] * KL[i];
}

double compute_rate(
    std::vector<double>& log_W_jk_den,
    std::vector<double>& Q_i,
    std::vector<uint32_t>& trans_canonical, uint8_t n, uint8_t k)
{
    const size_t num_x = trans_canonical.size();

    
    // Allocate GPU buffers
      
    uint32_t* d_trans_nums = nullptr;
    double*   d_log_W      = nullptr;
    double*   d_KL         = nullptr;
    double*   d_Q          = nullptr;
    double*   d_contrib    = nullptr;

    CUDA_CHECK(cudaMalloc(&d_trans_nums, num_x * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_log_W,      log_W_jk_den.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_KL,         num_x * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Q,          num_x * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_contrib,    num_x * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_trans_nums, trans_canonical.data(),
                          num_x * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_log_W, log_W_jk_den.data(),
                          log_W_jk_den.size() * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q, Q_i.data(),
                          num_x * sizeof(double),
                          cudaMemcpyHostToDevice));

    
    // Kernel to compute KL(x)
    uint32_t block_size = 1024;
    block_size = std::clamp(block_size, 32u, 1024u);

    const size_t gridSize  = num_x;
    const size_t sharedMem = block_size * sizeof(double);

    compute_KL_kernel<<<gridSize, block_size, sharedMem>>>(
        d_trans_nums,
        d_log_W,
        d_KL,
        n,
        k
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

   
    // Compute per-x rate contributions
    const int threads = 256;
    const int blocks  = (num_x + threads - 1) / threads;

    compute_rate_terms<<<blocks, threads>>>(
        d_trans_nums, d_Q, d_KL, d_contrib, n, num_x);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    
    // Parallel reduction
       
    thrust::device_ptr<double> d_ptr(d_contrib);
    double rate = thrust::reduce(d_ptr, d_ptr + num_x, 0.0, thrust::plus<double>());


    // Cleanup
    cudaFree(d_trans_nums);
    cudaFree(d_log_W);
    cudaFree(d_KL);
    cudaFree(d_Q);
    cudaFree(d_contrib);

    return rate;
}


double compute_max_deviation(const std::vector<double>& Q,
                             const std::vector<double>& Q_before)
{
    // Copy host distributions to device
    thrust::device_vector<double> d_Q(Q.begin(), Q.end());
    thrust::device_vector<double> d_Qb(Q_before.begin(), Q_before.end());

    // Build a zip iterator over pairs (Q_i, Qb_i)
    auto begin = thrust::make_zip_iterator(
        thrust::make_tuple(d_Q.begin(), d_Qb.begin()));
    auto end = thrust::make_zip_iterator(
        thrust::make_tuple(d_Q.end(), d_Qb.end()));

    // Initialize reduction with -∞
    double init = -std::numeric_limits<double>::infinity();

    // Device lambda: compute log2(Qi) - log2(Qbi) for each pair
    auto op = [] __device__(thrust::tuple<double, double> t) -> double {
        double Qi  = thrust::get<0>(t);
        double Qbi = thrust::get<1>(t);
        return log2(Qi) - log2(Qbi);
    };

    // Perform transform-reduce on GPU
    return thrust::transform_reduce(
        thrust::device,
        begin, end,
        op,
        init,
        thrust::maximum<double>());
}




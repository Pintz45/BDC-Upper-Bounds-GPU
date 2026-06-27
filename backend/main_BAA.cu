#include "BAA.cuh"

std::vector<double> read_distribution_file(const std::string& filename) {
    std::vector<double> result;
    std::ifstream infile(filename);
    
    if (!infile) {
        std::cerr << "Error: Could not open " << filename << " for reading.\n";
        return result;
    }

    double value;
    while (infile >> value) {
        result.push_back(value);
    }

    return result;
}

// Function to compute the channel capacity C_n',k' by running the Blahut-Arimoto algorithm
/**
 * Compute the channel capacity C_{n,k} using the Blahut–Arimoto algorithm.
 *
 * This function iteratively refines the input distribution Q until convergence,
 * and finally computes the mutual information (rate) for the channel.
 *
 * Steps:
 *   1. Initialize input distribution Q (from file or uniform).
 *   2. Upload GPU caches (transition tables, normalizers).
 *   3. Run BAA iterations until max deviation < threshold "a".
 *   4. Periodically save intermediate distributions to file.
 *   5. After convergence, compute final rate using GPU kernels.
 *
 * in_len          Length of input codewords (n).
 * out_len         Length of output codewords (k).
 * a               Convergence threshold.
 * read_from_file  If true, read initial distribution Q from file.
 */
void compute_C_n_k(size_t in_len, size_t out_len, double a, bool read_from_file, size_t save_iterations)
{
    double rate   = 0.0;
    double mx     = a;   // maximum deviation between Q^{t+1} and Q^t
    double time  = 0.0; // accumulated runtime in seconds
    int    i      = 1;   // iteration counter

    std::vector<uint32_t> trans_canonicals;
    for(uint32_t i = 0; i < (1 << in_len);i++){
        if(rep(i, in_len) != i) continue;
        trans_canonicals.push_back(i);
    }

    // Initialize distribution Q
    std::vector<double> Q(trans_canonicals.size()), Q_before;

    std::string file = "distributions/dist_" +
                       std::to_string(in_len) + "_" +
                       std::to_string(out_len) + ".txt";

    if (read_from_file) {
        std::cout << "Loading distribution from file: " << file << "\n";
        Q = read_distribution_file(file);
    } else {
        std::cout << "Initializing uniform distribution\n";
        std::fill(Q.begin(), Q.end(), 1.0 / (1 << in_len));
    }

    // Step 2: Load GPU caches
    load_cache(in_len, out_len);
    printf("Starting BAA iterations...\n");

    
    // Blahut–Arimoto iterations
    while (mx >= a) {
        auto start = std::chrono::high_resolution_clock::now();

        // Keep a copy of the previous distribution
        Q_before = Q;

        // Single BA iteration: update Q
        Q = baa_iteration(Q, trans_canonicals, in_len, out_len);

        // Periodically save distribution to file
        if (i % save_iterations == 0) {
            FILE* outfile = fopen(file.c_str(), "w");
            if (!outfile) return;

            auto t1 = std::chrono::high_resolution_clock::now();
            for (size_t j = 0; j < Q.size(); j++) {
                fprintf(outfile, "%.17g\n", Q[j]);
            }
            fclose(outfile);

            auto t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff1 = t2 - t1;
            printf("Distribution saved in %.3f seconds\n", diff1.count());
        }

        // Compute max deviation log2(Q_i/Q_i_before) on GPU
        mx = compute_max_deviation(Q, Q_before);


        // Track runtime for this iteration
        auto end_ = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end_ - start;

        printf("Iteration %d: mx = %.6f, time = %.3f s\n", i, mx, diff.count());
        time += diff.count();
        i++;
    }

    // Save final distribution
    
    FILE* outfile = fopen(file.c_str(), "w");
    if (!outfile) return;

    for (size_t j = 0; j < Q.size(); j++) {
        fprintf(outfile, "%.17g\n", Q[j]);
    }
    fclose(outfile);
    printf("Final distribution saved to %s\n", file.c_str());

    // Compute final capacity
    std::vector<double> log_D = compute_all_log_D(Q, trans_canonicals, in_len, out_len);
    rate = compute_rate(log_D, Q, trans_canonicals, in_len, out_len);
    rate /= log(2); // Convert to bits

    cudaDeviceReset();

    printf("Computed C_{%zu,%zu} = %.6f bits (max deviation %.6f)\n",
           in_len, out_len, rate, mx);
}


int main(int argc, char* argv[])
{
    // n and k values should be provided
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <n> <k> [reed_from_file] [a] [save_iterations]\n";
        std::cerr << "Exemple: " << argv[0]
                  << " 21 20\n";
        std::cerr << "Exemple: " << argv[0]
                  << " 21 20 1 0.01 200\n";
        return 1;
    }


    size_t in_len = std::stoul(argv[1]);
    size_t out_len = std::stoul(argv[2]);

    // Default values
    bool reed_from_file = false;
    double a = 0.005;
    size_t save_iterations = 100;

    // Optionally provided parameters
    if (argc > 3)
        reed_from_file = std::stoi(argv[3]) != 0;

    if (argc > 4)
        a = std::stod(argv[4]);

    if (argc > 5)
        save_iterations = std::stoul(argv[5]);

    compute_C_n_k(in_len, out_len, a, reed_from_file, save_iterations);

    return 0;
}


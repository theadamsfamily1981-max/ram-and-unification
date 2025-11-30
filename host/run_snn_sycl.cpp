// =============================================================================
// run_snn_sycl.cpp
//
// Intel oneAPI / SYCL Host Application for Kitten SNN FPGA Fabric
// Ara-SYNERGY Project
//
// This is the Intel Stratix-10 / Agilex equivalent of run_snn_fpga.cpp (XRT).
//
// Compile:
//   # Emulation
//   icpx -fsycl -fintelfpga -DUSE_FPGA_EMULATOR run_snn_sycl.cpp -o run_snn_emu
//
//   # Hardware (generates .aocx)
//   icpx -fsycl -fintelfpga -Xshardware -Xsboard=pac_s10 \
//        run_snn_sycl.cpp -o run_snn_hw
//
// Usage:
//   ./run_snn_sycl <num_timesteps> [fabric_dir]
//   ./run_snn_sycl 100 build/fpga/
//
// =============================================================================

#include <CL/sycl.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdint>
#include <cstring>

// Intel FPGA extensions
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl = cl::sycl;

// =============================================================================
// Fixed-Point Configuration (must match FABRIC_MAPPING.md)
// =============================================================================

constexpr int V_FRAC_BITS      = 10;  // Q5.10 membrane
constexpr int W_FRAC_BITS      = 6;   // Q1.6 weights (8-bit)
constexpr int I_FRAC_BITS      = 16;  // Q15.16 accumulator
constexpr int PARAM_FRAC_BITS  = 14;  // Q1.14 parameters

constexpr int NEURON_RECORD_SIZE = 6;
constexpr int V_OFFSET           = 0;
constexpr int V_TH_OFFSET        = 2;
constexpr int FLAGS_OFFSET       = 4;

constexpr uint16_t FLAG_SPIKED     = 0x0001;
constexpr uint16_t FLAG_REFRACTORY = 0x0002;

// =============================================================================
// File I/O Utilities
// =============================================================================

std::vector<uint8_t> read_binary_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("Failed to open: " + path);
    }
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(size);
    if (!f.read(reinterpret_cast<char*>(buf.data()), size)) {
        throw std::runtime_error("Failed to read: " + path);
    }
    return buf;
}

// =============================================================================
// Simple JSON Parser (minimal implementation)
// =============================================================================

class SimpleJSON {
public:
    std::string raw;

    SimpleJSON(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("Failed to open JSON: " + path);
        std::stringstream ss;
        ss << f.rdbuf();
        raw = ss.str();
    }

    uint32_t get_uint(const std::string& key) const {
        size_t pos = raw.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0;
        pos = raw.find(":", pos);
        if (pos == std::string::npos) return 0;
        while (pos < raw.size() && (raw[pos] == ':' || raw[pos] == ' ')) pos++;
        return static_cast<uint32_t>(std::stoul(raw.substr(pos)));
    }

    std::string get_string(const std::string& key) const {
        size_t pos = raw.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = raw.find(":", pos);
        if (pos == std::string::npos) return "";
        pos = raw.find("\"", pos);
        if (pos == std::string::npos) return "";
        size_t end = raw.find("\"", pos + 1);
        if (end == std::string::npos) return "";
        return raw.substr(pos + 1, end - pos - 1);
    }
};

// =============================================================================
// SYCL Device Selector
// =============================================================================

sycl::device select_fpga_device() {
#if defined(USE_FPGA_EMULATOR)
    return sycl::ext::intel::fpga_emulator_selector_v;
#elif defined(USE_FPGA_SIMULATOR)
    return sycl::ext::intel::fpga_simulator_selector_v;
#else
    return sycl::ext::intel::fpga_selector_v;
#endif
}

// =============================================================================
// SNN Kernel (SYCL single_task)
// =============================================================================

// Kernel functor for single timestep
class SNNKernel;

void run_snn_timestep(
    sycl::queue& q,
    sycl::buffer<uint8_t, 1>& weights_buf,
    sycl::buffer<uint8_t, 1>& neurons_buf,
    sycl::buffer<uint8_t, 1>& input_buf,
    sycl::buffer<uint8_t, 1>& output_buf,
    sycl::buffer<int32_t, 1>& I_post_buf,
    uint32_t num_neurons,
    uint32_t row_ptr_offset,
    uint32_t row_ptr_length,
    uint32_t col_idx_offset,
    uint32_t weights_offset,
    uint16_t alpha
) {
    q.submit([&](sycl::handler& h) {
        auto w_acc   = weights_buf.get_access<sycl::access::mode::read>(h);
        auto n_acc   = neurons_buf.get_access<sycl::access::mode::read_write>(h);
        auto in_acc  = input_buf.get_access<sycl::access::mode::read>(h);
        auto out_acc = output_buf.get_access<sycl::access::mode::write>(h);
        auto I_acc   = I_post_buf.get_access<sycl::access::mode::read_write>(h);

        h.single_task<SNNKernel>([=]() [[intel::kernel_args_restrict]] {
            // Get raw pointers
            const uint8_t* weights_ptr = w_acc.get_pointer();
            uint8_t*       neurons_ptr = n_acc.get_pointer();
            const uint8_t* in_ptr      = in_acc.get_pointer();
            uint8_t*       out_ptr     = out_acc.get_pointer();
            int32_t*       I_post      = I_acc.get_pointer();

            // CSR array pointers
            const uint32_t* row_ptr = reinterpret_cast<const uint32_t*>(
                weights_ptr + row_ptr_offset);
            const uint32_t* col_idx = reinterpret_cast<const uint32_t*>(
                weights_ptr + col_idx_offset);
            const int8_t*   w_vals  = reinterpret_cast<const int8_t*>(
                weights_ptr + weights_offset);

            // Phase 1: Clear I_post
            [[intel::initiation_interval(1)]]
            for (uint32_t n = 0; n < num_neurons; ++n) {
                I_post[n] = 0;
            }

            // Phase 2: CSR projection
            uint32_t N_pre = row_ptr_length - 1;
            for (uint32_t pre = 0; pre < N_pre; ++pre) {
                uint8_t spike = in_ptr[pre];
                if (spike == 0) continue;

                uint32_t start = row_ptr[pre];
                uint32_t end   = row_ptr[pre + 1];

                [[intel::initiation_interval(1)]]
                for (uint32_t idx = start; idx < end; ++idx) {
                    uint32_t post = col_idx[idx];
                    int8_t w_q = w_vals[idx];

                    // Scale weight: Q1.6 → Q15.16
                    int32_t w_scaled = static_cast<int32_t>(w_q)
                                       << (I_FRAC_BITS - W_FRAC_BITS);
                    I_post[post] += w_scaled;
                }
            }

            // Phase 3: Clear output spikes
            [[intel::initiation_interval(1)]]
            for (uint32_t n = 0; n < num_neurons; ++n) {
                out_ptr[n] = 0;
            }

            // Phase 4: LIF update
            [[intel::initiation_interval(1)]]
            for (uint32_t n = 0; n < num_neurons; ++n) {
                uint32_t base = n * NEURON_RECORD_SIZE;

                // Load neuron state
                int16_t v_fp    = *reinterpret_cast<int16_t*>(neurons_ptr + base + V_OFFSET);
                int16_t v_th_fp = *reinterpret_cast<int16_t*>(neurons_ptr + base + V_TH_OFFSET);
                uint16_t flags  = *reinterpret_cast<uint16_t*>(neurons_ptr + base + FLAGS_OFFSET);

                // Skip if refractory
                bool refractory = (flags & FLAG_REFRACTORY) != 0;
                if (refractory) continue;

                // Leak: v_new = alpha * v (Q1.14 × Q5.10 → shift 14)
                int32_t v_leak = (static_cast<int32_t>(alpha) * static_cast<int32_t>(v_fp))
                                 >> PARAM_FRAC_BITS;

                // Integrate: add scaled current (Q15.16 → Q5.10)
                int32_t I_scaled = I_post[n] >> (I_FRAC_BITS - V_FRAC_BITS);
                int32_t v_new = v_leak + I_scaled;

                // Saturate
                if (v_new > 32767) v_new = 32767;
                if (v_new < -32768) v_new = -32768;

                // Threshold check
                bool fired = (v_new >= v_th_fp);
                if (fired) {
                    out_ptr[n] = 1;
                    v_new -= v_th_fp;  // Soft reset
                    flags |= FLAG_SPIKED;
                } else {
                    flags &= ~FLAG_SPIKED;
                }

                // Store updated state
                *reinterpret_cast<int16_t*>(neurons_ptr + base + V_OFFSET) =
                    static_cast<int16_t>(v_new);
                *reinterpret_cast<uint16_t*>(neurons_ptr + base + FLAGS_OFFSET) = flags;
            }
        });
    }).wait();
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <num_timesteps> [fabric_dir]\n";
            std::cerr << "\nExample:\n";
            std::cerr << "  " << argv[0] << " 100 build/fpga/\n";
            return 1;
        }

        uint32_t num_timesteps = static_cast<uint32_t>(std::stoul(argv[1]));
        std::string fabric_dir = (argc >= 3) ? argv[2] : "build/fpga";

        std::string topo_path    = fabric_dir + "/fabric_topology.json";
        std::string weights_path = fabric_dir + "/weights.bin";
        std::string neurons_path = fabric_dir + "/neurons.bin";

        std::cout << "[sycl] Loading fabric from: " << fabric_dir << "\n";

        // Parse topology
        SimpleJSON topo(topo_path);
        uint32_t num_neurons = topo.get_uint("record_count");
        if (num_neurons == 0) {
            num_neurons = topo.get_uint("total_neurons");
        }

        // Get projection offsets (assumes first projection)
        uint32_t row_ptr_offset = topo.get_uint("row_ptr_offset_bytes");
        uint32_t row_ptr_length = topo.get_uint("row_ptr_length");
        uint32_t col_idx_offset = topo.get_uint("col_idx_offset_bytes");
        uint32_t weights_offset = topo.get_uint("weights_offset_bytes");

        // LIF parameters
        uint16_t alpha = 14746;  // 0.9 in Q1.14

        std::cout << "[sycl] Fabric config:\n";
        std::cout << "  num_neurons:     " << num_neurons << "\n";
        std::cout << "  row_ptr_offset:  " << row_ptr_offset << "\n";
        std::cout << "  row_ptr_length:  " << row_ptr_length << "\n";
        std::cout << "  col_idx_offset:  " << col_idx_offset << "\n";
        std::cout << "  weights_offset:  " << weights_offset << "\n";
        std::cout << "  alpha (Q1.14):   " << alpha << "\n";

        // Load binary files
        auto weights_bin = read_binary_file(weights_path);
        auto neurons_bin = read_binary_file(neurons_path);

        std::cout << "[sycl] Loaded weights: " << weights_bin.size() << " bytes\n";
        std::cout << "[sycl] Loaded neurons: " << neurons_bin.size() << " bytes\n";

        // Spike buffers (1 byte per neuron)
        std::vector<uint8_t> input_spikes(num_neurons, 0);
        std::vector<uint8_t> output_spikes(num_neurons, 0);
        std::vector<int32_t> I_post(num_neurons, 0);

        // Generate some test input (sparse random spikes)
        srand(42);
        for (uint32_t i = 0; i < num_neurons; ++i) {
            if (rand() % 100 < 5) {  // 5% spike probability
                input_spikes[i] = 1;
            }
        }

        // Select FPGA device
        sycl::device dev = select_fpga_device();
        sycl::queue q(dev, sycl::property::queue::enable_profiling{});

        std::cout << "[sycl] Using device: "
                  << q.get_device().get_info<sycl::info::device::name>() << "\n";

        // Create SYCL buffers
        sycl::buffer<uint8_t, 1> weights_buf(weights_bin.data(),
            sycl::range<1>(weights_bin.size()));
        sycl::buffer<uint8_t, 1> neurons_buf(neurons_bin.data(),
            sycl::range<1>(neurons_bin.size()));
        sycl::buffer<uint8_t, 1> input_buf(input_spikes.data(),
            sycl::range<1>(num_neurons));
        sycl::buffer<uint8_t, 1> output_buf(output_spikes.data(),
            sycl::range<1>(num_neurons));
        sycl::buffer<int32_t, 1> I_post_buf(I_post.data(),
            sycl::range<1>(num_neurons));

        // Warmup
        std::cout << "[sycl] Running warmup...\n";
        for (int i = 0; i < 5; ++i) {
            run_snn_timestep(q, weights_buf, neurons_buf, input_buf, output_buf,
                            I_post_buf, num_neurons, row_ptr_offset, row_ptr_length,
                            col_idx_offset, weights_offset, alpha);
        }

        // Timed run
        std::cout << "[sycl] Running " << num_timesteps << " timesteps...\n";
        auto start = std::chrono::high_resolution_clock::now();

        for (uint32_t t = 0; t < num_timesteps; ++t) {
            run_snn_timestep(q, weights_buf, neurons_buf, input_buf, output_buf,
                            I_post_buf, num_neurons, row_ptr_offset, row_ptr_length,
                            col_idx_offset, weights_offset, alpha);

            // For recurrent networks, swap input/output here
            // std::swap(input_buf, output_buf);

            if ((t + 1) % 100 == 0) {
                std::cout << "  [" << (t + 1) << "/" << num_timesteps << "]\n";
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        double ms_total = duration.count() / 1000.0;
        double ms_per_step = ms_total / num_timesteps;

        std::cout << "\n[sycl] Results:\n";
        std::cout << "  Total time:    " << ms_total << " ms\n";
        std::cout << "  Per timestep:  " << ms_per_step << " ms\n";
        std::cout << "  Throughput:    " << (1000.0 / ms_per_step) << " steps/sec\n";

        // Read back output spikes
        {
            auto out_host = output_buf.get_access<sycl::access::mode::read>();
            size_t total_spikes = 0;
            for (uint32_t i = 0; i < num_neurons; ++i) {
                if (out_host[i]) total_spikes++;
            }
            std::cout << "  Output spikes: " << total_spikes << " / " << num_neurons << "\n";
        }

        std::cout << "\n[sycl] Done.\n";
        return 0;

    } catch (const sycl::exception& e) {
        std::cerr << "[sycl] SYCL exception: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[sycl] Error: " << e.what() << "\n";
        return 1;
    }
}

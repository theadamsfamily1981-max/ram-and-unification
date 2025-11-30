// =============================================================================
// run_snn_fpga.cpp
//
// XRT/OpenCL Host Application for Kitten SNN FPGA Fabric
// Ara-SYNERGY Project
//
// Usage:
//   ./run_snn_fpga --xclbin snn_kernel.xclbin \
//                  --topology build/fpga/fabric_topology.json \
//                  --weights build/fpga/weights.bin \
//                  --neurons build/fpga/neurons.bin \
//                  --spikes_in input_spikes.bin \
//                  --timesteps 100
//
// Build:
//   g++ -std=c++17 -O2 -I${XILINX_XRT}/include -L${XILINX_XRT}/lib \
//       -o run_snn_fpga run_snn_fpga.cpp -lxrt_coreutil -pthread
//
// =============================================================================

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// JSON parsing (header-only, include nlohmann/json.hpp or use simple parser)
// For simplicity, we use a minimal parser here. In production, use nlohmann/json.
#include <sstream>

// =============================================================================
// Configuration Structures (matching FABRIC_MAPPING.md)
// =============================================================================

struct FixedPointConfig {
    uint32_t v_bits;
    uint32_t v_frac_bits;
    uint32_t w_bits;
    uint32_t w_frac_bits;
    uint32_t param_bits;
    uint32_t param_frac_bits;
};

struct PopulationConfig {
    std::string name;
    uint32_t size;
    uint32_t id_offset;
    std::string type;
};

struct ProjectionConfig {
    std::string name;
    std::string pre_population;
    std::string post_population;
    uint32_t pre_start;
    uint32_t pre_end;
    uint32_t post_start;
    uint32_t post_end;
    uint32_t row_ptr_offset_bytes;
    uint32_t row_ptr_length;
    uint32_t col_idx_offset_bytes;
    uint32_t col_idx_length;
    uint32_t weights_offset_bytes;
    uint32_t weights_length;
};

struct NeuronStateLayout {
    uint32_t record_size_bytes;
    uint32_t record_count;
    uint32_t v_offset_bytes;
    uint32_t v_stride_bytes;
    uint32_t threshold_offset_bytes;
    uint32_t threshold_stride_bytes;
    uint32_t flags_offset_bytes;
    uint32_t flags_stride_bytes;
};

struct FabricTopology {
    uint32_t version;
    std::string endianness;
    FixedPointConfig fixed_point;
    std::vector<PopulationConfig> populations;
    std::vector<ProjectionConfig> projections;
    NeuronStateLayout neuron_layout;
    uint32_t total_neurons;
    uint32_t total_synapses;
};

// =============================================================================
// Neuron State Record (6 bytes, packed)
// =============================================================================

#pragma pack(push, 1)
struct NeuronState {
    int16_t  v;        // Membrane potential (Q5.10)
    int16_t  v_th;     // Threshold (Q5.10)
    uint16_t flags;    // Status flags
};
#pragma pack(pop)

static_assert(sizeof(NeuronState) == 6, "NeuronState must be 6 bytes");

// Flag accessors
inline bool neuron_spiked(const NeuronState& n) { return n.flags & 0x0001; }
inline bool neuron_refractory(const NeuronState& n) { return n.flags & 0x0002; }
inline uint8_t neuron_ref_count(const NeuronState& n) { return (n.flags >> 2) & 0x3F; }

// =============================================================================
// File I/O Utilities
// =============================================================================

std::vector<uint8_t> load_binary_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(size);
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("Failed to read file: " + path);
    }

    return data;
}

void save_binary_file(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to create file: " + path);
    }
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
}

// =============================================================================
// Simple JSON Parser (minimal, for topology file)
// In production, use nlohmann/json or rapidjson
// =============================================================================

std::string extract_json_string(const std::string& json, const std::string& key) {
    size_t pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";

    pos = json.find(":", pos);
    if (pos == std::string::npos) return "";

    pos = json.find("\"", pos);
    if (pos == std::string::npos) return "";

    size_t end = json.find("\"", pos + 1);
    if (end == std::string::npos) return "";

    return json.substr(pos + 1, end - pos - 1);
}

uint32_t extract_json_uint(const std::string& json, const std::string& key) {
    size_t pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return 0;

    pos = json.find(":", pos);
    if (pos == std::string::npos) return 0;

    // Skip whitespace
    while (pos < json.size() && (json[pos] == ':' || json[pos] == ' ')) pos++;

    return static_cast<uint32_t>(std::stoul(json.substr(pos)));
}

FabricTopology parse_topology(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open topology file: " + path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();

    FabricTopology topo;
    topo.version = extract_json_uint(json, "version");
    topo.endianness = extract_json_string(json, "endianness");
    topo.total_neurons = extract_json_uint(json, "total_neurons");
    topo.total_synapses = extract_json_uint(json, "total_synapses");

    // Fixed-point config
    topo.fixed_point.v_bits = extract_json_uint(json, "v_bits");
    topo.fixed_point.v_frac_bits = extract_json_uint(json, "v_frac_bits");
    topo.fixed_point.w_bits = extract_json_uint(json, "w_bits");
    topo.fixed_point.w_frac_bits = extract_json_uint(json, "w_frac_bits");
    topo.fixed_point.param_bits = extract_json_uint(json, "param_bits");
    topo.fixed_point.param_frac_bits = extract_json_uint(json, "param_frac_bits");

    // Neuron layout
    topo.neuron_layout.record_size_bytes = extract_json_uint(json, "record_size_bytes");
    topo.neuron_layout.record_count = extract_json_uint(json, "record_count");

    // NOTE: Full projection/population parsing would require proper JSON library
    // This is a minimal implementation for demonstration

    std::cout << "[INFO] Parsed topology: " << topo.total_neurons << " neurons, "
              << topo.total_synapses << " synapses\n";

    return topo;
}

// =============================================================================
// XRT Host Application
// =============================================================================

class SNNFPGARunner {
public:
    SNNFPGARunner(const std::string& xclbin_path, int device_id = 0)
        : device_(device_id)
    {
        std::cout << "[INFO] Loading xclbin: " << xclbin_path << "\n";
        auto uuid = device_.load_xclbin(xclbin_path);

        // Get kernel handle
        kernel_ = xrt::kernel(device_, uuid, "snn_kernel");
        std::cout << "[INFO] Kernel 'snn_kernel' loaded\n";
    }

    void load_fabric(const std::string& topology_path,
                     const std::string& weights_path,
                     const std::string& neurons_path)
    {
        // Parse topology
        topology_ = parse_topology(topology_path);

        // Load binary files
        weights_data_ = load_binary_file(weights_path);
        neurons_data_ = load_binary_file(neurons_path);

        std::cout << "[INFO] Loaded weights: " << weights_data_.size() << " bytes\n";
        std::cout << "[INFO] Loaded neurons: " << neurons_data_.size() << " bytes\n";

        // Validate neuron data size
        size_t expected_neurons_size = topology_.total_neurons * sizeof(NeuronState);
        if (neurons_data_.size() != expected_neurons_size) {
            std::cerr << "[WARN] Neuron data size mismatch: got "
                      << neurons_data_.size() << ", expected " << expected_neurons_size << "\n";
        }

        // Allocate device buffers
        allocate_buffers();
    }

    void allocate_buffers() {
        // Buffer group IDs (must match kernel arguments)
        constexpr int GRP_WEIGHTS = 0;
        constexpr int GRP_NEURONS = 1;
        constexpr int GRP_SPIKES_IN = 2;
        constexpr int GRP_SPIKES_OUT = 3;
        constexpr int GRP_I_POST = 4;

        // Weights buffer (CSR data for all projections)
        bo_weights_ = xrt::bo(device_, weights_data_.size(), kernel_.group_id(GRP_WEIGHTS));

        // Neuron state buffer
        bo_neurons_ = xrt::bo(device_, neurons_data_.size(), kernel_.group_id(GRP_NEURONS));

        // Spike input buffer (packed bits: N_neurons / 8 bytes)
        size_t spike_buf_size = (topology_.total_neurons + 7) / 8;
        bo_spikes_in_ = xrt::bo(device_, spike_buf_size, kernel_.group_id(GRP_SPIKES_IN));
        bo_spikes_out_ = xrt::bo(device_, spike_buf_size, kernel_.group_id(GRP_SPIKES_OUT));

        // I_post accumulator buffer (N_post * 4 bytes for Q15.16)
        bo_i_post_ = xrt::bo(device_, topology_.total_neurons * 4, kernel_.group_id(GRP_I_POST));

        // Copy initial data to device
        bo_weights_.write(weights_data_.data());
        bo_weights_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        bo_neurons_.write(neurons_data_.data());
        bo_neurons_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Zero I_post buffer
        std::vector<uint8_t> zeros(topology_.total_neurons * 4, 0);
        bo_i_post_.write(zeros.data());
        bo_i_post_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        std::cout << "[INFO] Device buffers allocated and initialized\n";
    }

    std::vector<uint8_t> run_timestep(const std::vector<uint8_t>& input_spikes) {
        // Validate input size
        size_t expected_size = (topology_.total_neurons + 7) / 8;
        if (input_spikes.size() != expected_size) {
            throw std::runtime_error("Input spike buffer size mismatch");
        }

        // Copy input spikes to device
        bo_spikes_in_.write(input_spikes.data());
        bo_spikes_in_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Launch kernel
        auto run = kernel_(
            bo_weights_,
            bo_neurons_,
            bo_spikes_in_,
            bo_spikes_out_,
            bo_i_post_,
            topology_.total_neurons
        );
        run.wait();

        // Read output spikes
        bo_spikes_out_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::vector<uint8_t> output_spikes(expected_size);
        bo_spikes_out_.read(output_spikes.data());

        return output_spikes;
    }

    void run_simulation(const std::vector<std::vector<uint8_t>>& spike_trains,
                        std::vector<std::vector<uint8_t>>& output_trains)
    {
        size_t T = spike_trains.size();
        output_trains.resize(T);

        std::cout << "[INFO] Running " << T << " timesteps...\n";

        auto start = std::chrono::high_resolution_clock::now();

        for (size_t t = 0; t < T; t++) {
            output_trains[t] = run_timestep(spike_trains[t]);

            if ((t + 1) % 100 == 0) {
                std::cout << "  [" << (t + 1) << "/" << T << "]\n";
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        double ms_total = duration.count() / 1000.0;
        double ms_per_step = ms_total / T;

        std::cout << "[INFO] Simulation complete\n";
        std::cout << "  Total time:    " << ms_total << " ms\n";
        std::cout << "  Per timestep:  " << ms_per_step << " ms\n";
        std::cout << "  Throughput:    " << (1000.0 / ms_per_step) << " steps/sec\n";
    }

    void read_neuron_states(std::vector<NeuronState>& states) {
        bo_neurons_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        states.resize(topology_.total_neurons);
        bo_neurons_.read(states.data());
    }

    const FabricTopology& topology() const { return topology_; }

private:
    xrt::device device_;
    xrt::kernel kernel_;
    FabricTopology topology_;

    std::vector<uint8_t> weights_data_;
    std::vector<uint8_t> neurons_data_;

    xrt::bo bo_weights_;
    xrt::bo bo_neurons_;
    xrt::bo bo_spikes_in_;
    xrt::bo bo_spikes_out_;
    xrt::bo bo_i_post_;
};

// =============================================================================
// CLI
// =============================================================================

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --xclbin PATH       Path to xclbin file (required)\n"
              << "  --topology PATH     Path to fabric_topology.json (required)\n"
              << "  --weights PATH      Path to weights.bin (required)\n"
              << "  --neurons PATH      Path to neurons.bin (required)\n"
              << "  --spikes_in PATH    Path to input spike trains (optional)\n"
              << "  --spikes_out PATH   Path to save output spikes (optional)\n"
              << "  --timesteps N       Number of timesteps to simulate (default: 100)\n"
              << "  --device ID         Device ID (default: 0)\n"
              << "  --benchmark         Run latency benchmark\n"
              << "  --help              Show this help\n";
}

int main(int argc, char* argv[]) {
    std::string xclbin_path;
    std::string topology_path;
    std::string weights_path;
    std::string neurons_path;
    std::string spikes_in_path;
    std::string spikes_out_path;
    int timesteps = 100;
    int device_id = 0;
    bool benchmark = false;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--xclbin" && i + 1 < argc) {
            xclbin_path = argv[++i];
        } else if (arg == "--topology" && i + 1 < argc) {
            topology_path = argv[++i];
        } else if (arg == "--weights" && i + 1 < argc) {
            weights_path = argv[++i];
        } else if (arg == "--neurons" && i + 1 < argc) {
            neurons_path = argv[++i];
        } else if (arg == "--spikes_in" && i + 1 < argc) {
            spikes_in_path = argv[++i];
        } else if (arg == "--spikes_out" && i + 1 < argc) {
            spikes_out_path = argv[++i];
        } else if (arg == "--timesteps" && i + 1 < argc) {
            timesteps = std::stoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            device_id = std::stoi(argv[++i]);
        } else if (arg == "--benchmark") {
            benchmark = true;
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate required arguments
    if (xclbin_path.empty() || topology_path.empty() ||
        weights_path.empty() || neurons_path.empty()) {
        std::cerr << "Error: Missing required arguments\n";
        print_usage(argv[0]);
        return 1;
    }

    try {
        // Initialize runner
        SNNFPGARunner runner(xclbin_path, device_id);
        runner.load_fabric(topology_path, weights_path, neurons_path);

        const auto& topo = runner.topology();
        size_t spike_buf_size = (topo.total_neurons + 7) / 8;

        // Prepare input spike trains
        std::vector<std::vector<uint8_t>> input_trains(timesteps);

        if (!spikes_in_path.empty()) {
            // Load from file
            auto spike_data = load_binary_file(spikes_in_path);
            size_t expected = spike_buf_size * timesteps;
            if (spike_data.size() != expected) {
                std::cerr << "[WARN] Spike input size mismatch: "
                          << spike_data.size() << " vs " << expected << "\n";
            }

            for (int t = 0; t < timesteps && t * spike_buf_size < spike_data.size(); t++) {
                input_trains[t].assign(
                    spike_data.begin() + t * spike_buf_size,
                    spike_data.begin() + (t + 1) * spike_buf_size
                );
            }
        } else {
            // Generate random input for benchmark
            std::cout << "[INFO] No input spikes provided, generating random input\n";
            for (int t = 0; t < timesteps; t++) {
                input_trains[t].resize(spike_buf_size);
                for (size_t i = 0; i < spike_buf_size; i++) {
                    // ~10% spike probability
                    input_trains[t][i] = (rand() % 100 < 10) ? 0xFF : 0x00;
                }
            }
        }

        // Run simulation
        std::vector<std::vector<uint8_t>> output_trains;
        runner.run_simulation(input_trains, output_trains);

        // Save output spikes if path provided
        if (!spikes_out_path.empty()) {
            std::vector<uint8_t> flat_output;
            for (const auto& train : output_trains) {
                flat_output.insert(flat_output.end(), train.begin(), train.end());
            }
            save_binary_file(spikes_out_path, flat_output);
            std::cout << "[INFO] Saved output spikes to " << spikes_out_path << "\n";
        }

        // Count total output spikes
        size_t total_spikes = 0;
        for (const auto& train : output_trains) {
            for (uint8_t byte : train) {
                total_spikes += __builtin_popcount(byte);
            }
        }
        std::cout << "[INFO] Total output spikes: " << total_spikes << "\n";
        std::cout << "[INFO] Average spikes/timestep: "
                  << (double)total_spikes / timesteps << "\n";

        // Benchmark mode: run additional iterations
        if (benchmark) {
            std::cout << "\n[BENCHMARK] Running latency measurement...\n";

            constexpr int WARMUP = 10;
            constexpr int ITERS = 100;

            // Warmup
            for (int i = 0; i < WARMUP; i++) {
                runner.run_timestep(input_trains[0]);
            }

            // Timed iterations
            std::vector<double> latencies;
            for (int i = 0; i < ITERS; i++) {
                auto start = std::chrono::high_resolution_clock::now();
                runner.run_timestep(input_trains[i % timesteps]);
                auto end = std::chrono::high_resolution_clock::now();

                auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                latencies.push_back(us.count() / 1000.0);
            }

            // Compute stats
            std::sort(latencies.begin(), latencies.end());
            double sum = 0;
            for (double l : latencies) sum += l;
            double mean = sum / latencies.size();
            double p50 = latencies[latencies.size() / 2];
            double p95 = latencies[latencies.size() * 95 / 100];
            double p99 = latencies[latencies.size() * 99 / 100];

            std::cout << "[BENCHMARK] Results (" << ITERS << " iterations):\n";
            std::cout << "  Mean latency:  " << mean << " ms\n";
            std::cout << "  P50 latency:   " << p50 << " ms\n";
            std::cout << "  P95 latency:   " << p95 << " ms\n";
            std::cout << "  P99 latency:   " << p99 << " ms\n";
        }

        std::cout << "\n[DONE]\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}

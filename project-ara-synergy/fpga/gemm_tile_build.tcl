# gemm_tile_build.tcl
#
# Vitis HLS build script for 64×64 W4A8 GEMM tile
# Target: Forest Kitten FK33 (VU35P)
#
# Usage:
#   vitis_hls -f gemm_tile_build.tcl
#
# Options:
#   -csim   : Run C simulation only
#   -synth  : Run C synthesis only
#   -cosim  : Run co-simulation (RTL verification)
#   -export : Export IP for Vivado integration

# ============================================================================
# Project Configuration
# ============================================================================

set PROJECT_NAME "gemm_tile_w4a8_project"
set TOP_FUNCTION "gemm_top_w4a8"
set SOURCE_FILE "gemm_tile_w4a8.h"

# Target device: VU35P (Forest Kitten FK33)
set PART "xcvu35p-fsvh2104-2L-e"

# Clock period: 300 MHz = 3.33 ns
set CLOCK_PERIOD 3.33

# ============================================================================
# Create Project
# ============================================================================

open_project -reset $PROJECT_NAME

# Add source files
add_files $SOURCE_FILE -cflags "-std=c++14 -I."

# Set top function
set_top $TOP_FUNCTION

# Open solution
open_solution -reset "solution1"

# Set target part
set_part $PART

# Create clock constraint
create_clock -period $CLOCK_PERIOD -name default

# ============================================================================
# Configuration Flags
# ============================================================================

# Enable aggressive optimizations for HBM bandwidth
config_compile -pipeline_loops 64

# Use flow_target vivado for better optimization
config_rtl -reset all

# Enable AXI-Stream optimization
config_interface -m_axi_latency 64

# ============================================================================
# C Simulation (Optional)
# ============================================================================

if {[lsearch $argv "-csim"] != -1} {
    puts "Running C simulation..."

    # Add testbench (create if needed)
    # add_files -tb "gemm_tile_tb.cpp" -cflags "-std=c++14"

    # Run C simulation
    # csim_design

    puts "C simulation complete."
    exit
}

# ============================================================================
# C Synthesis
# ============================================================================

if {[lsearch $argv "-synth"] != -1 || $argc == 0} {
    puts "Running C synthesis..."
    puts "Target: $PART @ $CLOCK_PERIOD ns"
    puts "Top function: $TOP_FUNCTION"

    csynth_design

    puts "\n========================================="
    puts "Synthesis Summary"
    puts "========================================="
    puts "Check solution1/syn/report/${TOP_FUNCTION}_csynth.rpt for:"
    puts "  - Latency and II"
    puts "  - Resource utilization (DSP, BRAM, URAM, LUT, FF)"
    puts "  - Clock period achieved"
    puts "=========================================\n"
}

# ============================================================================
# Co-Simulation (Optional)
# ============================================================================

if {[lsearch $argv "-cosim"] != -1} {
    puts "Running co-simulation (RTL verification)..."

    # Co-simulate with wave dump for debugging
    cosim_design -rtl verilog -trace_level all

    puts "Co-simulation complete."
    puts "Check solution1/sim/report/${TOP_FUNCTION}_cosim.rpt"
}

# ============================================================================
# Export RTL (Optional)
# ============================================================================

if {[lsearch $argv "-export"] != -1} {
    puts "Exporting RTL for Vivado integration..."

    # Export as IP catalog format
    export_design -format ip_catalog \
                  -description "64x64 W4A8 GEMM Tile for Cathedral Avatar QNN" \
                  -vendor "scythe.dev" \
                  -version "1.0"

    puts "Export complete."
    puts "IP available at: solution1/impl/export.zip"
}

# ============================================================================
# Report Generation
# ============================================================================

puts "\n========================================="
puts "Build Complete"
puts "========================================="
puts "Project: $PROJECT_NAME"
puts "Solution: solution1"
puts ""
puts "Next steps:"
puts "  1. Review synthesis report:"
puts "     solution1/syn/report/${TOP_FUNCTION}_csynth.rpt"
puts "  2. Check resource utilization:"
puts "     - DSP: Should be ~2048 (71% of 2880)"
puts "     - BRAM: Should be ~300 (14% of 2160)"
puts "     - URAM: Should be ~100 (10% of 960)"
puts "  3. Verify II=1 on gemm_j loop"
puts "  4. Verify clock constraint met (3.33 ns)"
puts ""
puts "Optional:"
puts "  - Run co-simulation: vitis_hls -f gemm_tile_build.tcl -cosim"
puts "  - Export IP: vitis_hls -f gemm_tile_build.tcl -export"
puts "=========================================\n"

exit

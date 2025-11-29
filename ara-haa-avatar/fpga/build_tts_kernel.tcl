#!/usr/bin/tclsh
#
# Vitis HLS Build Script for TTS Kernel
# Target: SQRL Forest Kitten (Xilinx VU35P + HBM2)
#
# Usage:
#   vitis_hls -f build_tts_kernel.tcl
#

# ============================================================================
# Configuration
# ============================================================================

set PROJECT_NAME "tts_kernel_hls"
set TOP_FUNCTION "tts_kernel"
set SOURCE_FILE "tts_kernel_hls.cpp"

# Target FPGA part (VU35P on Forest Kitten)
set FPGA_PART "xvu35p-fsvh2104-2-e"

# Clock period (target 250 MHz = 4ns)
set CLOCK_PERIOD 4.0

# ============================================================================
# Create Project
# ============================================================================

# Open or create project
open_project -reset ${PROJECT_NAME}

# Add source files
add_files ${SOURCE_FILE}

# Set top-level function
set_top ${TOP_FUNCTION}

# Open solution
open_solution -reset "solution1"

# Set target FPGA part
set_part ${FPGA_PART}

# Create clock constraint
create_clock -period ${CLOCK_PERIOD} -name default

# ============================================================================
# Optimization Directives
# ============================================================================

# These are additional optimization hints beyond the pragmas in the source

# Enable dataflow optimization
config_dataflow -strict_mode warning

# Enable aggressive loop optimizations
config_schedule -enable_dsp_full_reg=true

# HBM2 interface configuration
config_interface -m_axi_addr64

# ============================================================================
# Run Synthesis
# ============================================================================

puts "=================================="
puts "Starting C Synthesis..."
puts "=================================="

csynth_design

# ============================================================================
# Run Co-simulation (Optional)
# ============================================================================

# Uncomment to run co-simulation (requires test bench)
# puts "=================================="
# puts "Starting Co-simulation..."
# puts "=================================="
# cosim_design -rtl verilog -trace_level all

# ============================================================================
# Export Design
# ============================================================================

puts "=================================="
puts "Exporting RTL Design..."
puts "=================================="

export_design -format ip_catalog \
              -description "TTS Acceleration Kernel for Cathedral Avatar" \
              -vendor "cathedral" \
              -library "avatar" \
              -version "1.0" \
              -display_name "TTS Kernel HLS"

# ============================================================================
# Generate Reports
# ============================================================================

puts "=================================="
puts "Build Complete!"
puts "=================================="
puts ""
puts "Output files:"
puts "  - RTL:       ${PROJECT_NAME}/solution1/syn/verilog/"
puts "  - IP:        ${PROJECT_NAME}/solution1/impl/ip/"
puts "  - Reports:   ${PROJECT_NAME}/solution1/syn/report/"
puts ""
puts "Next steps:"
puts "  1. Review synthesis report:"
puts "     ${PROJECT_NAME}/solution1/syn/report/${TOP_FUNCTION}_csynth.rpt"
puts ""
puts "  2. Integrate IP into Vivado project for bitstream generation"
puts ""
puts "  3. Program SQRL Forest Kitten with generated bitstream"
puts ""
puts "=================================="

# Exit
exit

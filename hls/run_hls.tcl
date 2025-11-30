# =============================================================================
# run_hls.tcl
#
# Vitis HLS synthesis script for Kitten SNN kernel
#
# Usage:
#   vitis_hls -f run_hls.tcl
#
# =============================================================================

# Project settings
set PROJ_NAME "kitten_snn_hls"
set SOLUTION_NAME "solution1"
set TOP_FUNC "snn_core"

# Target device (Alveo U50)
set PART "xcu50-fsvh2104-2-e"
set CLOCK_PERIOD "3.33"  ;# 300 MHz

# =============================================================================
# Create/open project
# =============================================================================

open_project -reset $PROJ_NAME
set_top $TOP_FUNC
add_files snn_core_hls.cpp

# =============================================================================
# Create solution with target device
# =============================================================================

open_solution -reset $SOLUTION_NAME -flow_target vitis
set_part $PART
create_clock -period $CLOCK_PERIOD -name default

# =============================================================================
# Synthesis directives
# =============================================================================

# Memory interface configuration
config_interface -m_axi_alignment_byte_size 64
config_interface -m_axi_latency 64
config_interface -m_axi_max_widen_bitwidth 512

# Dataflow/pipeline settings
config_compile -pipeline_loops 64
config_schedule -effort high

# =============================================================================
# Run C synthesis
# =============================================================================

puts "Running C synthesis..."
csynth_design

# =============================================================================
# Reports
# =============================================================================

puts ""
puts "==================================================================="
puts "HLS Synthesis Complete"
puts "==================================================================="
puts ""
puts "Reports available in:"
puts "  $PROJ_NAME/$SOLUTION_NAME/syn/report/"
puts ""
puts "Key metrics to check:"
puts "  - Estimated clock period (target: $CLOCK_PERIOD ns)"
puts "  - Resource utilization (LUT, FF, BRAM, DSP)"
puts "  - Loop initiation interval (II)"
puts ""

# Optional: Export IP or run co-simulation
# export_design -flow syn -rtl verilog
# cosim_design

exit

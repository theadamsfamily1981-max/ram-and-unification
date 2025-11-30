# =============================================================================
# Makefile for Kitten SNN HLS Kernel
# =============================================================================

# Vitis/Vivado Installation
XILINX_VITIS ?= /opt/Xilinx/Vitis/2023.2
XILINX_HLS   ?= $(XILINX_VITIS)/Vitis_HLS

# Target platform (modify for your board)
PLATFORM ?= xilinx_u50_gen3x16_xdma_5_202210_1

# Kernel settings
KERNEL_NAME := snn_core
KERNEL_SRC  := snn_core_hls.cpp
XO_FILE     := $(KERNEL_NAME).xo
XCLBIN_FILE := $(KERNEL_NAME).xclbin

# Build target: sw_emu, hw_emu, or hw
TARGET ?= hw_emu

# Clock frequency (MHz)
CLOCK_FREQ ?= 300

# =============================================================================
# V++ Flags
# =============================================================================

VPP := v++

VPP_COMMON := --platform $(PLATFORM) \
              --save-temps \
              --log_dir ./logs \
              --temp_dir ./tmp

VPP_COMPILE := $(VPP_COMMON) \
               -c \
               -t $(TARGET) \
               -k $(KERNEL_NAME) \
               --hls.clock $(CLOCK_FREQ)000000:$(KERNEL_NAME)

VPP_LINK := $(VPP_COMMON) \
            -l \
            -t $(TARGET) \
            --clock.freqHz $(CLOCK_FREQ)000000:$(KERNEL_NAME)

# =============================================================================
# Targets
# =============================================================================

.PHONY: all compile link clean hls_csynth help

all: $(XCLBIN_FILE)

# Compile kernel to .xo
compile: $(XO_FILE)

$(XO_FILE): $(KERNEL_SRC)
	mkdir -p logs tmp
	$(VPP) $(VPP_COMPILE) -o $@ $<

# Link .xo to .xclbin
link: $(XCLBIN_FILE)

$(XCLBIN_FILE): $(XO_FILE)
	$(VPP) $(VPP_LINK) -o $@ $<

# Run Vitis HLS C-synthesis only (for quick iteration)
hls_csynth: $(KERNEL_SRC)
	mkdir -p hls_project
	vitis_hls -f run_hls.tcl

clean:
	rm -rf *.xo *.xclbin *.log *.info
	rm -rf logs tmp hls_project
	rm -rf _x .Xil

# =============================================================================
# Help
# =============================================================================

help:
	@echo "Kitten SNN HLS Kernel Build"
	@echo ""
	@echo "Targets:"
	@echo "  all         Build complete xclbin (default)"
	@echo "  compile     Compile kernel to .xo only"
	@echo "  link        Link .xo to .xclbin"
	@echo "  hls_csynth  Run HLS C-synthesis only (fast)"
	@echo "  clean       Remove build artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  TARGET      sw_emu, hw_emu, or hw (default: hw_emu)"
	@echo "  PLATFORM    Target FPGA platform"
	@echo "  CLOCK_FREQ  Target clock in MHz (default: 300)"
	@echo ""
	@echo "Examples:"
	@echo "  make TARGET=sw_emu          # Software emulation"
	@echo "  make TARGET=hw_emu          # Hardware emulation"
	@echo "  make TARGET=hw              # Full hardware build"
	@echo "  make hls_csynth             # HLS synthesis only"

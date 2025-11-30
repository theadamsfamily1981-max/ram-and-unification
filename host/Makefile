# =============================================================================
# Makefile for Kitten SNN FPGA Host Application
# =============================================================================

# XRT Installation (set via environment or modify here)
XILINX_XRT ?= /opt/xilinx/xrt

# Compiler settings
CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra
CXXFLAGS += -I$(XILINX_XRT)/include

LDFLAGS := -L$(XILINX_XRT)/lib
LDLIBS := -lxrt_coreutil -pthread

# Targets
TARGET := run_snn_fpga

SRCS := run_snn_fpga.cpp
OBJS := $(SRCS:.cpp=.o)

# =============================================================================
# Rules
# =============================================================================

.PHONY: all clean help

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(TARGET) $(OBJS)

# =============================================================================
# Help
# =============================================================================

help:
	@echo "Kitten SNN FPGA Host Application"
	@echo ""
	@echo "Targets:"
	@echo "  all      Build the host application (default)"
	@echo "  clean    Remove build artifacts"
	@echo "  help     Show this help"
	@echo ""
	@echo "Environment:"
	@echo "  XILINX_XRT    Path to XRT installation (default: /opt/xilinx/xrt)"
	@echo ""
	@echo "Example:"
	@echo "  make XILINX_XRT=/tools/Xilinx/XRT"
	@echo ""
	@echo "Usage:"
	@echo "  ./run_snn_fpga --xclbin snn_kernel.xclbin \\"
	@echo "                 --topology build/fpga/fabric_topology.json \\"
	@echo "                 --weights build/fpga/weights.bin \\"
	@echo "                 --neurons build/fpga/neurons.bin \\"
	@echo "                 --timesteps 100"

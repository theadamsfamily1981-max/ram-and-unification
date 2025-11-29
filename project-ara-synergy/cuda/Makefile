CUDA_SRC = avatar_inference.cu
BUILD_DIR = ../build
LIB_NAME = libavatar_qnn.so

NVCC ?= nvcc
NVCCFLAGS = -O3 -Xcompiler -fPIC -std=c++17

all: $(BUILD_DIR)/$(LIB_NAME)

$(BUILD_DIR)/$(LIB_NAME): $(CUDA_SRC)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared $< -o $@

clean:
	rm -f $(BUILD_DIR)/$(LIB_NAME)

.PHONY: all clean

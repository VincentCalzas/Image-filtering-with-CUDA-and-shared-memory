# ============================================================================
# Lab 5 - TDDD56: CUDA Image Filtering
# ============================================================================

NVCC       = nvcc
LDFLAGS    = -lGL -lglut
COMMON_SRC = src/common/milli.cpp src/common/readppm.cpp

# Output binaries (at project root for easy execution)
all: naive shared separable gaussian gaussian_ns median

naive: src/naive/filter.cu $(COMMON_SRC)
	$(NVCC) $^ -o filter $(LDFLAGS)

shared: src/shared/filter_shared.cu $(COMMON_SRC)
	$(NVCC) $^ -o filter_shared $(LDFLAGS)

separable: src/separable/filter_separable.cu $(COMMON_SRC)
	$(NVCC) $^ -o filter_separable $(LDFLAGS)

gaussian: src/gaussian/filter_gaussian.cu $(COMMON_SRC)
	$(NVCC) $^ -o filter_gaussian $(LDFLAGS)

gaussian_ns: src/gaussian/filter_gaussian_non_separable.cu $(COMMON_SRC)
	$(NVCC) $^ -o filter_gaussian_non_separable $(LDFLAGS)

median: src/median/filter_median.cu $(COMMON_SRC)
	$(NVCC) $^ -o filter_median $(LDFLAGS)

clean:
	rm -f filter filter_shared filter_separable filter_gaussian filter_gaussian_non_separable filter_median

.PHONY: all naive shared separable gaussian gaussian_ns median clean

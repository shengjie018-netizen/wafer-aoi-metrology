CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -fopenmp -DWAFER_AOI_OPENMP=1 -Iinclude
LDFLAGS ?= -fopenmp -pthread

ifeq ($(OS),Windows_NT)
    EXE_EXT := .exe
    MKDIR_P := powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path build/obj/core,build/obj/algo,build/bin | Out-Null"
    RM_RF := powershell -NoProfile -Command "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue build"
else
    EXE_EXT :=
    MKDIR_P := mkdir -p build/obj/core build/obj/algo build/bin
    RM_RF := rm -rf build
endif

LIB_SRCS := \
    src/core/image.cpp \
    src/core/image_ops.cpp \
    src/algo/shape_matcher.cpp \
    src/algo/subpixel_metrology.cpp \
    src/algo/d2d_inspector.cpp

LIB_OBJS := $(patsubst src/%.cpp,build/obj/%.o,$(LIB_SRCS))

DEMO_BIN := build/bin/wafer_aoi_demo$(EXE_EXT)
TEST_BIN := build/bin/wafer_aoi_tests$(EXE_EXT)

.PHONY: all dirs test clean

all: dirs $(DEMO_BIN) $(TEST_BIN)

dirs:
	@$(MKDIR_P)

build/obj/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(DEMO_BIN): $(LIB_OBJS) src/main_demo.cpp
	$(CXX) $(CXXFLAGS) $(LIB_OBJS) src/main_demo.cpp -o $@ $(LDFLAGS)

$(TEST_BIN): $(LIB_OBJS) tests/test_all.cpp
	$(CXX) $(CXXFLAGS) $(LIB_OBJS) tests/test_all.cpp -o $@ $(LDFLAGS)

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	@$(RM_RF)

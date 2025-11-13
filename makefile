# Detect Operating System
ifeq ($(OS),Windows_NT)
    RM := del /F /Q
    EXE_EXT := .exe
else
    RM := rm -f
    EXE_EXT :=
endif

# ==== CONFIGURABLE BUILD MODE ====
CUDA ?= 0

CXX := clang++

# ==== FLAGS ====
CXXFLAGS := -O3 -std=c++20 -DNDEBUG
GPUFLAGS :=

ifeq ($(OS),Windows_NT)
  ARCH := $(PROCESSOR_ARCHITECTURE)
else
  ARCH := $(shell uname -m)
endif

IS_ARM := $(filter ARM arm64 aarch64 arm%,$(ARCH))

ifeq ($(IS_ARM),)
  LINKFLAGS := -fuse-ld=lld -pthread -fopenmp -lopenblas
  ARCHFLAGS := -march=native
else
  LINKFLAGS := -pthread -fopenmp -lopenblas
  ARCHFLAGS := -mcpu=native
endif

# ==== FILES ====
EXE      ?= Ember$(EXE_EXT)
SRCS     := $(wildcard ./src/*.cpp) $(wildcard ./src/*.cu) $(wildcard ./src/*/*.cpp) $(wildcard ./src/*/*.cu) ./external/fmt/format.cpp
OBJS     := $(SRCS:.cpp=.o)
OBJS     := $(OBJS:.cu=.o)
OBJS     := $(OBJS:.cu=.o)
DEPS     := $(OBJS:.o=.d)


# ==== BUILD ====
all: $(EXE)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) -funroll-loops -flto -Wall -Wextra -MMD -MP -c $< -o $@

%.o: %.cu
ifeq ($(CUDA),1)
	nvcc -ccbin=clang++ -DEMBER_CUDA -DFMT_USE_BITINT=0 $(CXXFLAGS) -MMD -MP -c $< -o $@ $(GPUFLAGS)
else
	$(CXX) -x c++ $(CXXFLAGS) $(ARCHFLAGS) -flto -MMD -MP -c $< -o $@
endif

-include $(DEPS)

$(EXE): $(OBJS)
ifeq ($(CUDA),1)
	$(CXX) $(CXXFLAGS) $(OBJS) $(SHAREDLNKFLAGS) $(LINKFLAGS) -flto -L/usr/local/cuda/lib64 -lcudart -lcublas -o $@
else
	$(CXX) $(CXXFLAGS) $(OBJS) $(SHAREDLNKFLAGS) $(LINKFLAGS) -flto -o $@
endif

# ==== CLEAN ====
CLEAN_STUFF := $(EXE) Ember.exp Ember.lib Ember.pdb $(OBJS) $(DEPS)
ifeq ($(OS),Windows_NT)
    CLEAN_STUFF := $(subst /,\\,$(CLEAN_STUFF))
endif

.PHONY: clean
clean:
	$(RM) $(CLEAN_STUFF)

# ==== FLAGS ====
BASE_CXXFLAGS := -std=c++20
RELEASE_FLAGS := -O3 -DNDEBUG
DEBUG_FLAGS   := -O0 -g -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC
SAN_FLAGS     := -O2 -g -fsanitize=address,undefined
PROFILE_FLAGS := -O3 -g -DNDEBUG

# Default build is release
CXXFLAGS := $(BASE_CXXFLAGS) $(RELEASE_FLAGS)
GPUFLAGS :=

# ==== BUILD MODES ====
.PHONY: all debug sanitize profile clean force

all: $(EXE)

debug: CXXFLAGS := $(BASE_CXXFLAGS) $(DEBUG_FLAGS)
debug: LINKFLAGS += -rdynamic
debug: GPUFLAGS := -G $(GPUFLAGS)
debug: all

sanitize: CXXFLAGS := $(BASE_CXXFLAGS) $(SAN_FLAGS)
sanitize: LINKFLAGS += -rdynamic
sanitize: GPUFLAGS := -G $(GPUFLAGS)
sanitize: all

profile: CXXFLAGS := $(BASE_CXXFLAGS) $(PROFILE_FLAGS)
profile: GPUFLAGS := -G $(GPUFLAGS)
profile: all
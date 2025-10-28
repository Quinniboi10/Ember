# Detect Operating System
ifeq ($(OS),Windows_NT)
    # Windows settings
    RM := del /F /Q
    EXE_EXT := .exe
else
    # Unix/Linux settings
    RM := rm -f
    EXE_EXT :=
endif

# Compiler and flags
CXX      := clang++
CXXFLAGS := -O3 -std=c++20 -flto -funroll-loops -DNDEBUG

ifeq ($(OS),Windows_NT)
  ARCH := $(PROCESSOR_ARCHITECTURE)
else
  ARCH := $(shell uname -m)
endif

IS_ARM := $(filter ARM arm64 aarch64 arm%,$(ARCH))

ifeq ($(IS_ARM),)
  LINKFLAGS := -fuse-ld=lld -pthread -lopenblas -fopenmp
  ARCHFLAGS := -march=native
else
  LINKFLAGS :=
  ARCHFLAGS := -mcpu=native
endif

# Default target executable name
EXE      ?= Ember$(EXE_EXT)

# Source and object files
SRCS     := $(wildcard ./src/*.cpp)
SRCS     += ./external/fmt/format.cpp
OBJS     := $(SRCS:.cpp=.o)
DEPS     := $(OBJS:.o=.d)

# Default target
all: $(EXE)

# Build the objects
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) -c $< -o $@

-include $(DEPS)

# Link the executable
$(EXE): $(OBJS)
	$(CXX) $(CXXFLAGS) -MMD -MP $(OBJS) $(LINKFLAGS) -o $@

# Files for make clean
CLEAN_STUFF := $(EXE) Ember.exp Ember.lib Ember.pdb $(OBJS) $(DEPS)
ifeq ($(OS),Windows_NT)
    CLEAN_STUFF := $(subst /,\\,$(CLEAN_STUFF))
endif

# Debug build
.PHONY: debug
debug: CXXFLAGS = -O3 -std=c++20 -flto -fsanitize=address,undefined -fno-omit-frame-pointer -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -Wall -Wextra
debug: all

# Debug build
.PHONY: profile
profile: CXXFLAGS = -O3 -std=c++20 -flto -funroll-loops -ggdb -fno-omit-frame-pointer -DNDEBUG
profile: all

# Force rebuild
.PHONY: force
force: clean
force: all

# Clean up
.PHONY: clean
clean:
	$(RM) $(CLEAN_STUFF)
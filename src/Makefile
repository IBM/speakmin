# Copyright contributors to the speakmin project
# SPDX-License-Identifier: Apache-2.0

# Get short git revision
GIT_REV := $(shell git rev-parse --short HEAD)

# Compiler and flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # Check if Homebrew is installed
    HOMEBREW := $(shell command -v brew 2>/dev/null)
    ifeq ($(HOMEBREW),)
        $(error Homebrew not found! Please install Homebrew from https://brew.sh/)
    endif

    # Find latest g++ version installed via Homebrew
    GPP_BIN := $(shell ls $(shell brew --prefix)/bin | grep -E '^g\+\+\-[0-9]+$$' | sort -V | tail -n 1)
    GPP_PATH := $(shell brew --prefix)/bin/$(GPP_BIN)
    ifeq ("$(wildcard $(GPP_PATH))","")
        $(error No Homebrew-installed g++ found! Please install it using: brew install gcc)
    endif

    CXX := $(GPP_PATH)
else
    CXX := g++
endif
CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -O2 -g -DREFRACTORY -fopenmp -D__GIT_REV__=\"$(GIT_REV)\"
LDFLAGS := -fopenmp -lstdc++fs

# Default to TRAIN_FA if TRAIN_MODE is not specified
TRAIN_MODE ?= DFA

# Set the TRAIN macro based on the TRAIN_MODE value
ifeq ($(TRAIN_MODE), DFA)
    CXXFLAGS += -DTRAIN_DFA
else ifeq ($(TRAIN_MODE), FA)
    CXXFLAGS += -DTRAIN_FA
endif

TRAIN_PHASIC_ENABLED ?= 0
ifeq ($(TRAIN_PHASIC_ENABLED),1)
    CXXFLAGS += -DTRAIN_PHASE
endif

TRAIN_ELIGIBLETRACE_ENABLED ?= 0
ifeq ($(TRAIN_ELIGIBLETRACE_ENABLED),1)
    CXXFLAGS += -DTRAIN_ELIGIBLETRACE
endif

# Include directories
INCLUDES := -I../include

# Source, object, and dependency files
SRCS := SMsim.cpp Core.cpp Event_unit.cpp Spike.cpp
OBJS := $(SRCS:.cpp=.o)
DEPS := $(SRCS:.cpp=.d)

# Target executable
TARGET := SMsim

# Default installation directory
INSTALL_DIR ?= /usr/local/bin

# Phony targets
.PHONY: all clean debug install uninstall run

# Default target
all: check_includes $(TARGET)

# Debug target
debug: CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -O0 -g -DDEBUG -DREFRACTORY -fopenmp -D__GIT_REV__=\"$(GIT_REV)\"
debug: clean check_includes $(TARGET)

# Check includes directory
check_includes:
	@echo "Checking include directory contents:"
	@ls -R ../include/

# Link target
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Include dependency files
-include $(DEPS)

# Compile object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

# Clean target
clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)

# Install target
install: $(TARGET)
	install -d $(INSTALL_DIR)
	install $(TARGET) $(INSTALL_DIR)

# Uninstall target
uninstall:
	rm -f $(INSTALL_DIR)/$(TARGET)

# Run target
run: all
	@echo "Running $(TARGET)..."
	./$(TARGET)

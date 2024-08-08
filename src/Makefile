# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -O2 -g -DREFRACTORY -DPHASIC_TRAIN -fopenmp
LDFLAGS := -fopenmp

# Include directories
INCLUDES := -I../include

# Get short git revision
GIT_REV := $(shell git rev-parse --short HEAD)

# Source, object, and dependency files
SRCS := SMsim.cpp Core.cpp Event_unit.cpp Spike.cpp
OBJS := $(SRCS:.cpp=.o)
DEPS := $(SRCS:.cpp=.d)

# Target executable
TARGET := SMsim

# Phony targets
.PHONY: all clean debug install uninstall run

# Default target
all: check_includes $(TARGET)

# Debug target
debug: CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -O0 -g -DDEBUG -DREFRACTORY -DPHASIC_TRAIN -fopenmp
debug: clean check_includes $(TARGET)

# Check includes directory
check_includes:
	@echo "Checking include directory contents:"
	@ls -R ../include/

# Link target
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) -o $(TARGET) $(LDFLAGS) -D__GIT_REV__=\"$(GIT_REV)\"

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
	install -d /usr/local/bin
	install $(TARGET) /usr/local/bin

# Uninstall target
uninstall:
	rm -f /usr/local/bin/$(TARGET)

# Run target
run: all
	@echo "Running $(TARGET)..."
	./$(TARGET)

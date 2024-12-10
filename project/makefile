CXX = g++
CXXFLAGS = -std=c++17 -Wall -I.
LDFLAGS = 

# Target executable
TARGET = sorting_visualizer

# Source files
SRCS = main.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Compilation rule
$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

# Generic compilation rule
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: clean

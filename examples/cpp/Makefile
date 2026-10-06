# examples/cpp/Makefile
REPO_ROOT := $(abspath ../..)
INC       := $(REPO_ROOT)/bindings/julia/include
LIB       := $(REPO_ROOT)/target/release

example: example.cpp
	$(CXX) $< -I$(INC) -L$(LIB) -lgrav -Wl,-rpath,$(LIB) -o $@

.PHONY: clean
clean:
	rm -f example

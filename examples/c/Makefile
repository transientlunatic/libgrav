# examples/c/Makefile
REPO_ROOT := $(abspath ../..)
INC       := $(REPO_ROOT)/bindings/julia/include
LIB       := $(REPO_ROOT)/target/release

example: example.c
	$(CC) $< -I$(INC) -L$(LIB) -lgrav -Wl,-rpath,$(LIB) -o $@ -lm

.PHONY: clean
clean:
	rm -f example

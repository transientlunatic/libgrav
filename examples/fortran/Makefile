# examples/fortran/Makefile
REPO_ROOT    := $(abspath ../..)
LIB          := $(REPO_ROOT)/target/release
GRAV_MOD   := $(REPO_ROOT)/bindings/julia/include/grav_binary.f90

example: example.f90 $(GRAV_MOD)
	$(FC) $(GRAV_MOD) $< -L$(LIB) -lgrav -Wl,-rpath,$(LIB) -o $@

.PHONY: clean
clean:
	rm -f example *.mod

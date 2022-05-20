# Project Tools
CC := iverilog

# Specify compiler flags
CFLAGS := -Wall

# Include directories
INCLUDE := -I./rtl-src/systolic-elements
INCLUDE += -I./rtl-src/arithmetic-logic
INCLUDE += -I./rtl-src/pooling-activations
INCLUDE += -I./rtl-src/memories
INCLUDE += -I./rtl-src/uart
INCLUDE += -I./rtl-src/IO

# Specify binary, object directories
BUILDDIR := ./build
SIMDIR   := ./$(BUILDDIR)/rtl-sim
VCDDIR   := $(SIMDIR)/vcd
OUTDIR   := $(SIMDIR)/out

# Specify source directories
SRCDIR := ./rtl-src/systolic-elements
SRCDIR += ./rtl-src/arithmetic-logic
SRCDIR += ./rtl-src/pooling-activations
SRCDIR += ./rtl-src/memories
SRCDIR += ./rtl-src/uart
SRCDIR += ./rtl-src/IO

# Specify test directories
TSTDIR := ./rtl-tests/systolic-elements
TSTDIR += ./rtl-tests/arithmetic-logic
TSTDIR += ./rtl-tests/pooling-activations
TSTDIR += ./rtl-tests/memories
TSTDIR += ./rtl-tests/uart
TSTDIR += ./rtl-tests/IO

# Add testbench paths to use in implicit building
vpath %.v $(TSTDIR)

# Define variables for verilog srcs, verilog testbenches, etc.
VSOURCES   := $(shell find $(SRCDIR) -type f -name *.v)
VTBSOURCES := $(shell find $(TSTDIR) -type f -name *.v -exec basename {} \;)
OUTPUTS    := $(patsubst %.out, $(OUTDIR)/%.out, $(VTBSOURCES:.v=.out))

PYTHON     ?= python

# Default build rules
all: $(OUTPUTS)

# Compile sources
$(OUTDIR)/%.out: %.v $(VSOURCES)
	@mkdir -p $(SIMDIR)
	@mkdir -p $(VCDDIR)
	@mkdir -p $(OUTDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ $<
	@./$@

# Clean rule
PHONY: clean
clean:
	rm -rf $(BUILDDIR)/rtl-sim 2> /dev/null || true

PHONY: mnist
mnist:
	$(PYTHON) ./scripts/train_mnist.py
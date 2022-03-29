# SEAsynth 🌊

A Systolic-Engine based Accelerator (SEA) that can be synthesised to FPGA fabric.

⚠ [Work In Progress]: Repo still subject to change and is in an unfinished state ⚠

## Key hardware components of Accelerator

- Systolic Convolution Engine (SCE)
	- Comprised of stacked systolic arrays
    - Adder-trees at each SCE output
- ReLU Activation-Function Unit (RAFU)
- Max-Pooling Unit (MPU)
- An activation memory

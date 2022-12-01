# Exercises

## HIP exercises 

**Log onto ACP/ACC**

Usually you would use `salloc -N 1 -p MI250 --gpus=8 –exclusive` to get exclusive use of a node. But that can be wasteful when resources are in high demand.

For these exercises, we’ll use either batch commands or short interactive sessions. For batch sessions, create a script that starts with
```bash
#!/bin/bash
#SBATCH -p MI250
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --reservation=enccs //depending on the workshop day enccs_2, enccs_3, enccs_4
<serial commands>
<srun for parallel commands>
...
```

For an interactive session, we’ll use “salloc -N 1 -p MI250 --gpus=1 -t 10” or “salloc -N 1 -p MI210 --gpus=1 -t 10” for these exercises so that the nodes can be shared. Check what is available with “sinfo” and look for a partition with nodes in the “idle” state. 
Load environment with module command. ROCm is needed for all and cmake is needed for openmp-helloworld
module load rocm/5.3.0 cmake

If you are only getting some of the GPUs on a node, the GPU detection will fail in some cases in the rocm_agent_enumerator utility. The problem is caused by Slurm removing permissions for the GPUs that you don’t have permission to use. That causes the rocm_agent_enumerator utility to crash when it queries the GPU to get the type. There are a couple of workarounds. You can set HCC_AMDGPU_TARGET to bypass GPU detection. The MI250 GPU is gfx90a. You can get your GPU type with rocminfo.

```shell
export HCC_AMDGPU_TARGET=gfx90a
```

You can also use the rocminfo command to autodetect the GPU type:

```shell
export HCC_AMDGPU_TARGET=`rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *\(gfx[0-9,a-f]*\) *$/\1/'`
```

For compilation with hipcc, use the clang compiler option --offload-arch.

ROCM_GPU= `rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *\(gfx[0-9,a-f]*\) *$/\1/'`
hipcc --offload-arch=${ROCM_GPU} ...

Get HIP-Examples

```shell
git clone https://github.com/ROCm-Developer-Tools/HIP-Examples
cd HIP-Examples/vectorAdd
```

Examine files here – README, Makefile and vectoradd_hip.cpp
Notice that Makefile requires HIP_PATH to be set. Check with `module show rocm/5.3.0` or `echo $HIP_PATH`
Also, the Makefile builds and runs the code. We’ll do the steps separately

```shell
make vectoradd_hip.exe
make test
```

Now let’s try the cuda-stream example. This example is from the original McCalpin code as ported to CUDA by Nvidia. This version has been ported to use HIP. See add4 for another similar stream example.

```shell
cd cuda-stream
make
./stream
```

Note that it builds with the hipcc compiler. You should get a report of the Copy, Scale, Add, and Triad cases.

The batch version of this would be:

```bash
#!/bin/bash
#SBATCH -p MI250
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH –t 10
#SBATCH --reservation=enccs //depending on the workshop day enccs_2, enccs_3, enccs_4

module load rocm/5.3.0

# If only getting some of the GPUs on a node, the GPU detection will fail
#   in some cases in rocm_agent_enumerator utility. Set HCC_AMDGPU_TARGET to
#   bypass GPU detection
# Setting explicit GPU target
#export HCC_AMDGPU_TARGET=gfx90a
# Using rocminfo to determine which GPU to build code for
export HCC_AMDGPU_TARGET=`rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *\(gfx[0-9,a-f]*\) *$/\1/'`

cd HIP-Examples/vectorAdd
make vectoradd_hip.exe
make test
cd ../..

cd HIP-Examples/cuda-stream
make
./stream
cd ../..
```

Save these commands in a batch file, hip_batch.sh, and then submit it to the queue with `sbatch < hip_batch.sh`. Check for status of job with `squeue -u <username>`. The output will come out in a file named `slurm-<job-id>.out`. Note that with some versions of ROCm, the GPU type detection using rocm_agent_enumerator will fail if all the GPUs are not allocated to the job.

You can try all the examples with `./test_all.sh`. Or pick one of the examples from the test_all.sh script and follow the steps given there.

## Hipify example

We’ll use the same HIP-Examples that were downloaded for the first exercise
Get a node allocation. Check what is available with sinfo. Then `salloc -N 1 -p MI250 --gpus=1 –t 10` or `salloc -N 1 -p MI210 --gpus=1 –t 10`. A batch version of the example is also shown.

**Hipify Programming (20 mins)**

*Exercise 1: Manual code conversion from CUDA to HIP (10 min)*

Choose one or more of the CUDA samples in `HIP-Examples/mini-nbody/cuda` repository and manually convert them to HIP. Some code suggestions include `mini-nbody/cuda/<nbody-block.cu,nbody-orig.cu,nbody-soa.cu>`

1.	The CUDA samples are located in HIP-Examples/mini-nbody/cuda
2.	Manually convert the source code of your choice to HIP 
3.	You’ll want to compile on the node you’ve been allocated so that hipcc will choose the correct GPU architecture.

*Exercise 2: Code conversion from CUDA to HIP using HIPify tools (10 min)*

Use the `hipify-perl.sh -inplace -print-stats` to "hipify" the CUDA samples you used to manually convert to HIP in Exercise 1. `hipify-perl.sh` is in `$ROCM_PATH/hip/bin` directory and should be in your path.

a.	For example, if helloworld.cu is a CUDA program, run `hipify-perl.sh -inplace –print-stats helloworld.cu`. You’ll see a `helloworld.cu.prehip` file that is the original and the `helloworld.cu` file now has HIP calls.
b.	You’ll also see statistics of HIP APIs that were converted. For example, for `hipify-perl.sh -inplace -print-stats nbody-orig.cu`:

```shell
[HIPIFY] info: file 'nbody-orig.cu' statistics:
  CONVERTED refs count: 7
  TOTAL lines of code: 91
  WARNINGS: 0
[HIPIFY] info: CONVERTED refs by names:
  cudaFree => hipFree: 1
  cudaMalloc => hipMalloc: 1
  cudaMemcpyDeviceToHost => hipMemcpyDeviceToHost: 1
  cudaMemcpyHostToDevice => hipMemcpyHostToDevice: 1
```

c.	Compile the HIP programs. Fix any compiler issues, for example, if there was something that didn’t hipify correctly. Be on the lookout for hard-coded Nvidia specific things like warp sizes and PTX.
For the nbody-orig.cu code, compile with `hipcc -DSHMOO -I ../ nbody-orig.cu -o nbody-orig`.  The `#define SHMOO` fixes some timer printouts. Add `--offload-arch=<gpu_type>` to specify the GPU type and avoid the autodetection issues when running on a single GPU on a node.
d.	Run the programs.
A batch version of Exercise 2 is:

```bash
#!/bin/bash
#SBATCH -p MI250
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 10
#SBATCH --reservation=enccs //depending on the workshop day enccs_2, enccs_3, enccs_4

module load rocm/5.3.0

# Setting explicit GPU target
#export ROCM_GPU=gfx90a
# Using rocminfo to determine which GPU to build code for
export ROCM_GPU=`rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *\(gfx[0-9,a-f]*\) *$/\1/'`

hipify-perl.sh -inplace -print-stats nbody-orig.cu
cd HIP-Examples/mini-nbody/cuda
hipcc --offload-arch=${ROCM_GPU} -DSHMOO -I ../ nbody-orig.cu -o nbody-orig
./nbody-orig
cd ../..
```

**Notes:**

- Hipify tools do not check correctness 
- Hipify-perl can’t handle library calls, hipify-clang can handle library calls
- hipconv


## OpenMP Programming

*Exercise: Getting Started with OpenMP on AMD Accelerators*

The goal of this exercise is to offload simple OpenMP codelets onto AMD GPU. By default, GNU compilers are used to build these mini-apps that can then be executed on host (CPU). So:

a.	The codelet source codes are located in “exercises/openmp_samples”. Copy the codelet repository to your local directory and let’s consider the saxpy (C) and Fibonacci (Fortran) examples.

b.	You will instruct the compiler to offload certain code sections (loops) within these min-apps

a.	For the C/C++ codelet (saxpy example), in “codelet.c” file:
-	replace `#pragma omp parallel for simd` by `#pragma omp target teams distribute parallel for simd map(to: x[0:n],y[0:n]) map(from: z[0:n])`.

b.	For the Fortran codelet (Fibonacci example) in “freduce.f90” file:

-	Add the following instruction just before the beginning of the innermost loop: `!$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD REDUCTION(+sum2) MAP(TO:array(1:10))`
-	Add the following instruction right after the end of the innermost loop code section: `!$OMP END TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD`

c.	In “Makefile”, replace “gcc” (gfortran) by “amdclang” (amdflang) and add `--offload-arch=gfx90a` to compiler flags to enable offloading on AMD GPU MI200.
d.	Build and then run these codelets on an ACP node using an input size of your choice like 123456789. 
e.	While running one of these codelets, open another terminal and “ssh” to the ACP node you are working on. Then, run “watch -n 0.1 rocm-smi” command line from that terminal to visualize GPU activities.
f.	Next, run the codelet on your preferred GPU device. For example, to execute on GPU ID #2, set the following environment variable: “export ROCR_VISIBLE_DEVICES=2” then run the code
g.	While running this code on your preferred GPU device, open another terminal then run `watch -n 0.1 rocm-smi` command line to visualize GPU activities

h.	Profile the codelet and then compare output by setting:
- a.	`export LIBOMPTARGET_KERNEL_TRACE=1`
- b.	`export LIBOMPTARGET_KERNEL_TRACE=2`

**Note:**

- rocminfo can be used to get target architecture information.
- If for any reason `--offload-arch=gfx90a` is not working as expected, consider using alternative flags: `-fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a` to enable offloading on AMD GPU MI200. 

## HIPFort Example

Get a node allocation. Check what is available with sinfo. Then `salloc -N 1 -p MI250 --gpus=1` or `salloc -N 1 -p MI210 --gpus=1`

`module load rocm`

Check if hipfort is installed -- /opt/rocm<-version>/bin/hipfort

Install HIPFort:

- `export HIPFORT_INSTALL_DIR=`pwd`/hipfort`
- `git clone https://github.com/ROCmSoftwarePlatform/hipfort hipfort-source`
- `mkdir hipfort-build; cd hipfort-build`
- `cmake -DHIPFORT_INSTALL_DIR=${HIPFORT_INSTALL_DIR} ../hipfort-source`
- `make install`
 

Try example from source directory:

- `export PATH=${HIPFORT_INSTALL_DIR}/bin:$PATH`
- ``ROCM_GPU=`rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//' -e 's/[[:space:]]*$//'```
- `cd hipfort-source/test/f2003/vecadd`
- `hipfc -v --offload-arch=${ROCM_GPU} hip_implementation.cpp main.f03`
- `./a.out`

Examine the code in hip_implementation.cpp The kernel code is in C and has a wrapper so it can be called from Fortran. Now look at the code in main.f03. The Fortran code declares an interface to the GPU kernel routine and invokes it with a call statement. Note that the hip calls can be made from the Fortran code courtesy of the wrappers provided by hipfort.

Example with Fortran 2008 interface – on your own:
- `cd hipfort-source/test/f2003/vecadd`
- `hipfc -v --offload-arch=${ROCM_GPU} hip_implementation.cpp main.f08`
- `./a.out`

**Fortran with OpenMP offloading or OpenACC**

Get the training examples – AMDTrainingExamples_ver0.2.tgz. Pick one of the examples in PragmaExamples for Fortran in OpenMP or OpenACC. We’ll use the new Siemen’s compiler.

```shell
tar -xzvf AMDTrainingExamples_ver0.2.tgz
cd AMDTrainingExamples/PragmaExamples
module load siemens-gcc
export FC=/global/software/siemens-gcc/bin/x86_64-none-linux-gnu-gfortran
cd OpenACC/Fortran/Make/vecadd
```

Note in the comiler output:

```shell
vecadd.F:50:62: optimized: assigned OpenACC gang vector loop parallelism
```

Run the executable: `./vecadd`

Output:
```shell
Final result:   1.000000
Runtime is: 0.102539 secs
```

Try setting:
```shell
export GCN_DEBUG=1
```

And rerun. You should get a lot of output which confirms that the code is running on the GPU.


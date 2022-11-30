# Introduction to HIP Programming

## Slides

{download}`Slides <slides/intro_hip_programming.pdf>`


## Exercises

Usually you would use `salloc -N 1 -p MI250 --gpus=8 –exclusive` to
get exclusive use of a node. But that can be wasteful when resources
are in high demand.

For these exercises, we’ll use either batch commands or short
interactive sessions. For batch sessions, create a script that starts
with

```
#!/bin/bash
#SBATCH -p MI250
##SBATCH -p MI250-x4-IB
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --reservation=enccs //depending on the workshop day enccs_2, enccs_3, enccs_4
<serial commands>
<srun for parallel commands>
```

For an interactive session, we’ll use `salloc -N 1 -p MI250 --gpus=1
--reservation=enccs -t 10` or `salloc -N 1 -p MI250-x4-IB --gpus=1
--reservation=enccs -t 10` for these excercises so that the nodes can
be shared. Check what is available with `sinfo` and look for a
partition with nodes in the `idle` state.  Load environment with
module command. ROCm is needed for all and `cmake` is needed for
openmp-helloworld

`module load rocm/5.3.0 cmake`

If you are only getting some of the GPUs on a node, the GPU detection will fail in some cases in the rocm_agent_enumerator utility. The problem is caused by Slurm removing permissions for the GPUs that you don’t have permission to use. That causes the rocm_agent_enumerator utility to crash when it queries the GPU to get the type. There are a couple of workarounds. You can set HCC_AMDGPU_TARGET to bypass GPU detection. The MI250 GPU is gfx90a. You can get your GPU type with `rocminfo`.

Get HIP-Examples
```
git clone https://github.com/ROCm-Developer-Tools/HIP-Examples
cd HIP-Examples/vectorAdd
```

Examine files here – README, Makefile and vectoradd_hip.cpp

Notice that Makefile requires HIP_PATH to be set. Check with `module show rocm/5.3.0` or `echo $HIP_PATH`
Also, the Makefile builds and runs the code. We’ll do the steps separately

```
make vectoradd_hip.exe
make test
```

In order to provide the AMD GPU achitecture for MI200, `gfx90a`, you can add in the makefile a variable like `HIPFLAGS=--offload-arch=gfx90a` and add in the compilation  `$(HIPCC) $(HIPFLAGS) $(OBJECTS) -o $@`


Now let’s try the cuda-stream example. This example is from the original McCalpin code as ported to CUDA by Nvidia. This version has been ported to use HIP. See add4 for another similar stream example. Add also the HIPFLAGS variable.

```
cd cuda-stream
make
./stream
```
Note that it builds with the hipcc compiler. You should get a report of the Copy, Scale, Add, and Triad cases.


The batch version of this would be
```
#!/bin/bash
#SBATCH -p MI250
##SBATCH -p MI250-x4-IB
#SBATCH -N 1
#SBATCH --reservation=enccs
#SBATCH --gpus=1

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

Save these commands in a batch file, `hip_batch.sh`, and then submit it to the queue with `sbatch hip_batch.sh`. Check for status of job with `squeue -u <username>`. The output will come out in a file named `slurm-<job-id>.out`. Note that with some versions of ROCm, the GPU type detection using rocm_agent_enumerator will fail if all the GPUs are not allocated to the job.

You can try all the examples with `./test_all.sh`. Or pick one of the examples from the `test_all.sh` script and follow the steps given there.


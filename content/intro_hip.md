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



### November 30

#### Point-to-point and collective 

Get the node with at least two GPUs and tasks:

`salloc -N 1 -p MI250 --gpus=2 --ntasks 2`

Load OpenMPI module

`module load openmpi/4.1.4-gcc`

Change mpicxx wrapper compiler to use hipcc
`export OMPI_CXX=hipcc`
Compile and run the code
`mpicxx -o ./pt2pt ./pt2pt.cpp $mpirun -n 2 ./pt2pt`
You can get around the message "WARNING: There was an error initializing an OpenFabrics device" by telling OpenMPI to exclude openib:
`mpirun -n 2 --mca btl ^'openib' ./pt2pt`

#### OSU Bandwidth benchmark code
Get a node allocation. Check what is available with sinfo. Then
`salloc -N 1 --ntasks 8 –gpus=8 -p MI250`

##### Set up environment
`module load rocm openmpi/4.1.4-gcc`

##### Check if OpenMPI is built with UCX

`ompi_info`

##### Check if UCX is built with ROCm?
`/global/software/openmpi/gcc/ucx/bin/ucx_info  -v`
##### Get Ohio State University micro-benchmark code
`wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.0.tar.gz`
##### Build micro-benchmark code
```
./configure --prefix=${HOME}/osu-mb \
        CC=/global/software/openmpi/gcc/ompi/bin/mpicc \
        CXX=/global/software/openmpi/gcc/ompi/bin/mpicxx \
        --enable-rocm --with-rocm=${ROCM_PATH}
make –j12
make install
```

##### Run benchmark
```
export HIP_VISIBLE_DEVICES=0,1
mpirun –N 1 –n 2 ./osu-mb/mpi/pt2pt/osu_bw $((16*1024*1024)):$((16*1024*1024)) D D
```

Notes:
* Try different pairs of GPUs. How does the bandwidth vary?
* Try different communication options (blit kernel and SDMA) using the env variable HSA_ENABLE_SDMA. How does the bandwidth vary?


#### AMD Node Memory Model

##### Managed Memory

We’ll use the examples in the AMDTrainingExamples from before
```
tar -xzvf AMDTrainingExamples_ver0.2.tgz
cd AMDTrainingExamples/ManagedMemory
```
The example from HIP-Examples/vectorAdd has been duplicated here as vectoradd_hip.cpp. A slightly cleaned up version is in vectoradd_hip1.cpp. Check that you have the original version running first.
```
module load rocm
make vectoradd_hip1.exe
./vectoradd_hip1.exe
```
The original tests should run. If you encounter difficulties, you can check what is happening by setting the following environment variables.
```
export LIBOMPTARGET_KERNEL_TRACE=1
export LIBOMPTARGET_INFO=$((0x20 | 0x02 | 0x01 | 0x10))
```
Take that code and turn it into a managed memory version with the following steps:

1.Globally change all “host” strings to “vector”
2.Globally change all “device” strings to “vector”
3.Remove duplicate float declarations
4.Move both allocations above initialization loop
5.Comment out all hip data copies from host to device and device to host
6.Add hipDeviceSynchronize(); after the kerel launch

* First experiment:
    * comment out the hipMalloc/hipFrees
    * Test should fail with a memory access fault. Now set HSA_XNACK
     `export HSA_XNACK=1`
 
 Rerun and test should pass
 
 * Second experiment: 
 
     * comment out the malloc/frees instead and unset the HSA_XNACK variable or set it to 0
     * Test should pass
 
 ##### OpenMP Atomics
 
 The examples for this exercise are in `AMDTrainingExamples/atomics_openmp`.
 Set up your environment
 ```
 module load aomp rocm
 export CC=${AOMP}/bin/clang
 ```
 Now let’s look at the first example. The key lines are:
 ```
 #pragma omp target teams distribute parallel for map(tofrom: a[:n]) map(to: b[:n])
 for(int i = 0; i < n; i++) {
   a[i] += b[i];
   }
   ```
   Build the first example and run it. The map clause will copy the data from the host to device and back. The memory allocated will be coarse grain.
   ```
   Make arraysum1
   ./arraysum1
   ```
   
   It should run and pass. Now make the following changes 
   
   1.Remove map clause
   2.Add #pragma omp requires unified_shared_memory before main
   
   There is no memory movement so the system must move the memory for us. Let’s try it out.
   ```
   make arraysum2
   ./arraysum2
   ```
   
   It should fail because the memory has not moved. Setting HSA_XNACK should fix this problem.
   ```
   export HSA_XNACK=1
   ./arraysum2
   ```
   It should run now. The memory that is created is now fine-grained. We can change this back to coarse-grained by adding back the map clause. This is done in `arraysum3.c`.
   Now let’s move the initialization to a separate routine and use the map clause so that it is created as coarse-grained memory.
   ```
   void init(int n, double *a, double *b) {
     #pragma omp target map(from:a[:n],b[:n])
 for(int i = 0; i < n; i++){
     a[i] = b[i] = 1.0;
   }
   }
   ```
   The main loop is
   ```
   #pragma omp target teams distribute parallel for
   for(int i = 0; i < n; i++) {
     a[i] += b[i];
 }
 ```
 The memory in the main loop remains coarse-grained even though a map clause is not used. Once the memory is allocated, it stays that type. Note that the initialization loop is not run in parallel on the GPU. How would you fix that?
 
 We’ll skip ahead to the `arraysum8.c` with an atomic reduction. The key loop is
 ```
 #pragma omp target teams distribute parallel for reduction(+:ret)
 for(int i = 0; i < n; i++) {
   #pragma omp atomic hint(AMD_fast_fp_atomics)
     ret += b[i];
 }
 ```
 This test should fail because the memory is fine-grained. Add the map clause to the pragma as implemented in arraysum9.c. Note that the ret variable also needs to be mapped.
 `map(to: b[:n]) map(tofrom: ret)`
 
 Now compile and run. It should pass.
 ```
 make arraysum9
 ./arraysum9
 ```
 
 Another solution to fix the problem is to change the atomic pragma to a safe version. This is shown in arraysum10.c
 #pragma omp atomic hint(AMD_safe_fp_atomics)
 
The safe atomic will use a compare and swap (CAS) loop instead of an atomic add. This will work, but it will likely be slower.
 
The examples in arraysum4.c to arraysum7.c show the same atomic behavior with add the elements of array b to array a and storing it back to array a

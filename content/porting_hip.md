# Porting Applications to HIP

## Slides

{download}`Slides <slides/Porting_Applications_to_HIP.pdf>`

## Exercises

### Hipify example

We’ll use the same HIP-Examples that were downloaded for the first exercise
Get a node allocation. Check what is available with sinfo. Then `salloc -N 1 -p MI250 --reservation=enccs --gpus=1 –t 10` or `salloc -N 1 -p MI250-x4-IB --gpus=1 –t 10`. A batch version of the example is also shown.

#### Hipify Programming (20 mins)

* Exercise 1: Manual code conversion from CUDA to HIP (10 min)
Choose one or more of the CUDA samples in `HIP-Examples/mini-nbody/cuda` repository and manually convert them to HIP. Some code suggestions include `mini-nbody/cuda/<nbody-block.cu`,`nbody-orig.cu`,`nbody-soa.cu`>
1.The CUDA samples are located in `HIP-Examples/mini-nbody/cuda`
2.Manually convert the source code of your choice to HIP 
3.You’ll want to compile on the node you’ve been allocated so that hipcc will choose the correct GPU architecture.

* Exercise 2: Code conversion from CUDA to HIP using HIPify tools (10 min)

Use the `hipify-perl.sh -inplace -print-stats` to `hipify` the CUDA samples you used to manually convert to HIP in Exercise 1. `hipify-perl.sh` is in `$ROCM_PATH/hip/bin` directory and should be in your path.
a.For example, if helloworld.cu is a CUDA program, run `hipify-perl.sh -inplace –print-stats helloworld.cu`. You’ll see a `helloworld.cu.prehip` file that is the original and the helloworld.cu file now has HIP calls.
b.You’ll also see statistics of HIP APIs that were converted. For example, for `hipify-perl.sh -inplace -print-stats nbody-orig.cu`:

[HIPIFY] info: file 'nbody-orig.cu' statistics:
  CONVERTED refs count: 7
    TOTAL lines of code: 91
  WARNINGS: 0
  [HIPIFY] info: CONVERTED refs by names:
    cudaFree => hipFree: 1
  cudaMalloc => hipMalloc: 1
    cudaMemcpyDeviceToHost => hipMemcpyDeviceToHost: 1
  cudaMemcpyHostToDevice => hipMemcpyHostToDevice: 1
  
  c.Compile the HIP programs. Fix any compiler issues, for example, if there was something that didn’t hipify correctly. Be on the lookout for hard-coded Nvidia specific things like warp sizes and PTX.
  For the nbody-orig.cu code, compile with `hipcc -DSHMOO -I ../ nbody-orig.cu -o nbody-orig`.  The #define SHMOO fixes some timer printouts. Add `--offload-arch=<gpu_type>` to specify the GPU type and avoid the autodetection issues when running on a single GPU on a node.
  d.Run the programs.
  
  A batch version of Exercise 2 is:
  ```
  #!/bin/bash
  #SBATCH -p MI250
  #SBATCH -p MI250-x4-IB
  #SBATCH -N 1
  #SBATCH --reservation=enccs
  #SBATCH --gpus=1
  #SBATCH -t 10
  
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
  
  Notes:
  •Hipify tools do not check correctness 
  •Hipify-perl can’t handle library calls, hipify-clang can handle library calls
  •hipconvertinplace-perl.sh is a convenience script that does the “hipify-perl.sh -inplace -print-stats" command
  
  
  
  

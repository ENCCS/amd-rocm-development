#  Getting Started with OpenMP® Offload Applications on AMD Accelerators

## Slides

{download}`Slides <slides/ENCCS_Workshop_start_openmp.pdf>`

## Exercises

  ### OpenMP Programming
  
  Exercise: Getting Started with OpenMP on AMD Accelerators 
  
  The goal of this exercise is to offload simple OpenMP codelets onto AMD GPU. By default, GNU compilers are used to build these mini-apps that can then be executed on host (CPU). So:
  a.The codelet source codes are located in `exercises/openmp_samples` in the /global/training/enccs/exercises/openmp_samples directory.
  Copy the codelet repository to your local directory and let’s consider the saxpy (C) and Fibonacci (Fortran) examples.
  b.You will instruct the compiler to offload certain code sections (loops) within these mini-apps
  a.For the C/C++ codelet (saxpy example), in `codelet.c` file:
  •replace `#pragma omp parallel for simd` by `#pragma omp target teams distribute parallel for simd map(to: x[0:n],y[0:n]) map(from: z[0:n])`.
  b.For the Fortran codelet (Fibonacci example) in `freduce.f90` file:
  •Add the following instruction just before the beginning of the innermost loop: `!$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD REDUCTION(+sum2) MAP(TO:array(1:10))`
  •Add the following instruction right after the end of the innermost loop code section: `!$OMP END TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD`
  c.In `Makefile`, replace `gcc`(gfortran) by `amdclang`(amdflang) and add `--offload-arch=gfx90a` to compiler flags to enable offloading on AMD GPU MI200.
  d.Build and then run these codelets on an ACP node using an input size of your choice like 123456789. 
  e.While running one of these codelets, open another terminal and `ssh` to the ACP node you are working on. Then, run `watch -n 0.1 rocm-smi` command line from that terminal to visualize GPU activities.
  f.Next, run the codelet on your preferred GPU device. For example, to execute on GPU ID #2, set the following environment variable: “export ROCR_VISIBLE_DEVICES=2” then run the code
  g.While running this code on your preferred GPU device, open another terminal then run `watch -n 0.1 rocm-smi` command line to visualize GPU activities
  h.Profile the codelet and then compare output by setting:
  a.`export LIBOMPTARGET_KERNEL_TRACE=1`
  b.`export LIBOMPTARGET_KERNEL_TRACE=2`
  
Note: 

- rocminfo can be used to get target architecture information.
- If for any reason `--offload-arch=gfx90a` is not working as expected, consider using alternative flags: `-fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a` to enable offloading on AMD GPU MI200. 

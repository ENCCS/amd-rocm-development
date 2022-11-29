# Developing Fortran Applications: HIPFort, OpenMP®, and OpenACC

## Slides

{download}`Slides <slides/Developing_Fortran_Applications.pdf>`

## Exercises

### HIPFort Example

Get a node allocation. Check what is available with sinfo. Then `salloc -N 1 -p MI250 --reservation=enccs --gpus=1 –t 10` or `salloc -N 1 -p MI250-x4-IB --gpus=1 –t 10`
  Module load rocm
  Check if hipfort is installed -- /opt/rocm<-version>/bin/hipfort
  *Install HIPFort
* `export HIPFORT_INSTALL_DIR=`pwd`/hipfort`
* `git clone https://github.com/ROCmSoftwarePlatform/hipfort hipfort-source`
* `mkdir hipfort-build; cd hipfort-build`
* `cmake -DHIPFORT_INSTALL_DIR=${HIPFORT_INSTALL_DIR} ../hipfort-source`
* `make install`

* Try example from source directory
* `export PATH=${HIPFORT_INSTALL_DIR}/bin:$PATH`
* ROCM_GPU=`rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//' -e 's/[[:space:]]*$//'
* `cd hipfort-source/test/f2003/vecadd`
* `hipfc -v --offload-arch=${ROCM_GPU} hip_implementation.cpp main.f03`
* `./a.out`

Examine the code in `hip_implementation.cpp` The kernel code is in C and has a wrapper so it can be called from Fortran. Now look at the code in `main.f03`. The Fortran code declares an interface to the GPU kernel routine and invokes it with a call statement. Note that the hip calls can be made from the Fortran code courtesy of the wrappers provided by hipfort.

*Example with Fortran 2008 interface – on your own
*`cd hipfort-source/test/f2003/vecadd`
*`hipfc -v --offload-arch=${ROCM_GPU} hip_implementation.cpp main.f08`
*`./a.out`

### Fortran with OpenMP offloading or OpenACC

Get the training examples – `AMDTrainingExamples_ver0.2.tgz`, `cp /global/training/enccs/AMDTrainingExamples_ver0.2.tgz .`. Pick one of the examples in `PragmaExamples` for Fortran in OpenMP or OpenACC. 

We’ll use the new Siemen’s compiler.
* `tar -xzvf AMDTrainingExamples_ver0.2.tgz`
* `cd AMDTrainingExamples/PragmaExamples`
* `module load siemens-gcc`
* `export FC=/global/software/siemens-gcc/bin/x86_64-none-linux-gnu-gfortran`
* `cd OpenACC/Fortran/Make/vecadd`

Note in the comiler output:
`vecadd.F:50:62: optimized: assigned OpenACC gang vector loop parallelism`

Run the executable
`./vecadd`

```
Output
Final result:   1.000000
Runtime is: 0.102539 secs
```
Try setting 
`export GCN_DEBUG=1`

And rerun. You should get a lot of output which confirms that the code is running on the GPU.


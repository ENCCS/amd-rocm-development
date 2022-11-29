#  GPU-Aware MPI with ROCmTM

{download}`Slides <slides/GPU-AwareMPI+wcb.pdf>`

## Exercises

`

### Check if UCX is built with ROCm?

`/global/software/openmpi/gcc/ucx/bin/ucx_info  -v`

### Get Ohio State University micro-benchmark code

`wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.0.tar.gz`

### Build micro-benchmark code

```
./configure --prefix=${HOME}/osu-mb \
        CC=/global/software/openmpi/gcc/ompi/bin/mpicc \
        CXX=/global/software/openmpi/gcc/ompi/bin/mpicxx \
        --enable-rocm --with-rocm=${ROCM_PATH}
		make –j12
		make install
		```

### Run benchmark

```
export HIP_VISIBLE_DEVICES=0,1
mpirun –N 1 –n 2 ./osu-mb/mpi/pt2pt/osu_bw $((16*1024*1024)):$((16*1024*1024)) D D
```

Notes:
* Try different pairs of GPUs. How does the bandwidth vary?
* Try different communication options (blit kernel and SDMA) using the env variable HSA_ENABLE_SDMA. How does the bandwidth vary?

#  GPU-Aware MPI with ROCmTM

{download}`Slides <slides/GPU-AwareMPI+wcb.pdf>`

<iframe width="560" height="315" src="https://www.youtube.com/embed/R7Lj-oMF7-w?start=956" title="Developing Applications with the AMD ROCm Ecosystem - Day 1" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Exercises

### Point-to-point and collective 

Get the node with at least two GPUs and tasks:

`salloc -N 1 -p MI250 --gpus=2 --ntasks 2`

Load OpenMPI module

`module load openmpi/4.1.4-gcc`

Change mpicxx wrapper compiler to use hipcc
`export OMPI_CXX=hipcc`

Compile and run the code
```
mpicxx -o ./pt2pt ./pt2pt.cpp 
$mpirun -n 2 ./pt2pt
```

You can get around the message "WARNING: There was an error initializing an OpenFabrics device" by telling OpenMPI to exclude openib:
`mpirun -n 2 --mca btl ^'openib' ./pt2pt`

### OSU Bandwidth benchmark code

Get a node allocation. Check what is available with sinfo. Then
`salloc -N 1 --ntasks 8 –gpus=8 -p MI250`

#### Set up environment
`module load rocm openmpi/4.1.4-gcc`

#### Check if OpenMPI is built with UCX

`ompi_info`

#### Check if UCX is built with ROCm?
`/global/software/openmpi/gcc/ucx/bin/ucx_info  -v`

#### Get Ohio State University micro-benchmark code
`wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.0.tar.gz`

#### Build micro-benchmark code

```
./configure --prefix=${HOME}/osu-mb \
        CC=/global/software/openmpi/gcc/ompi/bin/mpicc \
        CXX=/global/software/openmpi/gcc/ompi/bin/mpicxx \
        --enable-rocm --with-rocm=${ROCM_PATH}
make –j12
make install
```

#### Run benchmark

```
export HIP_VISIBLE_DEVICES=0,1
mpirun –N 1 –n 2 ./osu-mb/mpi/pt2pt/osu_bw $((16*1024*1024)):$((16*1024*1024)) D D
```

Notes:
* Try different pairs of GPUs. How does the bandwidth vary?
* Try different communication options (blit kernel and SDMA) using the env variable HSA_ENABLE_SDMA. How does the bandwidth vary?



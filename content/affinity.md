# Affinity — Placement, Ordering and Binding

## Slides

{download}`Slides <slides/Affinity_Placement_Order_Binding.pdf>`


## Exercises

=${OMPI_COMM_WORLD_RANK} 
export local_rank=${OMPI_COMM_WORLD_LOCAL_RANK} 
export ranks_per_node=${OMPI_COMM_WORLD_LOCAL_SIZE} 

if [ -z "${NUM_CPUS}" ]; then 
    let NUM_CPUS=96 
fi 

if [ -z "${RANK_STRIDE}" ]; then 
    let RANK_STRIDE=${NUM_CPUS}/${ranks_per_node} 
fi 

if [ -z "${OMP_STRIDE}" ]; then 
    let OMP_STRIDE=1 
fi 

cpu_list=($(seq 0 ${NUM_CPUS})) 
let cpu_start_index=$(( ($RANK_STRIDE*${local_rank}) )) 
let cpu_start=${cpu_list[$cpu_start_index]} 
let cpu_stop=$(($cpu_start+$OMP_NUM_THREADS*$OMP_STRIDE-1)) 

export GOMP_CPU_AFFINITY=$cpu_start-$cpu_stop:$OMP_STRIDE 

"$@" 
```


Run the application using the helper script: 

`OMP_NUM_THREADS=2 mpirun -np 2 ./helper.sh ./hello_mpi_omp` 

* You’ll find that the threads of ranks 0 and 1 got bound to HWTs 0, 1 and 48, 49 respectively 

`OMP_NUM_THREADS=2 RANK_STRIDE=2 mpirun -np 2 ./helper.sh ./hello_mpi_omp` 

* Setting RANK_STRIDE to 2 will pack all threads to HWTs 0, 1, 2, and 3. 

```
OMP_NUM_THREADS=2 RANK_STRIDE=4 OMP_STRIDE=2 mpirun -np 2 ./helper.sh ./hello_mpi_omp 
```

* By setting RANK_STRIDE=4 and OMP_STRIDE=2, we can run the threads on HWTs 0, 2, 4 and 6 respectively. 

* Feel free to use this script on your own system by setting up NUM_CPUS accordingly. 

The above example and script can be found in ~/exercises/affinity/hello_mpi_omp directory 

##### Case 2: MPI + OpenMP + HIP 

* Allocate a node: salloc -N1 --exclusive -p MI250 -w mun-node-5
* Download hello_jobstep.cpp from here: https://code.ornl.gov/olcf/hello_jobstep 
* Load necessary modules in the environment: module load rocm openmpi/4.1.4-gcc 
* Use this simpler Makefile to compile: 

$ cat Makefile  

```
SOURCES = hello_jobstep.cpp 
OBJECTS = $(SOURCES:.cpp=.o) 
EXECUTABLE = hello_jobstep 

CXX=/global/software/openmpi/gcc/ompi/bin/mpic++ 
CXXFLAGS = -fopenmp -I${ROCM_PATH}/include -D__HIP_PLATFORM_AMD__ 
LDFLAGS = -L${ROCM_PATH}/lib -lhsa-runtime64 -lamdhip64 

all: ${EXECUTABLE} 

%.o: %.cpp 
$(CXX) $(CXXFLAGS) -o $@ -c $< 

$(EXECUTABLE): $(OBJECTS) 
$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)  

clean:   
rm -f $(EXECUTABLE)
rm -f $(OBJECTS) 
```

* Compile: make 
* Set up helper script for setting up ROCR_VISIBLE_DEVICES and GOMP_CPU_AFFINITY for indicating GPU and CPU core affinity respectively: 

```
#!/bin/bash 
export global_rank=${OMPI_COMM_WORLD_RANK} 
export local_rank=${OMPI_COMM_WORLD_LOCAL_RANK} 
export ranks_per_node=${OMPI_COMM_WORLD_LOCAL_SIZE} 

if [ -z "${NUM_CPUS}" ]; then 
    let NUM_CPUS=128 
fi 

if [ -z "${RANK_STRIDE}" ]; then 
    let RANK_STRIDE=${NUM_CPUS}/${ranks_per_node} 
fi 

if [ -z "${OMP_STRIDE}" ]; then 
    let OMP_STRIDE=1 
fi 

if [ -z "${NUM_GPUS}" ]; then 
    let NUM_GPUS=8 
fi 

if [ -z "${GPU_START}" ]; then 
    let GPU_START=0 
fi 

if [ -z "${GPU_STRIDE}" ]; then 
    let GPU_STRIDE=1 
fi 

cpu_list=($(seq 0 127)) 
let cpus_per_gpu=${NUM_CPUS}/${NUM_GPUS} 
let cpu_start_index=$(( ($RANK_STRIDE*${local_rank})+${GPU_START}*$cpus_per_gpu )) 
let cpu_start=${cpu_list[$cpu_start_index]} 
let cpu_stop=$(($cpu_start+$OMP_NUM_THREADS*$OMP_STRIDE-1)) 

gpu_list=(4 5 2 3 6 7 0 1) 
let ranks_per_gpu=$(((${ranks_per_node}+${NUM_GPUS}-1)/${NUM_GPUS})) 
let my_gpu_index=$(($local_rank*$GPU_STRIDE/$ranks_per_gpu))+${GPU_START} 
let my_gpu=${gpu_list[${my_gpu_index}]} 

export GOMP_CPU_AFFINITY=$cpu_start-$cpu_stop:$OMP_STRIDE 
export ROCR_VISIBLE_DEVICES=$my_gpu 

"$@" 
```

Run the application using the helper script: 

`OMP_NUM_THREADS=2 mpirun -np 8 ./helper.sh ./hello_jobstep` 

* Runs 2 threads per rank, 8 ranks, associating the GPUs in the order given for each rank and binding 2 CPU cores from each set of 16 cores for each rank 

`OMP_NUM_THREADS=2 mpirun -np 16 ./helper.sh ./hello_jobstep` 

* To run 2 ranks per GCD packed closely (ranks 0 and 1 run on GCD 4) and bind 2 cores from each set of 8 cores for each rank. 

The above example and scripts can be found in ~/exercises/affinity/hello_jobstep directory. 



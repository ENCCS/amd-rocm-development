# Profiling and debugging

## rocprof

{download}`Download slides <slides/intro_rocprof.pdf>`

## Omnitrace and Omniperf

{download}`Download slides <slides/intro_omnitools.pdf>`

## rocgdb

{download}`Download slides <slides/ENCCS-AMD-rocgdb-Dec2022.pdf>`

## Exercises

Reservation: enccs_3

### Rocprof

* Get the exercise:

`cp -r /global/training/enccs/exercises/HIP-Examples/mini-nbody/hip/ .`

* Compile and run the code

```
cd hip 
./HIP-nbody-orig.sh
hipcc -I../ -DSHMOO nbody-orig.cpp -o nbody-orig
65536, 161.871
```

* Check the file `HIP-nbody-orig.sh`

```
cat HIP-nbody-orig.sh

...

EXE=nbody-orig
...
./$EXE 65536
...

```
* The binary is called `nbody-orig`

* Use rocprof with `--stats`

```
rocprof --stats nbody-orig 65536

RPL: on '221130_200946' from '/global/software/rocm/rocm-5.3.0' in '/global/home/gmarko/HIP-Examples/mini-nbody/hip'
RPL: profiling '"nbody-orig" "65536"'
RPL: input file ''
RPL: output dir '/tmp/rpl_data_221130_200946_3670592'
RPL: result dir '/tmp/rpl_data_221130_200946_3670592/input_results_221130_200946'
ROCProfiler: input from "/tmp/rpl_data_221130_200946_3670592/input.xml"
  0 metrics
65536, 159.960

ROCPRofiler: 10 contexts collected, output directory /tmp/rpl_data_221130_200946_3670592/input_results_221130_200946
File '/global/home/gmarko/HIP-Examples/mini-nbody/hip/results.csv' is generating
File '/global/home/gmarko/HIP-Examples/mini-nbody/hip/results.stats.csv' is generating

```
Files with the prefix results are created

* Check the files results.csv

You can se einformation for each kernel call with their duration
```
 cat results.csv
 
"Index","KernelName","gpu-id","queue-id","queue-index","pid","tid","grd","wgr","lds","scr","arch_vgpr","accum_vgpr","sgpr","wave_size","sig","obj","DispatchNs","BeginNs","EndNs","CompleteNs","DurationNs"
0,"bodyForce(Body*, float, int) [clone .kd]",0,0,0,3670615,3670615,65536,256,0,0,20,4,16,64,"0x0","0x7f7b27c04500",1591372809346673,1591372809872935,1591372836189584,1591372836215944,26316649
1,"bodyForce(Body*, float, int) [clone .kd]",0,0,2,3670615,3670615,65536,256,0,0,20,4,16,64,"0x0","0x7f7b27c04500",1591372836774261,1591372837000949,1591372863116796,1591372863132315,26115847
2,"bodyForce(Body*, float, int) [clone .kd]",0,0,4,3670615,3670615,65536,256,0,0,20,4,16,64,"0x0","0x7f7b27c04500",1591372863652552,1591372863877281,1591372889980009,1591372889994436,26102728
3,"bodyForce(Body*, float, int) [clone .kd]",0,0,6,3670615,3670615,65536,256,0,0,20,4,16,64,"0x0","0x7f7b27c04500",1591372890512133,1591372890735562,1591372916796147,1591372916817087,26060585
4,"bodyForce(Body*, float, int) [clone .kd]",0,0,8,3670615,3670615,65536,256,0,0,20,4,16,64,"0x0","0x7f7b27c04500",1591372917332974,1591372917556629,1591372943652575,1591372943667909,26095946
5,"bodyForce(Body*, float, int) [clone .kd]",0,0,10,3670615,3670615,65536,256,0,0,20,4,16,64,"0x0","0x7f7b27c04500",1591372944181896,1591372944405378,1591372970475883,1591372970491020,26070505
6,"bodyForce(Body*, float, int) [clone .kd]",0,0,12,3670615,3670615,65536,256,0,0,20,4,16,64,"0x0","0x7f7b27c04500",1591372971009527,1591372971233309,1591372997318181,1591372997339821,26084872
7,"bodyForce(Body*, float, int) [clone .kd]",0,0,14,3670615,3670615,65536,256,0,0,20,4,16,64,"0x0","0x7f7b27c04500",1591372997856209,1591372998080743,1591373024164495,1591373024180993,26083752
8,"bodyForce(Body*, float, int) [clone .kd]",0,0,16,3670615,3670615,65536,256,0,0,20,4,16,64,"0x0","0x7f7b27c04500",1591373024701060,1591373024924818,1591373051023611,1591373051040364,26098793
9,"bodyForce(Body*, float, int) [clone .kd]",0,0,18,3670615,3670615,65536,256,0,0,20,4,16,64,"0x0","0x7f7b27c04500",1591373051559851,1591373051782878,1591373077878145,1591373077902255,26095267

```

* Check the statistics result file, one line per kernel

```
cat results.stats.csv
 
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
"bodyForce(Body*, float, int) [clone .kd]",10,261124944,26112494,100.0
```

* Profile the HIP calls with `--hip-trace`

```
rocprof --stats --hip-trace nbody-orig 65536
RPL: on '221130_201416' from '/global/software/rocm/rocm-5.3.0' in '/global/home/gmarko/HIP-Examples/mini-nbody/hip'
RPL: profiling '"nbody-orig" "65536"'
RPL: input file ''
RPL: output dir '/tmp/rpl_data_221130_201416_3670892'
RPL: result dir '/tmp/rpl_data_221130_201416_3670892/input_results_221130_201416'
ROCTracer (pid=3670915):
    HIP-trace()
65536, 161.051
hsa_copy_deps: 0
scan ops data 29:30                                                                                                    File '/global/home/gmarko/HIP-Examples/mini-nbody/hip/results.copy_stats.csv' is generating
dump json 19:20
File '/global/home/gmarko/HIP-Examples/mini-nbody/hip/results.json' is generating
File '/global/home/gmarko/HIP-Examples/mini-nbody/hip/results.hip_stats.csv' is generating
dump json 51:52
File '/global/home/gmarko/HIP-Examples/mini-nbody/hip/results.json' is generating
File '/global/home/gmarko/HIP-Examples/mini-nbody/hip/results.stats.csv' is generating
dump json 9:10
File '/global/home/gmarko/HIP-Examples/mini-nbody/hip/results.json' is generating
```

Now we have new files with the `hip` in their name like below, checl the file `results.hip_stats.csv`


```
 cat results.hip_stats.csv
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
"hipMemcpy",20,486845521,24342276,99.89375113830629
"hipLaunchKernel",10,467008,46700,0.09582337501179998
"hipMalloc",1,30570,30570,0.006272527610042495
"hipFree",1,14210,14210,0.0029156891507590398
"__hipPushCallConfiguration",10,3510,351,0.0007202018943817191
"__hipPopCallConfiguration",10,2520,252,0.0005170680267355932
```

* Profile also the HSA API with the `--hsa-trace`

```
rocprof --stats --hip-trace --hsa-trace nbody-orig 65536
RPL: on '221130_201737' from '/global/software/rocm/rocm-5.3.0' in '/global/home/gmarko/HIP-Examples/mini-nbody/hip'
RPL: profiling '"nbody-orig" "65536"'
RPL: input file ''
RPL: output dir '/tmp/rpl_data_221130_201737_3671219'
RPL: result dir '/tmp/rpl_data_221130_201737_3671219/input_results_221130_201737'
ROCProfiler: input from "/tmp/rpl_data_221130_201737_3671219/input.xml"
  0 metrics
ROCTracer (pid=3671242):
    HSA-trace()
    HSA-activity-trace()
    HIP-trace()
65536, 155.978

ROCPRofiler: 10 contexts collected, output directory /tmp/rpl_data_221130_201737_3671219/input_results_221130_201737
hsa_copy_deps: 1
scan hsa API data 5953:5954                                                                                                    hsa_copy_deps: 0
scan hip API data 51:52                                                                                                    File '/global/home/gmarko/HIP-Examples/mini-nbody/hip/results.csv' is generating
File '/global/home/gmarko/HIP-Examples/mini-nbody/hip/results.stats.csv' is generating
dump json 9:10
File '/global/home/gmarko/HIP-Examples/mini-nbody/hip/results.json' is generating
File '/global/home/gmarko/HIP-Examples/mini-nbody/hip/results.hsa_stats.csv' is generating
dump json 5963:5964
File '/global/home/gmarko/HIP-Examples/mini-nbody/hip/results.json' is generating
File '/global/home/gmarko/HIP-Examples/mini-nbody/hip/results.copy_stats.csv' is generating
dump json 19:20
File '/global/home/gmarko/HIP-Examples/mini-nbody/hip/results.json' is generating
File '/global/home/gmarko/HIP-Examples/mini-nbody/hip/results.hip_stats.csv' is generating
dump json 51:52
File '/global/home/gmarko/HIP-Examples/mini-nbody/hip/results.json' is generating
```

* See the content of the file `results.hsa_stats.csv`

```
cat results.hsa_stats.csv
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
"hsa_signal_wait_scacquire",50,264955082,5299101,82.69977799679005
"hsa_queue_create",1,39868279,39868279,12.443987854568068
"hsa_amd_memory_async_copy",20,4917141,245857,1.5347751249357586
"hsa_amd_signal_async_handler",20,4262555,213127,1.3304608069344652
"hsa_signal_store_screlease",40,1945998,48649,0.60739956889069
"hsa_amd_memory_lock_to_pool",20,1418202,70910,0.44265990170591873
"hsa_amd_memory_unlock",20,723957,36197,0.22596691758953363
"hsa_agent_get_info",80,671007,8387,0.20943976433821374
"hsa_amd_memory_pool_allocate",5,597926,119585,0.18662917157599068
"hsa_system_get_info",5005,367139,73,0.11459419296574767
"hsa_executable_load_agent_code_object",2,216629,108314,0.0676158768966984
"hsa_executable_freeze",2,156380,78190,0.048810504729771616
"hsa_amd_agents_allow_access",4,89470,22367,0.02792605101785821
"hsa_signal_create",57,51500,903,0.016074568318092074
"hsa_code_object_reader_create_from_memory",2,20940,10470,0.006535950690890253
"hsa_isa_get_info_alt",2,18180,9090,0.005674478680056581
"hsa_signal_load_relaxed",236,17880,75,0.005580840418009442
"hsa_system_get_major_extension_table",3,14920,4973,0.004656942899144344
"hsa_amd_profiling_get_async_copy_time",40,13430,335,0.004191872864310224
"hsa_executable_create_alt",2,9030,4515,0.0028185116876188626
"hsa_amd_memory_pool_get_info",106,7570,71,0.002362805478989456
"hsa_amd_agent_iterate_memory_pools",27,7440,275,0.0023222288987690297
"hsa_amd_memory_pool_free",1,4710,4710,0.0014701207141400712
"hsa_executable_get_symbol_by_name",15,3599,239,0.0011233470170255023
"hsa_queue_add_write_index_screlease",20,3500,175,0.0010924463905499467
"hsa_amd_profiling_get_dispatch_time",20,3110,155,0.0009707166498886669
"hsa_signal_silent_store_relaxed",40,2800,70,0.0008739571124399574
"hsa_amd_agent_memory_pool_get_info",19,2490,131,0.0007771975749912477
"hsa_iterate_agents",2,2250,1125,0.0007022869653535371
"hsa_queue_load_read_index_relaxed",20,2240,112,0.0006991656899519659
"hsa_signal_destroy",20,2190,109,0.0006835593129441095
"hsa_queue_load_read_index_scacquire",20,1790,89,0.0005587082968812585
"hsa_executable_symbol_get_info",30,1540,51,0.00048067641184197657
"hsa_amd_profiling_async_copy_enable",1,440,440,0.00013733611766913615
"hsa_agent_iterate_isas",1,400,400,0.00012485101606285104
"hsa_amd_profiling_set_profiler_enabled",1,140,140,4.3697855621997866e-05
"hsa_dispatch",10,0,0,0.0

```

* Download the results.json file on your laptop

From your laptop:
`scp -i id_ed25519 -P 8560 enccs_tr_0@89.202.73.189:/path/results.json .`

* Visit the web page:

https://ui.perfetto.dev/

* Click on the top left menu, "Open Trace File on th eleft top"

* Select the file results.json

Zoom in/out: W/S
Move left/right: A/D

![](https://i.imgur.com/ZKgVBKI.png)


Read about the counters: ` vim /global/software/rocm/rocm-5.3.0/lib/rocprofiler/gfx_metrics.xml`

* Create a file with the contents:
```
cat rocprof_counters.txt
pmc : Wavefronts VALUInsts VFetchInsts VWriteInsts VALUUtilization VALUBusy WriteSize
pmc : SALUInsts SFetchInsts LDSInsts FlatLDSInsts GDSInsts SALUBusy FetchSize
pmc : L2CacheHit MemUnitBusy MemUnitStalled WriteUnitStalled ALUStalledByLDS LDSBankConflict
```

* Execute with using the counters

```
 rocprof --timestamp on -i rocprof_counters.txt  nbody-orig 65536
RPL: on '221130_205737' from '/global/software/rocm/rocm-5.3.0' in '/global/home/gmarko/HIP-Examples/mini-nbody/hip'
RPL: profiling '"nbody-orig" "65536"'
RPL: input file 'rocprof_counters.txt'
RPL: output dir '/tmp/rpl_data_221130_205737_3673574'
RPL: result dir '/tmp/rpl_data_221130_205737_3673574/input0_results_221130_205737'
ROCProfiler: input from "/tmp/rpl_data_221130_205737_3673574/input0.xml"
  gpu_index =
  kernel =
  range =
  7 metrics
    Wavefronts, VALUInsts, VFetchInsts, VWriteInsts, VALUUtilization, VALUBusy, WriteSize
65536, 155.389

ROCPRofiler: 10 contexts collected, output directory /tmp/rpl_data_221130_205737_3673574/input0_results_221130_205737
RPL: result dir '/tmp/rpl_data_221130_205737_3673574/input1_results_221130_205737'
ROCProfiler: input from "/tmp/rpl_data_221130_205737_3673574/input1.xml"
  gpu_index =
  kernel =
  range =
  7 metrics
    SALUInsts, SFetchInsts, LDSInsts, FlatLDSInsts, GDSInsts, SALUBusy, FetchSize
65536, 156.996

ROCPRofiler: 10 contexts collected, output directory /tmp/rpl_data_221130_205737_3673574/input1_results_221130_205737
RPL: result dir '/tmp/rpl_data_221130_205737_3673574/input2_results_221130_205737'
ROCProfiler: input from "/tmp/rpl_data_221130_205737_3673574/input2.xml"
  gpu_index =
  kernel =
  range =
  6 metrics
    L2CacheHit, MemUnitBusy, MemUnitStalled, WriteUnitStalled, ALUStalledByLDS, LDSBankConflict
65536, 155.264

ROCPRofiler: 10 contexts collected, output directory /tmp/rpl_data_221130_205737_3673574/input2_results_221130_205737
File '/global/home/gmarko/HIP-Examples/mini-nbody/hip/rocprof_counters.csv' is generating
```

* Contents of the rocprof_counters.csv file
```
cat rocprof_counters.csv
Index,KernelName,gpu-id,queue-id,queue-index,pid,tid,grd,wgr,lds,scr,arch_vgpr,accum_vgpr,sgpr,wave_size,sig,obj,Wavefronts,VALUInsts,VFetchInsts,VWriteInsts,VALUUtilization,VALUBusy,WriteSize,SALUInsts,SFetchInsts,LDSInsts,FlatLDSInsts,GDSInsts,SALUBusy,FetchSize,L2CacheHit,MemUnitBusy,MemUnitStalled,WriteUnitStalled,ALUStalledByLDS,LDSBankConflict,DispatchNs,BeginNs,EndNs,CompleteNs
0,"bodyForce(Body*, float, int) [clone .kd]",0,0,0,3673711,3673711,65536,256,0,0,20,4,16,64,0x0,0x7f2b4d282500,2048.0000000000,1212443.0000000000,12.0000000000,12.0000000000,100.0000000000,68.1476813493,7872.0000000000,131228.5000000000,65553.0000000000,0.0000000000,0.0000000000,0.0000000000,6.3483148000,9429.1875000000,96.5684331443,0.0250344612,0.0044357832,0.0102024550,0.0000000000,0.0000000000,1594244102859719,1594244111978746,1594244138305243,1594244138330792
...
```

### Omnitrace

We have made special builds of the Omnitools, omnitrace and omniperf for use in the exercises

* Load Omnitrace

```
module use --append /global/training/enccs/modules/
module load omnitrace/1.7.3
```
* Reserve a GPU

* Check the various options and their values and also a second command for description

`srun -n 1 --gpus 1 omnitrace-avail --categories omnitrace`
`srun -n 1 --gpus 1 omnitrace-avail --categories omnitrace --brief --description`

* Create an Omnitrace configuration file with description per option

`srun -n 1 omnitrace-avail -G omnitrace_all.cfg --all`

* Declare to use this configuration file: `export OMNITRACE_CONFIG_FILE=/path/omnitrace_all.cfg`

* Get the file `cp /global/software/rocm/rocm-5.3.0/share/hip/samples/2_Cookbook/0_MatrixTranspose/MatrixTranspose.cpp .`
* Compile ` hipcc -o MatrixTranspose MatrixTranspose.cpp`
* Execute the binary: `time srun -n 1 --gpus 1 ./MatrixTranspose` and check the duration

#### Dynamic instrumentation

* Execute dynamic instrumentation: `time srun –n 1 –gpus 1 omnitrace -- ./MatrixTranspose` and check the duration
* Check what the binary calls and gets instrumented: `nm --demangle MatrixTranspose | egrep -i ' (t|u) '`
* Available functions to instrument: `srun -n 1 --gpus 1 omnitrace -v -1 --simulate --print-available functions -- ./MatrixTranspose`
    * the simulate option means that it will not execute the binary 

#### Binary rewriting
* Binary rewriting: `srun -n 1 --gpus 1 omnitrace -v -1 --print-available functions -o matrix.inst -- ./MatrixTranspose`
    * We created a new instrumented binary called matrix.inst 

* Executing the new instrumented binary: `time srun -n 1 --gpus 1 ./matrix.inst` and check the duration
* See the list of the instrumented GPU calls: `cat omnitrace-matrix.inst-output/TIMESTAMP/roctracer.txt`
#### Visualization

* Copy the `perfetto-trace.proto` to your laptop, open the web page https://ui.perfetto.dev/ click to open the trace and select the file

#### Hardware counters

* See a list of all the counters: `srun -n 1 --gpus 1 omnitrace-avail --all`
* Declare in your configuration file: `OMNITRACE_ROCM_EVENTS = GPUBusy,Wavefronts,VALUBusy,L2CacheHit,MemUnitBusy`
* Execute: `srun -n 1 --gpus 1 ./matrix.inst` and copy the perfetto file and visualize

#### Sampling

Activate in your configuration file `OMNITRACE_USE_SAMPLING = true` and `OMNITRACE_SAMPLING_FREQ = 100`, execute and visualize

#### Kernel timings

* Open the file `omnitrace-binary-output/timestamp/wall_clock.txt`  (replace binary and timestamp with your information)
* In order to see the kernels gathered in your configuration file, make sure that `OMNITRACE_USE_TIMEMORY = true` and `OMNITRACE_FLAT_PROFILE = true`, execute the code and open again the file `omnitrace-binary-output/timestamp/wall_clock.txt`


---

### Omniperf

We have made special builds of the Omnitools, omnitrace and omniperf for use in the exercises

* Load Omniperf

```
module use --append /global/training/enccs/modules/
module load omniperf/1.0.4
```
* Reserve a GPU, compile the exercise and execute Omniperf, observe how many times the code is executed

```
 salloc -N1 -p MI250 --reservation=enccs_3 --gpus=1 --time 00:10:00
cp /global/training/enccs/omniperf/1.0.4/share/sample/vcopy.cpp .
hipcc -o vcopy vcopy.cpp
srun -n 1 --gpus 1 omniperf profile -n vcopy_all -- ./vcopy 1048576 256
```

* Run `srun -n 1 --gpus 1 omniperf profile -h` to see all the options

* Now is created a workload in the directory workloads with the name vcopy_all (the argument of the -n). So, we can analyze it

```
 srun -n 1 --gpus 1 omniperf analyze -p workloads/vcopy_all/mi200/ &> vcopy_analyze.txt
```

There is no need for srun to analyze but we want to avoid everybody to use the login node. Explore the file `vcopy_analyze.txt`

* We can select specific IP Blocks, like:

```
srun -n 1 --gpus 1 omniperf analyze -p workloads/vcopy_all/mi200/ -b 7.1.2
```

But you need to know the code of the IP Block

* If you have installed Omniperf on your laptop (no ROCm required for analysis) then you can download the data and execute:

```
omniperf analyze -p workloads/vcopy_all/mi200/ --gui
```

* Open the web page: http://172.21.7.117:8050/


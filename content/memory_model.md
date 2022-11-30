# AMD Node Memory Model

{download}`Slides <slides/AMDNodeMemoryModel.pdf>`

## Exercises

### Managed Memory

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
 
### OpenMP Atomics
 
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
 
 The examples in arraysum4.c to arraysum7.c show the same atomic behavior with add the elements of array b to array a and storing it back to array a. 
 
 

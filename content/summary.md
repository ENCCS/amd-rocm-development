# Summary and outlook

## AMD GPU Accelerated Applications Catalog

Full list of AMD GPU Accelerated Application listed in the 
[GPU accelerated applications catalog](https://www.amd.com/system/files/documents/gpu-accelerated-applications-catalog.pdf)

The list includes a lot of tools and libraries that can be useful to application developers. Here is the list of
them from the catalog.

Note that the level of support for AMD and by AMD varies a lot across the list.

## Tools and Libraries

|Application Name             |Developer/Publisher |Description          |
|------------------|------------------------------------------|---------------------|
|AMG Solve | Lawrence Livermore National Library   | Parallel Algebraic Multigrid Solver library |
|AMReX | LBNL, NREL, and ANL  | Block-structured adaptive mesh refinement framework         |
|Cosma | ETH Zurich: Swiss Federal Institute of Technology in Zurich, <br> CSCS: Swiss National Computing Centre, <br> PASC: Platform for Advanced Scientific Computing, <br> ERC: European Research Council, MaX: Materials design at the Exascale | Matrix-matrix multiplication algorithm. |
|DBCSR | CP2K Developers Group | Sparse matrix-matrix multiplication and more. |
|Devito | NUMFOCUS | Domain-specific language (DSL) for finite difference kernels.|
|Ginkgo | Karlsruhe Institute of Technology, University of Tennessee, Universitat Jaume I | Sparse linear algebra library |
|Hashcat | Jens ‘atom’ Steube, Gabriele ‘matrix’ Gristina | Password recovery utility |
|hipMagma | The University of Tennessee   | Dense matrix algebra library |
|HPCToolkit | Rice University | Performance analysis tool |
|HP Developer Tools | HP | Large toolsuite on HPE systems |
|Kokkos | Sandia | Portable programming model in C++ |
|PAPI | Innovative Computing Laboratory at the University of Tennessee, Knoxville | Profiling tool for performance counters |
|PYFR | Engineering and Physical Sciences Research Council, Innovate UK, <br> the European Commission, BAE Systems, Airbus, <br> and the Air Force Office of Scientific Research. <br> We are also grateful for hardware donations from Nvidia, Intel, and AMD. | Python framework for advection-diffusion problems           |
|Raja  | Lawrence Livermore National Library | Portable programming abstractions for C++ |
|SIRIUS | SFI Centre for Research-based Innovation, University of Oslo            | Scalable data access for oil and gas. |
|SLATE          | Innovative Computing Laboratory, Exascale Computing Project | Dense linear algebra and more |
|TAU    | University of Oregon Performance Research Lab <br> The LANL Advanced Computing Laboratory, <br> and the Research Centre Jülich at ZAM, Germany. | Performance analysis tool    |
|Totalview      | Perforce  | Debugging tool |
|Trace Compass | Eclipse | Trace analysis tool |
|Vampir | Center for Applied Mathematics of Research Centre Jülich <br> and the Center for High Performance Computinn of the Technische Universität Dresden | Performance analysis tool  |


## Reporting issues

**Recommendation for best tracking of issues**:
- Report to your primary site 
    - LUMI
    - KTH

**Research projects (omnitrace, omniperf) are tracking github issues.**

**Siemens compiler is still in development:**
- Issues should be reported to AMD team and they will pass it on to Siemens
- Once the code gets upstreamed, it will be the regular GCC issues path

**Can report to AMD staff for help on proper reporting path.**

## Additional Resources

- LAB-NOTES: New “lab-notes” series on [GPUOpen.com](https://GPUOpen.com)
    - Finite Difference Method – Laplacian Part 1 of a series
    - AMD Matrix Cores – using the matrix math operations
    - [Search "lab notes"](https://gpuopen.com/?s=lab+notes) for all articles in the series
    - Will be released on a two-week cadence
- [Developer resources](https://developer.amd.com)
    - Videos on HIP and GPU programming
- [ROCm documentation](https://docs.amd.com)
- Quick start guides at Oak Ridge National Laboratory
    - [Crusher quick-start guide](https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html)
    - [Frontier user guide](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html)
- [Infinity Hub](https://www.amd.com/en/technologies/infinity-hub) with many popular applications ported to AMD GPUs
- AMD Cloud resources for trying out AMD systems
    - [AMD Cloud Platform (ACP)](https://acp.amd.com)
    - [AMD Accelerator Cloud (AAC)](https://www.amd.com/en/solutions/accelerated-computing)


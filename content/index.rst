Developing Applications with the AMD ROCm Ecosystem
===================================================

This training material is created by AMD in collaboration with ENCCS.
It covers how to develop and port
applications to run on AMD GPU and CPU hardware on top AMD-powered
supercomputers. You will learn about the ROCm software
development languages, libraries, and tools, as well as getting a
developer’s view of the hardware that powers the system. The material
focuses mostly on how to program applications to run on the GPU.



.. prereq::

   It is useful to have prior experience developing HPC applications,
   and some understanding of recent HPC computer hardware and the
   Linux operating system.



.. toctree::
   :maxdepth: 1
   :caption: The lesson

   intro_hip
   porting_hip
   openMP
   fortran
   exercises-1
   architecture
   mpi
   memory_model
   hierarchical_roofline
   affinity
   profilers
   openMP_offload
   ML_frameworks
   summary
      
.. toctree::
   :maxdepth: 1
   :caption: Reference

   quick-reference
   guide



.. _learner-personas:

Who is the course for?
----------------------





About the course
----------------

4 half-day schedule
^^^^^^^^^^^^^^^^^^^

**Day 1, 2022, 13:00-17:00 — Programming Environments**

- HIP (George Markomanolis, AMD)
- HIP Exercises on ACP cloud
- Break
- Hipify — porting applications to HIP
- Hipify exercises
- Getting Started with OpenMP® Offload Applications on AMD Accelerators (Jose Noudohouenou, AMD)
- Break
- OpenMP exercises
- Developing Fortran Applications, HIPFort & Flang (Bob Robey and Brian Cornille, AMD)
- Fortran exercises -- HIPFort

**Day 2, 13:00-17:00 — Understanding the Hardware**

- The AMD MI250X GPUs (George Markomanolis, AMD)
- AMD Communication Fabrics (Mahdieh Ghazimirsaeed,AMD)
- Memory Systems (Bob Robey, AMD) 
- Break
- Exercises – MPI Benchmark and Memory Systems
- Roofline Model (Noah Wolfe, AMD) 
- Affinity — Placement, Ordering and Binding (Gina Sitaraman and Bob Robey, AMD) 
- Exercises -- Affinity

**Day 3, 13:00-17:00 — Tools**

- Profiler - rocprof
- Exercises - rocprof
- Profiler - Omnitrace
- Break
- Profiler - Omniperf
- Break
- Debuggers — rocgdb
- Debugging - exercises

**Day 4, 13:00-17:00 — Special Topics**

- Using OpenMP® (Michael Klemm, AMD)
- Break
- Introduction to ML frameworks (Alessandro Fanfarillo, AMD)
- Break
- Discussion and feedback






See also
--------

- **LAB-NOTES: New “lab-notes” series on https://GPUOpen.com**
   - Finite Difference Method – Laplacian Part 1 of a series
   - AMD Matrix Cores – using the matrix math operations
   - https://gpuopen.com/?s=lab+notes for all articles in the series
   - Will be released on a two-week cadence
- **Developer resources https://developer.amd.com**
   - Videos on HIP and GPU programming
- **ROCm documentation https://docs.amd.com**
- **Quick start guides at Oak Ridge National Laboratory**
   - https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html
   - https://docs.olcf.ornl.gov/systems/frontier_user_guide.html 
- **Infinity Hub with many popular applications ported to AMD GPUs**
   - https://www.amd.com/en/technologies/infinity-hub 
- **AMD Cloud resources for trying out AMD systems**
   - AMD Cloud Platform (ACP) - https://acp.amd.com
   - AMD Accelerator Cloud (AAC) - https://www.amd.com/en/solutions/accelerated-computing




Credits
-------

The lesson file structure and browsing layout is inspired by and derived from
`work <https://github.com/coderefinery/sphinx-lesson>`__ by `CodeRefinery
<https://coderefinery.org/>`__ licensed under the `MIT license
<http://opensource.org/licenses/mit-license.html>`__. We have copied and adapted
most of their license text.

Instructional Material
^^^^^^^^^^^^^^^^^^^^^^

This instructional material is made available under the
`Creative Commons Attribution license (CC-BY-4.0) <https://creativecommons.org/licenses/by/4.0/>`__.
The following is a human-readable summary of (and not a substitute for) the
`full legal text of the CC-BY-4.0 license
<https://creativecommons.org/licenses/by/4.0/legalcode>`__.
You are free to:

- **share** - copy and redistribute the material in any medium or format
- **adapt** - remix, transform, and build upon the material for any purpose,
  even commercially.

The licensor cannot revoke these freedoms as long as you follow these license terms:

- **Attribution** - You must give appropriate credit (mentioning that your work
  is derived from work that is Copyright (c) ENCCS and individual contributors and, where practical, linking
  to `<https://enccs.github.io/sphinx-lesson-template>`_), provide a `link to the license
  <https://creativecommons.org/licenses/by/4.0/>`__, and indicate if changes were
  made. You may do so in any reasonable manner, but not in any way that suggests
  the licensor endorses you or your use.
- **No additional restrictions** - You may not apply legal terms or
  technological measures that legally restrict others from doing anything the
  license permits.

With the understanding that:

- You do not have to comply with the license for elements of the material in
  the public domain or where your use is permitted by an applicable exception
  or limitation.
- No warranties are given. The license may not give you all of the permissions
  necessary for your intended use. For example, other rights such as
  publicity, privacy, or moral rights may limit how you use the material.


Software
^^^^^^^^

Except where otherwise noted, the example programs and other software provided
with this repository are made available under the `OSI <http://opensource.org/>`__-approved
`MIT license <https://opensource.org/licenses/mit-license.html>`__.

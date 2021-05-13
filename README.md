CONTENTS OF THIS FILE
---------------------

 * Introduction
 * Requirements
 * Installation
 * Maintainers

To learn more about VGL framework, please visit [vgl.parallel.ru](https://vgl.parallel.ru) website.

INTRODUCTION
------------
 
 VGL is a high-performance graph processing framework, designed for modern NEC SX-Aurora TSUBASA vector architecture. NEC SX-Aurora TSUBASA is equipped with world highest memory bandwidth of 1.2-1.5 TB/s, which allows to significantly accelerate graph-processing.
 
 VGL significantly outperforms many state-of the art graph-processing frameworks for modern multicore CPUs and NVIDIA GPUs, such as Gunrock, CuSHA, Ligra, Galois, GAPBS
 
Requirements
------------
 
 GCC compiler, version >= 8.3.0
 NCC compiler, version >= 3.1.0
 ASL (Advanced Scientific Library), version >= 2.1.0
 Python, version >= 3.6
 
 (additional for GPU API) In order to use VGL GPU API you must also have CUDA toolkit, version >= 10.0
 (additional for automatic benchmarking) GNU bash, version >= 4.0
 
Installation
------------
 
 1. Download VGL:
 
     - source files from GitHub:
         > git clone https://github.com/afanasyev-ilya/VectorGraphLibrary
 
     - or .tar arcieve from this website:
 
         > wget https://afanasyev-ilya.github.io/vgl_site/VectorGraphLibrary.zip
         >                               
         > wget https://afanasyev-ilya.github.io/vgl_site/VectorGraphLibrary.tar.gz
 
 2. Check sample applications in _apps_ folder:
 
     > cd VectorGraphLibrary/apps
     >
     > vim bfs/bfs.cpp
 
 3. Modify Makefile.nec if neccesarry:
     > vim Makefile.nec
     
     you may change compiler paths:
     > VE_CXX = nc++
     > 
     > VH_CXX = g++
     
     or the path to ASL:
     > -I /opt/nec/ve/nlc/2.0.0/include/
 
 4. Build VGL samples:
     - all sample applications:
       > make -f Makefile.nec all

     - or a specific application (for example shortest paths):
       > make -f Makefile.nec sssp
       
 5. Run automatic verification and performance testing
     - prepare testing data and compiler sources:
       > python3 ./run_tests.py --arch=sx --compile --prepare
     - verification:
       > python3 ./run_tests.py --arch=sx --verify
     - performance evaluation:
       > python3 ./run_tests.py --arch=sx  --benchmark
       > file benchmarking_results_sx.xlsx is created with performance results for all algorithms

 6. Using VGL on Lomonosov-2 supercomputer:
     - Lomonosov-2 supercomputer is now equipped with NEC SX-Aurora TSUBASA vector engines.
     To start using VGL load the following modules:
    
        (1) nec/2.13.0 
        (2) anaconda3/3.7.0  
        (3) slurm/15.08.1
        (4) gcc/9.1
    
    - Download VGL to your home directory /home/user_name/ (Warning! you can'd download it to _scratch or its subfolders)
      > git clone https://github.com/afanasyev-ilya/VectorGraphLibrary

    - Compile VGL on head node using either:
      > make -f Makefile.nec all

    - Download input graphs from head node:
      > python3 ./run_tests.py --arch=sx --download-only

    - Prepare VGL input graphs on vector host of any partition (nec/pascal/test/etc):
      > sbatch -p nec ./graph_library.sh python3 ./run_tests.py --arch=sx --prepare

    - Run benchmarks and tests on vector engines:
      > sbatch -p nec ./graph_library.sh python3 ./run_tests.py --arch=sx --benchmark --verify

Maintainers
------------
 
 VGL is maintained by Ilya Afanasyev, afanasiev_ilya@icloud.com.
CONTENTS OF THIS FILE
---------------------

 * Introduction
 * Requirements
 * Installation
 * Maintainers

To learn more about VGL framework, please visit [vgl.parallel.ru](vgl.parallel.ru) website.

INTRODUCTION
------------
 
 VGL is a high-performance graph processing framework, designed for modern NEC SX-Aurora TSUBASA vector architecture. NEC SX-Aurora TSUBASA is equipped with world highest memory bandwidth of 1.2-1.5 TB/s, which allows to significantly accelerate graph-processing.
 
 VGL significantly outperforms many state-of the art graph-processing frameworks for modern multicore CPUs and NVIDIA GPUs, such as Gunrock, CuSHA, Ligra, Galois, GAPBS
 
Requirements
------------
 
 GCC compiler, version >= 8.3.0
 NCC compiler, version >= 3.1.0
 ASL (Advanced Scientific Library), version >= 2.1.0
 
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
 
Maintainers
------------
 
 VGL is maintained by Ilya Afanasyev, afanasiev_ilya@icloud.com.
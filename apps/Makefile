ifeq ($(MPI), true)
	MPI_Flags= -D __USE_MPI__
	BinSuffix=_mpi
endif

# Include paths...

COMPILER=$(CXX)

ifeq ($(COMPILER),g++)
	ifeq ($(MPI), true)
		ARCH_COMPILER=mpicxx
	else
		ARCH_COMPILER=g++
	endif
	Include_Path = -I ../
    Flags = -D __USE_MULTICORE__ $(MPI_Flags) -O3 -fopenmp -ftree-vectorize -std=c++17
  	Libraries = -O3 -fopenmp
    ArchSuffix=_mc
endif

ifeq ($(COMPILER),icpc)
	ifeq ($(MPI), true)
		ARCH_COMPILER=mpicxx
	else
		ARCH_COMPILER=icpc
	endif
	Include_Path = -I ../
	Flags = -D __USE_MULTICORE__ $(MPI_Flags) -O1 -fopenmp -std=c++17 #-xCOMMON-AVX512
	Libraries = -O3 -fopenmp
	ArchSuffix=_mc
endif

ifeq ($(COMPILER),nvcc)
	CUDA_DIR = /opt/cuda/cuda-10.1/
	CUDA_COMPILER = $(CUDA_DIR)/bin/nvcc
	Include_Path = -I $(CUDA_DIR)/include -I ../external_libraries/cub -I ../
	Flags = -O2 -D __USE_GPU__ -x cu -w -m64 -std=c++11 -Xptxas -dlcm=ca --expt-extended-lambda -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -Xcompiler -fopenmp
	Library_Path = -L $(CUDA_DIR)/lib -L $(CUDA_DIR)/lib64
	Libraries = -lcudart -lcudadevrt -lcudadevrt -Xcompiler -fopenmp
	ArchSuffix=_cu
endif

ifeq ($(COMPILER),nc++)
	ifeq ($(MPI), true)
		ARCH_COMPILER=mpic++
	else
		ARCH_COMPILER=nc++
	endif
	Include_Path = -I /opt/nec/ve/nlc/2.3.0/include/ -I ../ #2.0.0 was here for lom-2
	Flags = -D __USE_NEC_SX_AURORA__ $(MPI_Flags) -O3 -fopenmp -mparallel -mretain-list-vector -DASL_LIB_INT64 -report-file=vector_info.txt# -proginf -report-all -mno-vector #-ftrace
	Libraries = -fopenmp -ftrace -lasl_openmp_i64 # -report-all -ftrace
	ArchSuffix=_sx
endif

.DEFAULT_GOAL := all

##########
# binaries
##########

install:
	if [ ! -d ../external_libraries/cub ];then\
                        git clone https://github.com/NVlabs/cub ../external_libraries/cub;\
                fi
	if [ ! -d ../external_libraries/moderngpu ];then\
                        git clone https://www.github.com/moderngpu/moderngpu ../external_libraries/moderngpu;\
                fi

all: sssp sswp bfs scc pr cc rw hits coloring tc mf create_vgl_graphs

sssp: create_folders sssp.o
	$(ARCH_COMPILER) object_files/sssp.o $(Library_Path) $(Libraries) -o ./bin/sssp$(ArchSuffix)$(BinSuffix)

sswp: create_folders sswp.o
	$(ARCH_COMPILER) object_files/sswp.o $(Library_Path) $(Libraries) -o ./bin/sswp$(ArchSuffix)$(BinSuffix)

sssp_sharded: create_folders sssp_sharded.o
	$(ARCH_COMPILER) object_files/sssp_sharded.o $(Library_Path) $(Libraries) -o ./bin/sssp_sharded$(ArchSuffix)$(BinSuffix)

bfs: create_folders bfs.o
	$(ARCH_COMPILER) object_files/bfs.o $(Library_Path) $(Libraries) -o ./bin/bfs$(ArchSuffix)$(BinSuffix)

scc: create_folders scc.o
	$(ARCH_COMPILER) object_files/scc.o $(Library_Path) $(Libraries) -o ./bin/scc$(ArchSuffix)$(BinSuffix)

cc: create_folders cc.o
	$(ARCH_COMPILER) object_files/cc.o $(Library_Path) $(Libraries) -o ./bin/cc$(ArchSuffix)$(BinSuffix)

pr: create_folders pr.o
	$(ARCH_COMPILER) object_files/pr.o $(Library_Path) $(Libraries) -o ./bin/pr$(ArchSuffix)$(BinSuffix)

rw: create_folders rw.o
	$(ARCH_COMPILER) object_files/rw.o $(Library_Path) $(Libraries) -o ./bin/rw$(ArchSuffix)$(BinSuffix)

hits: create_folders hits.o
	$(ARCH_COMPILER) object_files/hits.o $(Library_Path) $(Libraries) -o ./bin/hits$(ArchSuffix)$(BinSuffix)

apsp: create_folders apsp.o
	$(ARCH_COMPILER) object_files/apsp.o $(Library_Path) $(Libraries) -o ./bin/apsp$(ArchSuffix)$(BinSuffix)

coloring: create_folders coloring.o
	$(ARCH_COMPILER) object_files/coloring.o $(Library_Path) $(Libraries) -o ./bin/coloring$(ArchSuffix)$(BinSuffix)

tc: create_folders tc.o
	$(ARCH_COMPILER) object_files/tc.o $(Library_Path) $(Libraries) -o ./bin/tc$(ArchSuffix)$(BinSuffix)

mf: create_folders mf.o
	$(ARCH_COMPILER) object_files/mf.o $(Library_Path) $(Libraries) -o ./bin/mf$(ArchSuffix)$(BinSuffix)

print_graph_info: create_folders print_graph_info.o
	$(ARCH_COMPILER) object_files/print_graph_info.o $(Library_Path) $(Libraries) -o ./bin/print_graph_info$(ArchSuffix)$(BinSuffix)

create_vgl_graphs: create_folders create_vgl_graphs.o
	g++ -fopenmp object_files/create_vgl_graphs.o -o ./bin/create_vgl_graphs$(ArchSuffix)$(BinSuffix)

mpi_graphs: create_folders mpi_graphs.o
	$(ARCH_COMPILER) object_files/mpi_graphs.o $(Library_Path) $(Libraries) -o ./bin/mpi_graphs$(ArchSuffix)$(BinSuffix)

##########
# CPPs
##########

sssp.o: sssp/sssp.cpp
	$(ARCH_COMPILER) $(Flags) $(Include_Path)  -c sssp/sssp.cpp -o object_files/sssp.o

sswp.o: sswp/sswp.cpp
	$(ARCH_COMPILER) $(Flags) $(Include_Path)  -c sswp/sswp.cpp -o object_files/sswp.o

sssp_sharded.o: sssp/sssp_sharded.cpp
	$(ARCH_COMPILER) $(Flags) $(Include_Path)  -c sssp/sssp_sharded.cpp -o object_files/sssp_sharded.o

bfs.o: bfs/bfs.cpp
	$(ARCH_COMPILER) $(Flags) $(Include_Path)  -c bfs/bfs.cpp -o object_files/bfs.o

scc.o: scc/scc.cpp
	$(ARCH_COMPILER) $(Flags) $(Include_Path)  -c scc/scc.cpp -o object_files/scc.o

cc.o: cc/cc.cpp
	$(ARCH_COMPILER) $(Flags) $(Include_Path)  -c cc/cc.cpp -o object_files/cc.o

pr.o: pr/pr.cpp
	$(ARCH_COMPILER) $(Flags) $(Include_Path)  -c pr/pr.cpp -o object_files/pr.o

rw.o: rw/rw.cpp
	$(ARCH_COMPILER) $(Flags) $(Include_Path)  -c rw/rw.cpp -o object_files/rw.o

hits.o: hits/hits.cpp
	$(ARCH_COMPILER) $(Flags) $(Include_Path)  -c hits/hits.cpp -o object_files/hits.o

apsp.o: apsp/apsp.cpp
	$(ARCH_COMPILER) $(Flags) $(Include_Path)  -c apsp/apsp.cpp -o object_files/apsp.o

coloring.o: coloring/coloring.cpp
	$(ARCH_COMPILER) $(Flags) $(Include_Path)  -c coloring/coloring.cpp -o object_files/coloring.o

tc.o: tc/tc.cpp
	$(ARCH_COMPILER) $(Flags) $(Include_Path)  -c tc/tc.cpp -o object_files/tc.o

mf.o: mf/mf.cpp
	$(ARCH_COMPILER) $(Flags) $(Include_Path)  -c mf/mf.cpp -o object_files/mf.o

print_graph_info.o: utilites/print_graph_info.cpp
	$(ARCH_COMPILER) $(Flags) $(Include_Path)  -c utilites/print_graph_info.cpp -o object_files/print_graph_info.o

create_vgl_graphs.o: utilites/create_vgl_graphs.cpp
	g++ $(Include_Path) -D __USE_MULTICORE__ -fopenmp -c utilites/create_vgl_graphs.cpp -o object_files/create_vgl_graphs.o

mpi_graphs.o: misc/mpi_graphs.cpp
	$(ARCH_COMPILER) $(Flags) $(Include_Path)  -c misc/mpi_graphs.cpp -o object_files/mpi_graphs.o


create_folders:
	-mkdir -p ./bin
	-cp graph_library.sh ./bin
	-mkdir -p ./object_files

clean:
	-rm -f object_files/*.o
	-rm -f bin/*$(ArchSuffix)*

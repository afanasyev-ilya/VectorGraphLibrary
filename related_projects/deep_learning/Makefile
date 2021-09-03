ifeq ($(MPI), true)
	MPI_Flags= -D __USE_MPI__
	BinSuffix=_mpi
endif

# Include paths...

COMPILER=$(CXX)

ifeq ($(COMPILER),g++)
	Include_Path = -I ../../
	Flags = -D __USE_MULTICORE__ $(MPI_Flags) -O3 -fopenmp -ftree-vectorize -std=c++17
	Libraries = -O3 -fopenmp
	ArchSuffix=_mc
endif

ifeq ($(COMPILER),icpc)
	Include_Path = -I ../../
	Flags = -D __USE_MULTICORE__ $(MPI_Flags) -O1 -fopenmp -std=c++17 #-xCOMMON-AVX512
	Libraries = -O3 -fopenmp
	ArchSuffix=_mc
endif

ifeq ($(COMPILER),nvcc)
	CUDA_DIR = /opt/cuda/cuda-10.1/
	CUDA_COMPILER = $(CUDA_DIR)/bin/nvcc
	Include_Path = -I $(CUDA_DIR)/include -I ../external_libraries/cub -I ../../
	Flags = -O2 -D __USE_GPU__ -x cu -w -m64 -std=c++11 -Xptxas -dlcm=ca --expt-extended-lambda -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -Xcompiler -fopenmp
	Library_Path = -L $(CUDA_DIR)/lib -L $(CUDA_DIR)/lib64
	Libraries = -lcudart -lcudadevrt -lcudadevrt -Xcompiler -fopenmp
	ArchSuffix=_cu
endif

ifeq ($(COMPILER),nc++)
	Include_Path = -I /opt/nec/ve/nlc/2.0.0/include/ -I ../
	Flags = -D __USE_NEC_SX_AURORA__ $(MPI_Flags) -O3 -fopenmp -mparallel -mretain-list-vector -DASL_LIB_INT64 -report-file=vector_info.txt# -proginf -report-all -mno-vector #-ftrace
	Libraries = -fopenmp -ftrace -lasl_openmp_i64 # -report-all -ftrace
	ArchSuffix=_sx
endif

.DEFAULT_GOAL := all

##########
# binaries
##########

all: create_folders generate_learning_data

generate_learning_data: generate_learning_data.o
	$(CXX) object_files/generate_learning_data.o $(Library_Path) $(Libraries) -o ./generate_learning_data.out

##########
# CPPs
##########

generate_learning_data.o: generate_learning_data.cpp
	$(CXX) $(Flags) $(Include_Path)  -c generate_learning_data.cpp -o object_files/generate_learning_data.o

create_folders:
	-mkdir -p ./object_files

clean:
	-rm -f object_files/*.o
	-rm -f bin/*$(ArchSuffix)*

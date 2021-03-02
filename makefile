##############################################################
# CUDA Sample Makefile                                       #
#                                                            #
# Author      : Jieqing Gan                                 #
# Version     : 3.5                                          #
# Date        : 10/01/2014                                   #
# Discription : generic Makefile for making CUDA programs    #
##############################################################

BIN               := gpusphere


# Compilers
CUDA_PATH ?= /usr/local/cuda-5.0
NVCC       := /usr/local/cuda-5.0/bin/nvcc 
CXX        := g++

# Paths
INCD =  -I. -I/usr/local/cuda-5.0/include
LIBS = -L"$(CUDA_PATH)/lib64" -lcudart

# internal flags
#NVCCFLAGS  :=
CCFLAGS     :=
NVCCLDFLAGS :=
LDFLAGS     :=

# CUDA code generation flags
GENCODE_FLAGS := -arch=compute_35 -code=sm_35,compute_35
#
# Program-specific
CPP_SOURCES       := 
CU_SOURCES        := dempacking.cu
CPP_OBJS          := 
CU_OBJS           := $(patsubst %.cu, %.cu_o, $(CU_SOURCES))


# Build Rules
%.cu_o : %.cu 
	$(NVCC) $(GENCODE_FLAGS) -c $(INCD) -o $@ $<

%.o: %.cpp 
	$(CXX) -c $(INCD) -o $@ $<

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $(BIN) $(CU_OBJS) $(INCD) $(LIBS) 

clean:
	rm -f $(BIN) *.o *.cu_o
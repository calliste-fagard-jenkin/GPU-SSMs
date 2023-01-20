CUDA_ROOT_DIR=/usr/local/cuda-11.3
CC=g++
CC_FLAGS=
CC_LIBS=
NVCC=nvcc
NVCC_FLAGS=
NVCC_LIBS=
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
CUDA_LINK_LIBS= -lcudart -lcurand
#USECURAND:=1
SRC_DIR = src
OBJ_DIR = bin
INC_DIR = include
EXE = perform_mcmc
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/stolenR.o $(OBJ_DIR)/sealCode.o $(OBJ_DIR)/obtainDeviceInfo.o
$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)
$(OBJ_DIR)/%.o : %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp include/%.h
	$(CC) $(CC_FLAGS) -c $< -o $@
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)
clean:
	$(RM) bin/* *.o $(EXE)
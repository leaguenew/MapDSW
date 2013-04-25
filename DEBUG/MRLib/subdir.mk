################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../MRLib/Common.cpp 

CU_SRCS += \
../MRLib/MemAlloc.cu \
../MRLib/SMCahce.cu \
../MRLib/TaskScheduler.cu 

CU_DEPS += \
./MRLib/MemAlloc.d \
./MRLib/SMCahce.d \
./MRLib/TaskScheduler.d 

OBJS += \
./MRLib/Common.o \
./MRLib/MemAlloc.o \
./MRLib/SMCahce.o \
./MRLib/TaskScheduler.o 

CPP_DEPS += \
./MRLib/Common.d 


# Each subdirectory must supply rules for building sources it contributes
MRLib/%.o: ../MRLib/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/usr/local/cuda-5.0/include -G -g -O0  -odir "MRLib" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -I/usr/local/cuda-5.0/include -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

MRLib/%.o: ../MRLib/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/usr/local/cuda-5.0/include -G -g -O0  -odir "MRLib" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -I/usr/local/cuda-5.0/include -O0 -g  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



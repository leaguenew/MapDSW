################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../UtilLib/GpuUtil.cu \
../UtilLib/hash.cu 

CU_DEPS += \
./UtilLib/GpuUtil.d \
./UtilLib/hash.d 

OBJS += \
./UtilLib/GpuUtil.o \
./UtilLib/hash.o 


# Each subdirectory must supply rules for building sources it contributes
UtilLib/%.o: ../UtilLib/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/usr/local/cuda-5.0/include -O2 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -odir "UtilLib" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -I/usr/local/cuda-5.0/include -O2 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_21  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



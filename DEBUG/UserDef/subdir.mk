################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../UserDef/Mapreduce.cu 

CU_DEPS += \
./UserDef/Mapreduce.d 

OBJS += \
./UserDef/Mapreduce.o 


# Each subdirectory must supply rules for building sources it contributes
UserDef/%.o: ../UserDef/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/usr/local/cuda-5.0/include -G -g -lineinfo -pg -O0 -gencode arch=compute_20,code=sm_21 -odir "UserDef" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --device-c -G -I/usr/local/cuda-5.0/include -O0 -g -gencode arch=compute_20,code=sm_21 -lineinfo -pg  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



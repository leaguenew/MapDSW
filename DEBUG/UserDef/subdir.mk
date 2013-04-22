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
	nvcc -G -g -O0  -odir "UserDef" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -O0 -g  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



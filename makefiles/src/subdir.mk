################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../FITS.cpp \
../SparseRecovery.cpp \
../PhaseScreen.cpp \
../ImageQualityMetric.cpp \
../ConvexOptimization.cpp \
../Metric.cpp \
../NoiseEstimator.cpp \
../NoiseFilter.cpp \
../Optics.cpp \
../PDMain.cpp \
../ToolBox.cpp \
../SubimageLayout.cpp \
../OpticalSetup.cpp \
../TestRoom.cpp \
../WavefrontSensor.cpp \
../Zernike.cpp 

OBJS += \
./src/FITS.o \
./src/SparseRecovery.o \
./src/PhaseScreen.o \
./src/ImageQualityMetric.o \
./src/ConvexOptimization.o \
./src/Metric.o \
./src/NoiseEstimator.o \
./src/NoiseFilter.o \
./src/Optics.o \
./src/PDMain.o \
./src/ToolBox.o \
./src/SubimageLayout.o \
./src/OpticalSetup.o \
./src/TestRoom.o \
./src/WavefrontSensor.o \
./src/Zernike.o 

CPP_DEPS += \
./src/FITS.d \
./src/SparseRecovery.d \
./src/PhaseScreen.d \
./src/ImageQualityMetric.d \
./src/ConvexOptimization.d \
./src/Metric.d \
./src/NoiseEstimator.d \
./src/NoiseFilter.d \
./src/Optics.d \
./src/PDMain.d \
./src/ToolBox.d \
./src/SubimageLayout.d \
./src/OpticalSetup.d \
./src/TestRoom.d \
./src/WavefrontSensor.d \
./src/Zernike.d 


# default: g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -ggdb3 -std=c++11 -L../lib -I../include -Wall -c -fmessage-length=0 -MMD -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<" -Wl,-rpath=../lib
	@echo 'Finished building: $<'
	@echo ' '

################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../AWMLE.cpp \
../EffectivePixel.cpp \
../FITS.cpp \
../Fusion.cpp \
../ImageQualityMetric.cpp \
../Minimization.cpp \
../Metric.cpp \
../NoiseEstimator.cpp \
../NoiseFilter.cpp \
../OpticalSystem.cpp \
../PDMain.cpp \
../PDTools.cpp \
../SubimageLayout.cpp \
../TelescopeSettings.cpp \
../TestRoom.cpp \
../WavefrontSensor.cpp \
../WaveletTransform.cpp \
../Zernikes.cpp 

OBJS += \
./src/AWMLE.o \
./src/EffectivePixel.o \
./src/FITS.o \
./src/Fusion.o \
./src/ImageQualityMetric.o \
./src/Minimization.o \
./src/Metric.o \
./src/NoiseEstimator.o \
./src/NoiseFilter.o \
./src/OpticalSystem.o \
./src/PDMain.o \
./src/PDTools.o \
./src/SubimageLayout.o \
./src/TelescopeSettings.o \
./src/TestRoom.o \
./src/WavefrontSensor.o \
./src/WaveletTransform.o \
./src/Zernikes.o 

CPP_DEPS += \
./src/AWMLE.d \
./src/EffectivePixel.d \
./src/FITS.d \
./src/Fusion.d \
./src/ImageQualityMetric.d \
./src/Minimization.d \
./src/Metric.d \
./src/NoiseEstimator.d \
./src/NoiseFilter.d \
./src/OpticalSystem.d \
./src/PDMain.d \
./src/PDTools.d \
./src/SubimageLayout.d \
./src/TelescopeSettings.d \
./src/TestRoom.d \
./src/WavefrontSensor.d \
./src/WaveletTransform.d \
./src/Zernikes.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ --param ggc-min-expand=0 --param ggc-min-heapsize=101072 -std=c++11 -fPIC -L../lib -I../include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<" -Wl,-rpath=../lib
	@echo 'Finished building: $<'
	@echo ' '



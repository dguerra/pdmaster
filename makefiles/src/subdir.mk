################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../AWMLE.cpp \
../EffectivePixel.cpp \
../ErrorMetric.cpp \
../FITS.cpp \
../ImageQualityMetric.cpp \
../Minimization.cpp \
../Metric.cpp \
../MatrixEquation.cpp \
../NoiseEstimator.cpp \
../NoiseFilter.cpp \
../OpticalSystem.cpp \
../Optimization.cpp \
../PDPlayground.cpp \
../PDTools.cpp \
../SubimageLayout.cpp \
../TelescopeSettings.cpp \
../TestRoom.cpp \
../GetStep.cpp \
../WavefrontSensor.cpp \
../WaveletTransform.cpp \
../Zernikes.cpp 

OBJS += \
./src/AWMLE.o \
./src/EffectivePixel.o \
./src/ErrorMetric.o \
./src/FITS.o \
./src/ImageQualityMetric.o \
./src/Minimization.o \
./src/Metric.o \
./src/MatrixEquation.o \
./src/NoiseEstimator.o \
./src/NoiseFilter.o \
./src/OpticalSystem.o \
./src/Optimization.o \
./src/PDPlayground.o \
./src/PDTools.o \
./src/SubimageLayout.o \
./src/TelescopeSettings.o \
./src/TestRoom.o \
./src/GetStep.o \
./src/WavefrontSensor.o \
./src/WaveletTransform.o \
./src/Zernikes.o 

CPP_DEPS += \
./src/AWMLE.d \
./src/EffectivePixel.d \
./src/ErrorMetric.d \
./src/FITS.d \
./src/ImageQualityMetric.d \
./src/Minimization.d \
./src/Metric.d \
./src/MatrixEquation.d \
./src/NoiseEstimator.d \
./src/NoiseFilter.d \
./src/OpticalSystem.d \
./src/Optimization.d \
./src/PDPlayground.d \
./src/PDTools.d \
./src/SubimageLayout.d \
./src/TelescopeSettings.d \
./src/TestRoom.d \
./src/GetStep.d \
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



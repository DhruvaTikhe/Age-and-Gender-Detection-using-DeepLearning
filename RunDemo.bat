@echo off
title Age and Gender Detection Demo Menu
cls

echo ==============================================
echo       Age and Gender Detection - Demo
echo ==============================================
echo Choose a backbone model to run demo:
echo.
echo 1. MobileNetV2
echo 2. ResNet18
echo 3. ResNet50
echo 4. VGG16
echo 5. DenseNet121
echo 6. Swin-Tiny
echo.
set /p choice="Enter your choice (1-6): "

set model=
set weights=

if "%choice%"=="1" set model=mobilenet_v2 & set weights=MobileNetV2.pth
if "%choice%"=="2" set model=resnet18   & set weights=ResNet18.pth
if "%choice%"=="3" set model=resnet50   & set weights=ResNet50.pth
if "%choice%"=="4" set model=vgg16      & set weights=VGG16.pth
if "%choice%"=="5" set model=densenet121 & set weights=DenseNet121.pth
if "%choice%"=="6" set model=swin_t     & set weights=Swin_T.pth

if "%model%"=="" (
    echo Invalid choice! Exiting...
    pause
    exit /b
)

echo You selected: %model%
echo.

REM >>> Run demo with selected model <<<
python AgeGenderDetectionDemo.py --backbone %model% --weights %weights%

pause
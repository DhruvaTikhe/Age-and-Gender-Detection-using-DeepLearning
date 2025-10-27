@echo off
title Age and Gender Detection Training Menu
cls

echo ==============================================
echo       Age and Gender Detection Training
echo ==============================================
echo Choose a backbone model to train:
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

if "%choice%"=="1" set model=MobileNetV2
if "%choice%"=="2" set model=ResNet18
if "%choice%"=="3" set model=ResNet50
if "%choice%"=="4" set model=VGG16
if "%choice%"=="5" set model=DenseNet121
if "%choice%"=="6" set model=Swin_T

if "%model%"=="" (
    echo Invalid choice! Exiting...
    pause
    exit /b
)

echo You selected: %model%
echo.

REM >>> Change dataset paths as needed <<<
set CSV=Dataset-AgeGender\Dataset-CSV-AgeGender.csv
set IMG_DIR=Dataset-AgeGender\Dataset-Images-AgeGender

REM >>> Start training <<<
python AgeGenderDetectionTrain.py --csv %CSV% --img_dir %IMG_DIR% --backbone %model%

pause
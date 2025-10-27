# Age-and-Gender-Detection-using-DeepLearning
Age and Gender Detection using Deep Learning techniques with help of PreTrained models

Models: MobileNetV2, ResNet18, ResNet50, DenseNet121, SwinTransformer, VGG16

Hardware: GTX1070 CUDA

Software: Python, Sci-Kit Learn, Pytorch


The project uses 23,703 images (200x200) for training of a CNN model based on a lightweight MobileNetV2 framework. It classifies the person based on age [infant,adult,elder] and gender [male,female]. Based on real-time testing using OpenCV, the model performed with an average accuracy of 92% for gender detection and 98% for age detection.   

Note: The entire working was performed using PyTorch for leveraing the CUDA capabilities of a home grade GTX1070 8gb GPU [Train time: 09 minutes] and exported to a .pth file for later real-time inference. Future scope for the model includes underage driving detection and current use case is limited to automated demographic generation for public spaces and vehicals.

#Train:
python AgeGenderDetection.py --mode train --csv ./Dataset-AgeGender/Dataset-CSV-AgeGender.csv --img_dir ./Dataset-AgeGender/Dataset-Images-AgeGender --weights MobileNetV2.pth

**python filename.py --mode train --csv <csvPath> --img-dir <imgPath> --weights <modelPath>**


#Real-time Demo:
python AgeGenderDetection.py --mode demo --weights MobileNetV2.pth

**python filename.py --mode demo --csv <csvPath> --img-dir <imgPath> --weights <modelPath>**

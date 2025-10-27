import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 96
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_WORKERS = 6

class AgeGenderDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        gender = int(self.data.iloc[idx, 2])
        age = int(self.data.iloc[idx, 1])

        if age <= 5:
            age_group = 0
        elif age <= 59:
            age_group = 1
        else:
            age_group = 2

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(age_group, dtype=torch.long), torch.tensor(gender, dtype=torch.long)

# Model factory
def build_model(backbone="mobilenet_v2", num_age_classes=3, num_gender_classes=2):
    backbone = backbone.lower()

    if backbone == "mobilenetv2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = m.classifier[1].in_features
        m.classifier = nn.Identity()
        model_name = "MobileNetV2"
    elif backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = m.fc.in_features
        m.fc = nn.Identity()
        model_name = "ResNet18"
    elif backbone == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = m.fc.in_features
        m.fc = nn.Identity()
        model_name = "ResNet50"
    elif backbone == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Identity()
        model_name = "VGG16"
    elif backbone == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_features = m.classifier.in_features
        m.classifier = nn.Identity()
        model_name = "DenseNet121"
    elif backbone == "swin_t":
        m = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        in_features = m.head.in_features
        m.head = nn.Identity()
        model_name = "SwinT"
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    class MultiHead(nn.Module):
        def __init__(self, backbone, in_features):
            super().__init__()
            self.backbone = backbone
            self.fc_age = nn.Linear(in_features, num_age_classes)
            self.fc_gender = nn.Linear(in_features, num_gender_classes)
        def forward(self, x):
            feats = self.backbone(x)
            if feats.ndim > 2:
                feats = torch.flatten(feats, 1)
            return self.fc_age(feats), self.fc_gender(feats)

    return MultiHead(m, in_features), model_name

def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct_age, correct_gender, correct_both, total = 0, 0, 0, 0

    with torch.no_grad():
        for images, age_labels, gender_labels in dataloader:
            images = images.to(device)
            age_labels = age_labels.to(device)
            gender_labels = gender_labels.to(device)

            age_out, gender_out = model(images)
            loss_age = criterion(age_out, age_labels)
            loss_gender = criterion(gender_out, gender_labels)
            loss = loss_age + loss_gender
            running_loss += loss.item()

            age_preds = torch.argmax(age_out, dim=1)
            gender_preds = torch.argmax(gender_out, dim=1)

            correct_age += (age_preds == age_labels).sum().item()
            correct_gender += (gender_preds == gender_labels).sum().item()
            correct_both += ((age_preds == age_labels) & (gender_preds == gender_labels)).sum().item()
            total += age_labels.size(0)

    return (running_loss / len(dataloader),
            correct_age / total,
            correct_gender / total,
            correct_both / total)

def train_model(csv_file, img_dir, backbone, save_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = AgeGenderDataset(csv_file, img_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model, model_name = build_model(backbone)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_model_path = save_path if save_path else f"{model_name}.pth"

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, age_labels, gender_labels in train_loader:
            images = images.to(device)
            age_labels = age_labels.to(device)
            gender_labels = gender_labels.to(device)

            optimizer.zero_grad()
            age_out, gender_out = model(images)
            loss_age = criterion(age_out, age_labels)
            loss_gender = criterion(gender_out, gender_labels)
            loss = loss_age + loss_gender
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss, val_age_acc, val_gender_acc, val_combined_acc = evaluate_model(model, val_loader, criterion)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, Age Acc: {val_age_acc:.2%}, "
              f"Gender Acc: {val_gender_acc:.2%}, Combined Acc: {val_combined_acc:.2%}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model updated and saved to {best_model_path}")

if __name__ == "__main__":
    print(f"✅ Using {device.type.upper()}:",
        torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="mobilenet_v2")
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args()

    train_model(args.csv, args.img_dir, args.backbone, args.weights)
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 96

def build_model(backbone="mobilenet_v2", num_age_classes=3, num_gender_classes=2):
    backbone = backbone.lower()

    if backbone == "mobilenet_v2":
        m = models.mobilenet_v2(weights=None)
        in_features = m.classifier[1].in_features
        m.classifier = nn.Identity()
        model_name = "MobileNetV2"
    elif backbone == "resnet18":
        m = models.resnet18(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Identity()
        model_name = "ResNet18"
    elif backbone == "resnet50":
        m = models.resnet50(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Identity()
        model_name = "ResNet50"
    elif backbone == "vgg16":
        m = models.vgg16(weights=None)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Identity()
        model_name = "VGG16"
    elif backbone == "densenet121":
        m = models.densenet121(weights=None)
        in_features = m.classifier.in_features
        m.classifier = nn.Identity()
        model_name = "DenseNet121"
    elif backbone == "swin_t":
        m = models.swin_t(weights=None)
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

def run_demo(model_path, backbone):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model, model_name = build_model(backbone)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                img_t = transform(pil_img).unsqueeze(0).to(device)

                age_out, gender_out = model(img_t)
                age_pred = torch.argmax(age_out, dim=1).item()
                gender_pred = torch.argmax(gender_out, dim=1).item()

                age_str = ["Infant", "Adult", "Elder"][age_pred]
                gender_str = ["Male", "Female"][gender_pred]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, f"{gender_str}, {age_str}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

            cv2.imshow("Age & Gender Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print(f"✅ Using {device.type.upper()}:",
        torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="mobilenet_v2")
    args = parser.parse_args()

    run_demo(args.weights, args.backbone)
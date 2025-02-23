import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, densenet201,
    resnet152, mobilenet_v3_large
)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, channels, height, width = x.size()
        query = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, height * width)
        value = self.value(x).view(batch, -1, height * width).permute(0, 2, 1)

        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        out = torch.bmm(attention, value).permute(0, 2, 1).view(batch, channels, height, width)
        return self.gamma * out + x


class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes, selfsa=False, dropout_prob=0.5):
        super(EfficientNetB0Classifier, self).__init__()
        self.backbone = efficientnet_b0(weights="IMAGENET1K_V1")
        
        # Lấy in_features từ classifier ban đầu trước khi thay thế
        in_features = self.backbone.classifier[1].in_features
        
        # Thay thế classifier gốc bằng Identity
        self.backbone.classifier[1] = nn.Identity()

        # Đóng băng các lớp trước (feature extraction)
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        # Tùy chọn Self-Attention
        self.attention = SelfAttention(self.backbone.features[-1][0].out_channels) if selfsa else None
        
        # Thêm dropout và classifier mới
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        if self.attention:
            x = self.attention(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class DenseNet201Classifier(nn.Module):
    def __init__(self, num_classes, selfsa=False, dropout_prob=0.5):
        super(DenseNet201Classifier, self).__init__()
        self.backbone = densenet201(weights="IMAGENET1K_V1")
        self.backbone.classifier = nn.Identity()  # Replace the original classifier

        # Đóng băng các lớp trước
        for param in self.backbone.features.parameters():
           param.requires_grad = False
        
        self.dropout = nn.Dropout(p=dropout_prob)
        self.attention = SelfAttention(self.backbone.features[-1].out_channels) if selfsa else None
        self.fc = nn.Linear(self.backbone.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        if self.attention:
            x = self.attention(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
        
        
class ResNet152Classifier(nn.Module):
    def __init__(self, num_classes, selfsa=False, dropout_prob=0.5):
        super(ResNet152Classifier, self).__init__()
        self.model = resnet152(pretrained=True)
        
        # Đóng băng các lớp trước
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze lớp fc
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # Add dropout after the fully connected layer
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # If selfsa is True, add self-attention
        if selfsa:
            self.attention = SelfAttention(self.model.layer4[-1].conv2.out_channels)
        else:
            self.attention = None

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)  # Apply dropout
        if self.attention:
            x = self.attention(x)  # Apply self-attention if exists
        return x

class MobileNetV3LargeClassifier(nn.Module):
    def __init__(self, num_classes, selfsa=False, dropout_prob=0.5):
        super(MobileNetV3LargeClassifier, self).__init__()
        self.model = mobilenet_v3_large(pretrained=True)
        self.model.classifier[0] = nn.Linear(self.model.classifier[0].in_features, num_classes)

        # Đóng băng các lớp trước
        for param in self.model.features.parameters():
            param.requires_grad = False


        # Add dropout after the fully connected layer
        self.dropout = nn.Dropout(p=dropout_prob)

        # If selfsa is True, add self-attention
        if selfsa:
            self.attention = SelfAttention(self.model.features[-1].out_channels)
        else:
            self.attention = None

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)  # Apply dropout
        if self.attention:
            x = self.attention(x)  # Apply self-attention if exists
        return x

    
def get_model(args):
    dropout_prob=0.2
    if args.model == "efficientnetb0":
        return EfficientNetB0Classifier(num_classes=3, selfsa=args.selfsa,dropout_prob=dropout_prob)
    elif args.model == "densenet201":
        return DenseNet201Classifier(num_classes=3, selfsa=args.selfsa,dropout_prob=dropout_prob)
    elif args.model == "resnet152":
        return ResNet152Classifier(num_classes=3, selfsa=args.selfsa,dropout_prob=dropout_prob)
    elif args.model == "mobilenetv3":
        return MobileNetV3LargeClassifier(num_classes=3, selfsa=args.selfsa,dropout_prob=dropout_prob)
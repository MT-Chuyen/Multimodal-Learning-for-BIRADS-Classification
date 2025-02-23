import os
import cv2
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
# Custom dataset class for mammography data
class MammographyDataset(Dataset):
    def __init__(self, df_path, img_dir, target_size=(256, 256), transform=None):
        self.df = pd.read_csv(df_path)
        self.img_dir = img_dir
        self.transform = transform
        self.target_size = target_size
        self.images, self.labels = self.load_data()

    def load_data(self):
        images, labels = [], []
        le = LabelEncoder()  # LabelEncoder for 'breast_birads'
        
        print("Loading data with progress bar:")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing images"):
            img_path = os.path.join(self.img_dir, row['image_id'] + '.png')
            img = cv2.imread(img_path)

            # Resize image if target_size is specified
            if self.target_size:
                img = cv2.resize(img, self.target_size)

            if self.transform:
                img = self.transform(image=img)['image']  # Use albumentations transform
            
            images.append(img)
            labels.append(row['breast_birads'])  # Add breast_birads label

        labels = le.fit_transform(labels)  # Encode breast_birads
        print(f"LabelEncoder classes: {le.classes_}")

        labels = torch.tensor(labels).long()  # Convert labels to tensor
        
        return images, labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def data_loader(batch_size=32):
    label_folder_dir  = '/media/mountHDD3/data_storage/biomedical_data/Dataset/VinDr-Mammo/Processed/'
    main_img_dir = '/media/mountHDD3/data_storage/biomedical_data/Dataset/VinDr-Mammo-PNG-MAMMO-CLIP-CROPPED-TEXT_REMOVE-FLIP-MEDIAN-CLAHE-UNSHARP'
    train_label = label_folder_dir + 'balanced_train_df_3_classes.csv'
    val_label = label_folder_dir +  'balanced_val_df_3_classes.csv'
    test_label = label_folder_dir +  'balanced_test_df_3_classes.csv'
    target_size = (256,256)

    # Define image transformations
    train_transform = A.Compose([
        A.Resize(target_size[0], target_size[1]),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
 
    # Transforms cho validation/test (không augmentation)
    val_test_transform = A.Compose([
        A.Resize(target_size[0], target_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Tạo datasets
    train_dataset = MammographyDataset(train_label, main_img_dir, target_size, transform=train_transform)
    val_dataset = MammographyDataset(val_label, main_img_dir, target_size, transform=val_test_transform)
    test_dataset = MammographyDataset(test_label, main_img_dir, target_size, transform=val_test_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

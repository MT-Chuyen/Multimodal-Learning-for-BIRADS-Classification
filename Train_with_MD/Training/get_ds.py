import os
import cv2
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from tqdm import tqdm
# Custom dataset class for mammography data
class MammographyDataset(Dataset):
    def __init__(self, df_path, img_dir, target_size, transform=None):
        self.df = pd.read_csv(df_path)
        self.img_dir = img_dir
        self.target_size = target_size
        self.transform = transform
        self.images, self.metadata, self.labels = self.load_data()

    def load_data(self):
        images, labels = [], []
        age, density = [], []
        
        le_birads = LabelEncoder()  # LabelEncoder for 'breast_birads'
        le_density = LabelEncoder()  # LabelEncoder for 'breast_density'

        print("Loading data with progress bar:")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing images"):
        # for _, row in self.df.iterrows():
            img_path = os.path.join(self.img_dir, row['image_id']+'.png')
            img = cv2.imread(img_path)
            if img is not None:  # Check if the image is read correctly
                if self.transform:
                    img = self.transform(img)
                images.append(img)
                labels.append(row['breast_birads'])  # Add breast_birads label
                age.append(float(row['age']))  # Add age
                density.append(row['breast_density'])  # Add breast density
            else:
                print(f"Warning: Image at {img_path} could not be loaded.")

        labels = le_birads.fit_transform(labels)  # Encode breast_birads
        labels = torch.tensor(labels).long()  # Convert labels to tensor
        
        density_encoded = le_density.fit_transform(density)  # Encode breast_density
        density_tensor = torch.tensor(density_encoded).float()  # Convert encoded density to tensor
        
        metadata = {
            'age': torch.tensor(age).float(),  # Convert age to tensor
            'breast_density': density_tensor,  # Use encoded breast_density tensor
        }
        return images, metadata, labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.images[idx], self.metadata['age'][idx], self.metadata['breast_density'][idx], self.labels[idx]

def data_loader(batch_size=32):
 
    main_img_dir = '/media/mountHDD3/data_storage/biomedical_data/Dataset/VinDr-Mammo-PNG-MAMMO-CLIP-CROPPED-TEXT_REMOVE-FLIP-MEDIAN-CLAHE-UNSHARP'
    train_label = '/media/mountHDD3/data_storage/biomedical_data/Dataset/VinDr-Mammo/Processed/balanced_train_df_3_classes.csv'
    val_label = '/media/mountHDD3/data_storage/biomedical_data/Dataset/VinDr-Mammo/Processed/balanced_val_df_3_classes.csv'
    test_label = '/media/mountHDD3/data_storage/biomedical_data/Dataset/VinDr-Mammo/Processed/balanced_test_df_3_classes.csv'
    target_size = (256,256)

    # Define image transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
 
    # Create datasets
    train_dataset = MammographyDataset(train_label, main_img_dir, target_size, transform=transform)
    val_dataset = MammographyDataset(val_label, main_img_dir, target_size, transform=transform)
    test_dataset = MammographyDataset(test_label, main_img_dir, target_size, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

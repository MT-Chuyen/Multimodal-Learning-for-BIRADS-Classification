import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import get_model
from get_ds import data_loader

def tester(args):
    model = get_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.load_state_dict(torch.load(f'/media/mountHDD3/data_storage/biomedical_data/Dataset/VinDr-Mammo/Processed/Train_AD_weight/{args.model}{args.selfsa}{args.loss}_weights.pth'))
    model.eval()

    print("Model loaded successfully!")
    
    _, _, test_loader = data_loader()

    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    test_true, test_pred = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images,labels = batch
            images,labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted_labels = torch.max(outputs, 1)
            test_true.extend(labels.cpu().numpy())
            test_pred.extend(predicted_labels.cpu().numpy())

    test_acc = accuracy_score(test_true, test_pred)
    test_loss /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    conf_matrix = confusion_matrix(test_true, test_pred)
    metrics = calculate_metrics(conf_matrix)

    results_df = pd.DataFrame({
        'Class': [f'C{i+1}' for i in range(len(conf_matrix))],
        'Sensitivity (%)': metrics['sensitivity'],
        'Specificity (%)': metrics['specificity'],
        'Precision (%)': metrics['precision'],
        'Accuracy (%)': metrics['accuracy']
    })

    print(results_df)
    output_folder = f'/media/mountHDD2/chuyenmt/BrEaST/Mammo/Train_with_AD/Train_VSCode_AD/Output/output_{args.model}{args.selfsa}{args.loss}'
    os.makedirs(output_folder, exist_ok=True)

    plot_confusion_matrix(conf_matrix, test_acc, args)

 

    results_path = os.path.join(output_folder, 'results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

def calculate_metrics(conf_matrix):
    sensitivity, specificity, precision, accuracy = [], [], [], []
    
    for i in range(len(conf_matrix)):
        TP = conf_matrix[i, i]
        FN = np.sum(conf_matrix[i, :]) - TP
        FP = np.sum(conf_matrix[:, i]) - TP
        TN = np.sum(conf_matrix) - (TP + FN + FP)

        sen = TP / (TP + FN) if (TP + FN) != 0 else 0
        spe = TN / (TN + FP) if (TN + FP) != 0 else 0
        ppr = TP / (TP + FP) if (TP + FP) != 0 else 0
        acc = TP / (TP + FN) if (TP + FN) != 0 else 0


        sensitivity.append(sen * 100)
        specificity.append(spe * 100)
        precision.append(ppr * 100)
        accuracy.append(acc * 100)

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'accuracy': accuracy
    }

def plot_confusion_matrix(conf_matrix, test_acc, args):
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(f'Confusion Matrix - Test Accuracy: {test_acc:.4f}')
    plt.show()
    plt.savefig(f'/media/mountHDD2/chuyenmt/BrEaST/Mammo/Train_with_AD/Train_VSCode_AD/Output/output_{args.model}{args.selfsa}{args.loss}/confusion_matrix.png')
    plt.close()

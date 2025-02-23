import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import wandb   
from model import get_model
from get_ds import data_loader
from loss import get_loss_function

def trainer(args):  
    wandb.init(
        project="VinDr-Mammo_classification", 
        config={
            "learning_rate": 1e-5,
            "epochs": args.epoch,
        },
        tags=[args.model]   
    )

    model = get_model(args)  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader, val_loader, _ = data_loader()
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    criterion = get_loss_function(loss_type= args.loss ,device=device)
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)   
    num_epochs = wandb.config.epochs

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_true, train_pred = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images,labels = batch
            images,labels = (
                images.to(device),
                labels.to(device)
            )

            outputs = model(images)
            loss = criterion(outputs, labels)   
            _, predicted_labels = torch.max(outputs, 1)
            train_true.extend(labels.cpu().numpy())
            train_pred.extend(predicted_labels.cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_acc = accuracy_score(train_true, train_pred)
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc)
        wandb.log({
            "train/train_loss": train_loss / len(train_loader),
            "train/accuracy": train_acc,
            "train/epoch": epoch + 1,
        })

        model.eval()
        val_loss = 0
        val_true, val_pred = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                images,labels = batch
                images,labels = (
                    images.to(device),
                    labels.to(device)
                )

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted_labels = torch.max(outputs, 1)
                val_true.extend(labels.cpu().numpy())
                val_pred.extend(predicted_labels.cpu().numpy())

        val_acc = accuracy_score(val_true, val_pred)
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)

        wandb.log({
            "val/val_loss": val_loss / len(val_loader),
            "val/accuracy": val_acc,
            "val/epoch": epoch + 1,
        })

    torch.save(model.state_dict(), f'/media/mountHDD3/data_storage/biomedical_data/Dataset/VinDr-Mammo/Processed/Train_AD_weight/{args.model}{args.selfsa}{args.loss}_weights.pth')
    wandb.finish()

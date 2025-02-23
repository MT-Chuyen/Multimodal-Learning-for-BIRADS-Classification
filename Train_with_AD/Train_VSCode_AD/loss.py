import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, num_classes=5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.alpha = alpha if alpha is not None else torch.ones(num_classes)  # Mặc định là 1 cho tất cả các lớp

    def forward(self, inputs, targets):
        # inputs: (batch_size, num_classes) - raw logits
        # targets: (batch_size,) - integer labels
        
        # Convert logits to probabilities
        probs = F.softmax(inputs, dim=1)  # (batch_size, num_classes)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()  # (batch_size, num_classes)
        
        # Compute focal loss
        pt = (probs * targets_one_hot).sum(dim=1)  # Probability of true class (batch_size,)
        log_pt = torch.log(pt + 1e-9)  # Avoid log(0)
        focal_loss = -(1 - pt) ** self.gamma * log_pt  # Focal loss core formula
        
        # Apply alpha (class weights)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # Get alpha for each target class
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()  # Average loss across the batch



def get_loss_function(loss_type="cross_entropy", num_classes=3, gamma=2,device =None):
    """
    Lựa chọn hàm loss.
    loss_type: "cross_entropy" hoặc "focal_loss"
    num_classes: số lượng lớp
    gamma: tham số Focal Loss
    """
    # Class weights (đảm bảo tính chính xác)
    # class_weights = torch.tensor([0.352, 0.703, 3.262, 3.367, 7.772], dtype=torch.float32).to(device)
    class_weights = torch.tensor([ 2.1, 2.1, 16.9], dtype=torch.float32).to(device)


    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == "focal_loss":
        return FocalLoss(gamma=gamma, alpha=class_weights, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'cross_entropy' or 'focal_loss'.")

   
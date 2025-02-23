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


class HierarchicalCrossEntropyLoss(nn.Module):
    def __init__(self, device, num_classes=5):
        super(HierarchicalCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.class_weights = torch.tensor([0.352, 0.703, 3.262, 3.367, 7.772], dtype=torch.float32).to(device)

        # Define a matrix of weights, where closer classes have smaller weight
        self.distance_matrix = torch.tensor([
            [1.0, 0.8, 0.5, 0.3, 0.1],
            [0.8, 1.0, 0.7, 0.5, 0.3],
            [0.5, 0.7, 1.0, 0.7, 0.5],
            [0.3, 0.5, 0.7, 1.0, 0.8],
            [0.1, 0.3, 0.5, 0.8, 1.0]
        ], dtype=torch.float32).to(device)


    def forward(self, inputs, targets):
        batch_size = targets.size(0)

        # Get cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')

        # Generate weights based on the correct labels
        correct_class_indices = targets
        weights = self.distance_matrix[correct_class_indices, :]

        # Get predicted classes with torch.argmax
        _, predicted_classes = torch.max(F.softmax(inputs, dim=1), dim=1)
        
        # Use the predicted class to get the weights
        predicted_weights = weights[torch.arange(batch_size),predicted_classes]

        # Use the weights to scale the original cross entropy loss
        final_loss = (ce_loss * predicted_weights).mean()

        return final_loss

class AttentionEntropyLoss(nn.Module):
    def __init__(self, loss_type="cross_entropy", num_classes=3, gamma=2, lambda_entropy = 0.1, device=None):
        super(AttentionEntropyLoss, self).__init__()
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.gamma = gamma
        self.lambda_entropy = lambda_entropy
        # Class weights (đảm bảo tính chính xác)
        self.class_weights = torch.tensor([ 2.1, 2.1, 16.92], dtype=torch.float32).to(device)
    
    def forward(self, inputs, targets, attn_weights_img, attn_weights_meta):
        if self.loss_type == "cross_entropy":
            main_loss = F.cross_entropy(inputs, targets, weight=self.class_weights)
        elif self.loss_type == "focal_loss":
            probs = F.softmax(inputs, dim=1)
            targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
            pt = (probs * targets_one_hot).sum(dim=1)
            log_pt = torch.log(pt + 1e-9)
            focal_loss = -(1 - pt) ** self.gamma * log_pt
            alpha_t = self.class_weights[targets]
            main_loss =  (alpha_t * focal_loss).mean()
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}. Choose 'cross_entropy' or 'focal_loss'.")
         
        # tính entropy của attention
        entropy_img = - (attn_weights_img * torch.log(attn_weights_img + 1e-9)).sum(dim=(-2, -1)).mean()
        entropy_meta = - (attn_weights_meta * torch.log(attn_weights_meta + 1e-9)).sum(dim=(-2, -1)).mean()
        # Tính tổng loss
        total_loss = main_loss + self.lambda_entropy * (entropy_img + entropy_meta)
        return total_loss

class WeightedMultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, device=None, loss_type = "cross_entropy", num_classes = 3, gamma = 2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.gamma = gamma
        self.class_weights = torch.tensor([ 2.1, 2.1, 16.92], dtype=torch.float32).to(device)

    def forward(self, outputs_image, outputs_metadata, targets):
         
        if self.loss_type == "cross_entropy":
            loss_image = F.cross_entropy(outputs_image, targets, weight=self.class_weights)
            loss_metadata = F.cross_entropy(outputs_metadata, targets, weight=self.class_weights)
        elif self.loss_type == "focal_loss":
            probs_img = F.softmax(outputs_image, dim=1)
            probs_meta = F.softmax(outputs_metadata, dim=1)
            targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
            pt_img = (probs_img * targets_one_hot).sum(dim=1)
            log_pt_img = torch.log(pt_img + 1e-9)
            focal_loss_img = -(1 - pt_img) ** self.gamma * log_pt_img
            alpha_t_img = self.class_weights[targets]
            loss_image =  (alpha_t_img * focal_loss_img).mean()
            
            pt_meta = (probs_meta * targets_one_hot).sum(dim=1)
            log_pt_meta = torch.log(pt_meta + 1e-9)
            focal_loss_meta = -(1 - pt_meta) ** self.gamma * log_pt_meta
            alpha_t_meta = self.class_weights[targets]
            loss_metadata =  (alpha_t_meta * focal_loss_meta).mean()
        else:
             raise ValueError(f"Unsupported loss_type: {self.loss_type}. Choose 'cross_entropy' or 'focal_loss'.")
        return self.alpha * loss_image + self.beta * loss_metadata


def get_loss_function(loss_type="cross_entropy", num_classes=3, gamma=2, lambda_entropy = 0.1, alpha = 0.5, beta=0.5, device =None):
    """
    Lựa chọn hàm loss.
    loss_type: "cross_entropy" hoặc "focal_loss"
    num_classes: số lượng lớp
    gamma: tham số Focal Loss
    """
    class_weights = torch.tensor([ 2.1, 2.1, 16.9], dtype=torch.float32).to(device)


    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == "focal_loss":
        return FocalLoss(gamma=gamma, alpha=class_weights, num_classes=num_classes)
    elif loss_type == "hierarchical":
        return HierarchicalCrossEntropyLoss(device, num_classes=num_classes)
    elif loss_type == "attention_entropy":
       return AttentionEntropyLoss(loss_type, num_classes, gamma, lambda_entropy, device)
    elif loss_type == "weighted_multitask":
        return WeightedMultiTaskLoss(alpha, beta, device, loss_type, num_classes, gamma)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'cross_entropy' or 'focal_loss'.")
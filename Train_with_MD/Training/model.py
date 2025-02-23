# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import efficientnet_b0

# class SelfAttention(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_heads=4):
#         super(SelfAttention, self).__init__()
#         self.num_heads = num_heads
#         self.hidden_dim = hidden_dim
#         self.head_dim = hidden_dim // num_heads
        
#         self.query = nn.Linear(input_dim, hidden_dim)
#         self.key = nn.Linear(input_dim, hidden_dim)
#         self.value = nn.Linear(input_dim, hidden_dim)

#         self.out_proj = nn.Linear(hidden_dim, input_dim)

#         self.softmax = nn.Softmax(dim=-1)

#     def split_heads(self, x):
#         batch_size = x.size(0)
#         return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

#     def combine_heads(self, x):
#         batch_size = x.size(0)
#         return x.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)

#     def forward(self, x):
#         Q = self.split_heads(self.query(x))
#         K = self.split_heads(self.key(x))
#         V = self.split_heads(self.value(x))

#         # Attention scores
#         attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
#         attention_weights = self.softmax(attention_scores)

#         # Apply attention to values
#         output = torch.matmul(attention_weights, V)
#         output = self.combine_heads(output)
#         output = self.out_proj(output)
#         return output



# class CrossModalityAttention(nn.Module):
#     def __init__(self, img_dim, meta_dim, hidden_dim, num_heads=4):
#         super(CrossModalityAttention, self).__init__()
#         self.num_heads = num_heads
#         self.hidden_dim = hidden_dim
#         self.head_dim = hidden_dim // num_heads

#         self.query_fc_img = nn.Linear(img_dim, hidden_dim)
#         self.key_fc_img = nn.Linear(img_dim, hidden_dim)
#         self.value_fc_img = nn.Linear(img_dim, hidden_dim)

#         self.query_fc_meta = nn.Linear(meta_dim, hidden_dim)
#         self.key_fc_meta = nn.Linear(meta_dim, hidden_dim)
#         self.value_fc_meta = nn.Linear(meta_dim, hidden_dim)

#         self.out_proj_img = nn.Linear(hidden_dim, img_dim)
#         self.out_proj_meta = nn.Linear(hidden_dim, meta_dim)
       
#         self.softmax = nn.Softmax(dim=-1)
    
#     def split_heads(self, x):
#         batch_size = x.size(0)
#         return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

#     def combine_heads(self, x):
#         batch_size = x.size(0)
#         return x.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
    
#     def forward(self, img_features, meta_features):
            
#         batch_size = img_features.size(0)
#         # Split heads
#         Q_img = self.split_heads(self.query_fc_img(img_features))
#         K_img = self.split_heads(self.key_fc_img(img_features))
#         V_img = self.split_heads(self.value_fc_img(img_features))

#         Q_meta = self.split_heads(self.query_fc_meta(meta_features))
#         K_meta = self.split_heads(self.key_fc_meta(meta_features))
#         V_meta = self.split_heads(self.value_fc_meta(meta_features))

#         # Cross attention between image and metadata
#         attn_scores_img = torch.matmul(Q_meta, K_img.transpose(-2, -1)) / (self.head_dim ** 0.5)
#         attn_weights_img = self.softmax(attn_scores_img)
#         out_img = torch.matmul(attn_weights_img, V_img)

#         attn_scores_meta = torch.matmul(Q_img, K_meta.transpose(-2, -1)) / (self.head_dim ** 0.5)
#         attn_weights_meta = self.softmax(attn_scores_meta)
#         out_meta = torch.matmul(attn_weights_meta, V_meta)

#         out_img = self.combine_heads(out_img)
#         out_img = self.out_proj_img(out_img)

#         out_meta = self.combine_heads(out_meta)
#         out_meta = self.out_proj_meta(out_meta)

#          # Concatenate the output from both modalities and change dim
#         combined_output = torch.cat([out_img.unsqueeze(1), out_meta.unsqueeze(1)], dim=1)
#         combined_output = combined_output.view(batch_size, -1)
        
#         return combined_output


# class MultimodalModel_CrossAttn(nn.Module):
#     def __init__(self, num_heads=4, hidden_dim=512, self_meta=False, self_image=False):
#         super(MultimodalModel_CrossAttn, self).__init__()

#         # Image model using EfficientNetB0
#         self.image_model = efficientnet_b0(weights="IMAGENET1K_V1")
        
#         # Lấy in_features từ classifier ban đầu trước khi thay thế
#         in_features = self.image_model.classifier[1].in_features
        
#         # Thay thế classifier gốc bằng Identity
#         self.image_model.classifier[1] = nn.Identity()
        
#         # Freeze all layers of the EfficientNetB0 except the last classifier
#         for param in self.image_model.features.parameters():
#             param.requires_grad = False
            
#         # Metadata processing layers
#         self.age_norm = nn.BatchNorm1d(1)  # Normalize age
#         self.breast_density_emb = nn.Embedding(4, 4)  # Assuming 4 unique breast density categories

#         # Fully connected layers for image and metadata
#         self.fc_image = nn.Linear(in_features, 512) # Use the correct in_features
#         self.fc_metadata = nn.Linear(1 + 4, 512)  # 1 age + 4 breast_density

#         # Intra-Modality Self-Attention blocks (conditional)
#         self.self_attention_image = SelfAttention(input_dim=512, hidden_dim=hidden_dim, num_heads=num_heads) if self_image else None
#         self.self_attention_metadata = SelfAttention(input_dim=512, hidden_dim=hidden_dim, num_heads=num_heads) if self_meta else None
      

#         # Inter-Modality Cross-Attention block
#         self.cross_attention = CrossModalityAttention(img_dim=512, meta_dim=512, hidden_dim=hidden_dim, num_heads=num_heads)
        
       
#         # Final classification layer
#         self.fc_combined = nn.Linear(512 * 2, 5)  # Combine image and metadata features, số lớp là 5, input là 512 * 2

#     def forward(self, x_image, age, breast_density):
#         # Process image
#         x_image = self.image_model(x_image)
#         x_image = F.relu(self.fc_image(x_image))
#         # Process metadata
#         age_norm = self.age_norm(age.view(-1, 1))
#         breast_density = breast_density.long()
#         breast_density_emb = self.breast_density_emb(breast_density)
#         metadata = torch.cat([age_norm, breast_density_emb], dim=1)
#         x_metadata = F.relu(self.fc_metadata(metadata))
        
#         # Intra-modality Self-Attention (conditional)
#         if self.self_attention_image:
#             x_image = self.self_attention_image(x_image)
#         if self.self_attention_metadata:
#            x_metadata = self.self_attention_metadata(x_metadata)
#         # Inter-modality cross-attention
#         combined = self.cross_attention(x_image, x_metadata)

#         # Final classification layer
#         output = self.fc_combined(combined)
        
#         return output

# def get_model(args):
#     if args.model == "cross_attn":
#         return MultimodalModel_CrossAttn(self_meta = args.self_meta, self_image = args.self_image)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, input_dim)

        self.softmax = nn.Softmax(dim=-1)

    def split_heads(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        batch_size = x.size(0)
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)

    def forward(self, x):
        Q = self.split_heads(self.query(x))
        K = self.split_heads(self.key(x))
        V = self.split_heads(self.value(x))

        # Attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = self.softmax(attention_scores)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        output = self.combine_heads(output)
        output = self.out_proj(output)
        return output



class CrossModalityAttention(nn.Module):
    def __init__(self, img_dim, meta_dim, hidden_dim, num_heads=4):
        super(CrossModalityAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.query_fc_img = nn.Linear(img_dim, hidden_dim)
        self.key_fc_img = nn.Linear(img_dim, hidden_dim)
        self.value_fc_img = nn.Linear(img_dim, hidden_dim)

        self.query_fc_meta = nn.Linear(meta_dim, hidden_dim)
        self.key_fc_meta = nn.Linear(meta_dim, hidden_dim)
        self.value_fc_meta = nn.Linear(meta_dim, hidden_dim)

        self.out_proj_img = nn.Linear(hidden_dim, img_dim)
        self.out_proj_meta = nn.Linear(hidden_dim, meta_dim)
       
        self.softmax = nn.Softmax(dim=-1)
    
    def split_heads(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        batch_size = x.size(0)
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
    
    def forward(self, img_features, meta_features):
            
        batch_size = img_features.size(0)
        # Split heads
        Q_img = self.split_heads(self.query_fc_img(img_features))
        K_img = self.split_heads(self.key_fc_img(img_features))
        V_img = self.split_heads(self.value_fc_img(img_features))

        Q_meta = self.split_heads(self.query_fc_meta(meta_features))
        K_meta = self.split_heads(self.key_fc_meta(meta_features))
        V_meta = self.split_heads(self.value_fc_meta(meta_features))

        # Cross attention between image and metadata
        attn_scores_img = torch.matmul(Q_meta, K_img.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights_img = self.softmax(attn_scores_img)
        out_img = torch.matmul(attn_weights_img, V_img)

        attn_scores_meta = torch.matmul(Q_img, K_meta.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights_meta = self.softmax(attn_scores_meta)
        out_meta = torch.matmul(attn_weights_meta, V_meta)

        out_img = self.combine_heads(out_img)
        out_img = self.out_proj_img(out_img)

        out_meta = self.combine_heads(out_meta)
        out_meta = self.out_proj_meta(out_meta)

         # Concatenate the output from both modalities and change dim
        combined_output = torch.cat([out_img.unsqueeze(1), out_meta.unsqueeze(1)], dim=1)
        combined_output = combined_output.view(batch_size, -1)
        
        return combined_output


class MultimodalModel_CrossAttn(nn.Module):
    def __init__(self, num_heads=4, hidden_dim=512, self_meta=False, self_image=False, cross_attn = True, num_classes=3):
        super(MultimodalModel_CrossAttn, self).__init__()
        self.cross_attn = cross_attn
        self.num_classes = num_classes

        # Image model using EfficientNetB0
        self.image_model = efficientnet_b0(weights="IMAGENET1K_V1")
        
        # Lấy in_features từ classifier ban đầu trước khi thay thế
        in_features = self.image_model.classifier[1].in_features
        
        # Thay thế classifier gốc bằng Identity
        self.image_model.classifier[1] = nn.Identity()
        
        # Freeze all layers of the EfficientNetB0 except the last classifier
        for param in self.image_model.features.parameters():
            param.requires_grad = False
            
        # Metadata processing layers
        self.age_norm = nn.BatchNorm1d(1)  # Normalize age
        self.breast_density_emb = nn.Embedding(4, 4)  # Assuming 4 unique breast density categories

        # Fully connected layers for image and metadata
        self.fc_image = nn.Linear(in_features, 512) # Use the correct in_features
        self.fc_metadata = nn.Linear(1 + 4, 512)  # 1 age + 4 breast_density

        # Intra-Modality Self-Attention blocks (conditional)
        self.self_attention_image = SelfAttention(input_dim=512, hidden_dim=hidden_dim, num_heads=num_heads) if self_image else None
        self.self_attention_metadata = SelfAttention(input_dim=512, hidden_dim=hidden_dim, num_heads=num_heads) if self_meta else None
      

        # Inter-Modality Cross-Attention block
        self.cross_attention = CrossModalityAttention(img_dim=512, meta_dim=512, hidden_dim=hidden_dim, num_heads=num_heads)
        
       
        # Final classification layer
        self.fc_combined = nn.Linear(512 * 2, self.num_classes) if self.cross_attn else nn.Linear(512 + 512, self.num_classes) # Combine image and metadata features, số lớp là num_classes

    def forward(self, x_image, age, breast_density):
        # Process image
        x_image = self.image_model(x_image)
        x_image = F.relu(self.fc_image(x_image))
        # Process metadata
        age_norm = self.age_norm(age.view(-1, 1))
        breast_density = breast_density.long()
        breast_density_emb = self.breast_density_emb(breast_density)
        metadata = torch.cat([age_norm, breast_density_emb], dim=1)
        x_metadata = F.relu(self.fc_metadata(metadata))
        
        # Intra-modality Self-Attention (conditional)
        if self.self_attention_image:
            x_image = self.self_attention_image(x_image)
        if self.self_attention_metadata:
           x_metadata = self.self_attention_metadata(x_metadata)
        # Inter-modality cross-attention
        if self.cross_attn:
           combined = self.cross_attention(x_image, x_metadata)
        else:
            combined = torch.cat([x_image, x_metadata], dim=1)
        # Final classification layer
        output = self.fc_combined(combined)
        
        return output

class MultimodalModel_Fusion(nn.Module):
    def __init__(self, num_heads=4, hidden_dim=512, self_meta=False, self_image=False, num_classes=3):
        super(MultimodalModel_Fusion, self).__init__()
        self.num_classes = num_classes
        # Image model using EfficientNetB0
        self.image_model = efficientnet_b0(weights="IMAGENET1K_V1")
        
        # Lấy in_features từ classifier ban đầu trước khi thay thế
        in_features = self.image_model.classifier[1].in_features
        
        # Thay thế classifier gốc bằng Identity
        self.image_model.classifier[1] = nn.Identity()
        
        # Freeze all layers of the EfficientNetB0 except the last classifier
        for param in self.image_model.features.parameters():
            param.requires_grad = False
            
        # Metadata processing layers
        self.age_norm = nn.BatchNorm1d(1)  # Normalize age
        self.breast_density_emb = nn.Embedding(4, 4)  # Assuming 4 unique breast density categories

        # Fully connected layers for image and metadata
        self.fc_image = nn.Linear(in_features, 512) # Use the correct in_features
        self.fc_metadata = nn.Linear(1 + 4, 512)  # 1 age + 4 breast_density
        
         # Intra-Modality Self-Attention blocks (conditional)
        self.self_attention_image = SelfAttention(input_dim=512, hidden_dim=hidden_dim, num_heads=num_heads) if self_image else None
        self.self_attention_metadata = SelfAttention(input_dim=512, hidden_dim=hidden_dim, num_heads=num_heads) if self_meta else None
      
       
        # Final classification layer
        self.fc_combined = nn.Linear(512 + 512, self.num_classes) # Combine image and metadata features, số lớp là num_classes

    def forward(self, x_image, age, breast_density):
        # Process image
        x_image = self.image_model(x_image)
        x_image = F.relu(self.fc_image(x_image))
        # Process metadata
        age_norm = self.age_norm(age.view(-1, 1))
        breast_density = breast_density.long()
        breast_density_emb = self.breast_density_emb(breast_density)
        metadata = torch.cat([age_norm, breast_density_emb], dim=1)
        x_metadata = F.relu(self.fc_metadata(metadata))
        
         # Intra-modality Self-Attention (conditional)
        if self.self_attention_image:
            x_image = self.self_attention_image(x_image)
        if self.self_attention_metadata:
           x_metadata = self.self_attention_metadata(x_metadata)

        # Final classification layer
        combined = torch.cat([x_image, x_metadata], dim=1)
        output = self.fc_combined(combined)
        
        return output

def get_model(args):
    if args.model == "cross_attn":
        return MultimodalModel_CrossAttn(self_meta = args.self_meta, self_image = args.self_image, cross_attn = args.cross_attn, num_classes = args.num_classes)
    elif args.model == "fusion":
      return MultimodalModel_Fusion(self_meta = args.self_meta, self_image = args.self_image, num_classes = args.num_classes)
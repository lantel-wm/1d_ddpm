import torch
from torch import nn
import torch.nn.functional as F
import json

# from .resnet import ResNet, BasicBlock
# from .resnet_adabn import AdaBNResNet, AdaBNBlock
from .resnet_time_embd import ResNetTE, TEBlock
from .clip import CLIP


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLIPFinetune(nn.Module):
    """
    Noisy image encoder subtitutes the all the batch norm layers to adaptive batch norm (adabn) layers
    in the original image encoder. Before finetuning, the weights of the image encoder are copied to
    the noisy image encoder except for the batch norm layers.
    """
    def __init__(
        self,
        temperature=1.0,
        image_embedding=512,
        text_embedding=512,
        diffusion_steps=1000,
        clip_model_path = None,
        device=None,
    ):
        super().__init__()
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.clip_model = CLIP(temperature, image_embedding, text_embedding).to(self.device)
        self.clip_model.load_state_dict(torch.load(clip_model_path))
        self.clip_model.eval()
        
        self.image_encoder = self.clip_model.image_encoder
        self.noisy_image_encoder = ResNetTE(img_channels=1,
            num_layers=18, 
            block=TEBlock,
            diffusion_steps=diffusion_steps,
        ).to(self.device)
        self.image_projection = self.clip_model.image_projection
        self.noisy_image_projection = ProjectionHead(embedding_dim=image_embedding).to(self.device)
        self.temperature = temperature
        
        # ie_keys = list(self.image_encoder.state_dict().keys())
        # nie_keys = list(self.noisy_image_encoder.state_dict().keys())
        # print(set(ie_keys) - set(nie_keys))
        # print(set(nie_keys) - set(ie_keys))
        
        # assert 0, "Stop here"
        
        self._load_noisy_image_encoder()
        self._load_noisy_image_projection()
        self._freeze_image_encoder()   
    
    
    def _load_noisy_image_encoder(self):
        # Copying the weights of the image encoder to the noisy image encoder except for the batch norm layers
        ie_dict = self.image_encoder.state_dict()
        nie_dict = dict(self.noisy_image_encoder.state_dict())
        for k in ie_dict.keys():
            if 'bn' in k:
                continue
            if k not in self.noisy_image_encoder.state_dict().keys():
                continue
            nie_dict[k].copy_(ie_dict[k])
        self.noisy_image_encoder.load_state_dict(nie_dict)
            
            
    def _load_noisy_image_projection(self):
        # Copying the weights of the image projection to the noisy image projection
        ip_dict = self.image_projection.state_dict()
        for k in ip_dict.keys():
            self.noisy_image_projection.state_dict()[k].copy_(ip_dict[k])
            
            
    def _freeze_image_encoder(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.image_projection.parameters():
            param.requires_grad = False
    
    # @profile
    def forward(self, image, noisy_image, class_id):
        # Getting Image and Text Features
        image_features = self.image_encoder(image) # shape: (batch_size, 512)
        noisy_image_features = self.noisy_image_encoder(noisy_image, class_id) # shape: (batch_size, 512)
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        noisy_image_embeddings = self.noisy_image_projection(noisy_image_features)

        # Calculating the Loss
        logits = (noisy_image_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        noisy_image_similarity = noisy_image_embeddings @ noisy_image_embeddings.T
        targets = F.softmax(
            (images_similarity + noisy_image_similarity) / 2 * self.temperature, dim=-1
        )
        
        noisy_images_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + noisy_images_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    

import torch
from torch import nn
import torch.nn.functional as F

from .resnet import ResNet, BasicBlock


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


class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=1.0,
        image_embedding=512,
        text_embedding=512,
    ):
        super().__init__()
        # image_encoder and text_encoder are the same (resnet18)
        self.image_encoder = ResNet(img_channels=1, num_layers=18, block=BasicBlock)
        self.text_encoder = ResNet(img_channels=1, num_layers=18, block=BasicBlock)
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, image, text):
        # Getting Image and Text Features
        image_features = self.image_encoder(image) # shape: (batch_size, 512)
        text_features = self.text_encoder(text) # shape: (batch_size, 512)
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        
        # print(torch.sqrt(torch.sum(image_embeddings**2, dim=-1)))
        # print(torch.sqrt(torch.sum(text_embeddings**2, dim=-1)))

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        # targets = torch.eye(logits.shape[0], device=logits.device)
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
    
if __name__ == '__main__':
    model = CLIPModel()
    img = torch.randn(16, 1, 960)
    txt = torch.randn(16, 1, 240)
    # print(img, txt)
    loss = model(img, txt)
    print(loss)
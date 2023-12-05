import torch
from .clip import CLIPModel

class CLIPInfer:
    def __init__(self, model_path:str, device=None) -> None:
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.model = CLIPModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
    def image_encoder(self, image:torch.Tensor) -> torch.Tensor:
        image_features = self.model.image_encoder(image) # shape: (batch_size, 512)
        image_embeddings = self.model.image_projection(image_features)
        return image_embeddings
    
    def text_encoder(self, text:torch.Tensor) -> torch.Tensor:
        text_features = self.model.text_encoder(text)
        text_embeddings = self.model.text_projection(text_features)
        return text_embeddings
        
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim=960, hidden_dim=512, z_dim=64):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, z_dim * 2)
        self.relu = nn.LeakyReLU()
        
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = x[:, :, :self.z_dim]
        logvar = x[:, :, self.z_dim:]
        
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder block
    Built to be a mirror of the encoder block
    """

    def __init__(self, hidden_dim=512, z_dim=64, output_dim=960):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.LeakyReLU()
        

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x 


class VAE1d(nn.Module):
    """
    VAE network, uses the above encoder and decoder blocks
    """
    def __init__(self, dim=960, hidden_dim=512, z_dim=64):
        super(VAE1d, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image 
        z = the number of channels of the latent representation
        (for a 64x64 image this is the size of the latent vector)
        """
        
        self.encoder = Encoder(input_dim=dim, hidden_dim=hidden_dim, z_dim=z_dim)
        self.decoder = Decoder(output_dim=dim, hidden_dim=hidden_dim, z_dim=z_dim)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        recon_img = self.decoder(z)
        return recon_img, mean, logvar
    
    def encode(self, x):
        mean, logvar = self.encoder(x)
        return self.reparameterize(mean, logvar)
    
    def decode(self, z):
        return self.decoder(z)
    
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    
    
if __name__ == '__main__':
    model = VAE1d(z_dim=64, hidden_dim=512)
    img = torch.randn(16, 1, 960)
    recon_img, mu, log_var = model(img)
    print(recon_img.shape, mu.shape, log_var.shape)
    
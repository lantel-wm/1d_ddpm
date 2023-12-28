import torch
from torch import nn, optim
import torch.nn.functional as F

class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, is_conv=True):
        super(Conv_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding 
        self.pool_op = nn.AvgPool1d(2, ) if is_conv \
                  else nn.Upsample(scale_factor=2, mode='linear')
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, padding_mode='circular')
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.99)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.pool_op(x)
    
    
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, in_length, nclasses, latent_size, encoder_out_channels):
        super(Encoder, self).__init__()
        
        self.in_channels = in_channels
        self.in_length = in_length
        self.nclasses = nclasses
        self.latent_size = latent_size
        self.encoder_out_channels = encoder_out_channels
        length = self.in_length
        self.bn0 = torch.nn.BatchNorm1d(self.in_channels, eps=0.001, momentum=0.99)
        # Layer 1
        in_channels = self.in_channels
        out_channels = 32
        kernel_size = 81
        padding = kernel_size // 2
        self.conv_block_1 = Conv_block(in_channels, out_channels, kernel_size, padding)
        length = length // 2
        # Layer 2
        in_channels = out_channels
        out_channels = 32
        kernel_size = 81
        padding = kernel_size // 2
        self.conv_block_2 = Conv_block(in_channels, out_channels, kernel_size, padding)
        length = length // 2
        
        # Layer 3
        in_channels = out_channels
        last_featuremaps_channels = 64
        kernel_size = 81
        padding = kernel_size // 2
        self.conv_block_3 = Conv_block(in_channels, last_featuremaps_channels, kernel_size, padding)
        length = length // 2
        
        in_channels = last_featuremaps_channels
        out_channels = 1
        kernel_size = 27
        padding = kernel_size // 2
        self.conv_final = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, padding_mode='circular')
        self.gp_final = torch.nn.AvgPool1d(length)
        
        # encoder
        in_channels = last_featuremaps_channels
        out_channels = self.encoder_out_channels
        kernel_size = 51
        padding = kernel_size // 2
        self.adapt_pool = torch.nn.AvgPool1d(2); length = length // 2
        self.adapt_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, padding_mode='circular')
        self.encode_mean = torch.nn.Linear(length*out_channels, self.latent_size)
        self.encode_logvar = torch.nn.Linear(length*out_channels, self.latent_size)
        self.relu = torch.nn.ReLU()
        length = 1

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.in_length)
        x = self.bn0(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        cv_final = self.conv_final(x)
        oh_class = self.gp_final(cv_final)
        x = self.adapt_pool(x)
        x = self.adapt_conv(x)
        x = x.view(x.size(0), -1)
        mean = self.relu(self.encode_mean(x)) 
        logvar = self.relu(self.encode_logvar(x))
        return mean, logvar
    
    
class Decoder(torch.nn.Module):
    def __init__(self, length, in_channels, nclasses, latent_size):
        super(Decoder, self).__init__()
        
        self.in_channels = in_channels
        self.length = length
        self.latent_size = latent_size
        length = self.length  
        length = length // 2 // 2 // 2 
        # Adapt Layer
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.adapt_nn = torch.nn.Linear(latent_size, self.in_channels*length)
        # Layer 1
        in_channels = self.in_channels
        out_channels = 64
        kernel_size = 81
        padding = kernel_size // 2
        self.deconv_block_1 = Conv_block(in_channels, out_channels, kernel_size, padding, is_conv=False)
        length = length * 2
        # Layer 2
        in_channels = out_channels
        out_channels = 32
        kernel_size = 81
        padding = kernel_size // 2
        self.deconv_block_2 = Conv_block(in_channels, out_channels, kernel_size, padding, is_conv=False)
        length = length * 2
        
        # Layer 3
        in_channels = out_channels
        out_channels = 32
        kernel_size = 81
        padding = kernel_size // 2
        self.deconv_block_3 = Conv_block(in_channels, out_channels, kernel_size, padding, is_conv=False)
        length = length * 2
        
        in_channels = out_channels
        out_channels = 1
        kernel_size = 27
        padding = kernel_size // 2
        self.decode_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, padding_mode='circular')
        
    def forward(self, z):

        x = self.relu(self.adapt_nn(z))
        x = x.view(x.size(0), self.in_channels, self.length // 2 // 2 // 2)
        x = self.deconv_block_1(x)
        x = self.deconv_block_2(x)
        x = self.deconv_block_3(x)
        x = self.decode_conv(x)
        out = self.tanh(x)
        return out
    
    
class VAE1d(torch.nn.Module):
    def __init__(self, length=960, nclasses=1, z_dim=64, transition_channels=64):
        super(VAE1d, self).__init__()
        self.encoder = Encoder(1, length, nclasses, z_dim, transition_channels)
        self.decoder = Decoder(length, transition_channels, nclasses, z_dim)


    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decoder(z)
        return x_recon, mean, logvar
    
    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean
    
    
if __name__ == '__main__':
    vae = VAE1d(960, 1, 64, 64)
    x_recon, mean, logvar = vae(torch.randn(16, 1, 960))
    print(x_recon.shape)
    print(mean.shape)
    print(logvar.shape)
    
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3):
        super(ResDown, self).__init__()
        self.conv1 = nn.Conv1d(channel_in, channel_out // 2, kernel_size, 2, kernel_size // 2, padding_mode='circular')
        self.bn1 = nn.BatchNorm1d(channel_out // 2, eps=1e-4)
        self.conv2 = nn.Conv1d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2, padding_mode='circular')
        self.bn2 = nn.BatchNorm1d(channel_out, eps=1e-4)

        self.conv3 = nn.Conv1d(channel_in, channel_out, kernel_size, 2, kernel_size // 2, padding_mode='circular')

        self.act_fnc = nn.ELU()

    def forward(self, x):
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return self.act_fnc(self.bn2(x + skip))


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2):
        super(ResUp, self).__init__()

        self.conv1 = nn.Conv1d(channel_in, channel_in // 2, kernel_size, 1, kernel_size // 2, padding_mode='circular')
        self.bn1 = nn.BatchNorm1d(channel_in // 2, eps=1e-4)
        self.conv2 = nn.Conv1d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2, padding_mode='circular')
        self.bn2 = nn.BatchNorm1d(channel_out, eps=1e-4)

        self.conv3 = nn.Conv1d(channel_in, channel_out, kernel_size, 1, kernel_size // 2, padding_mode='circular')

        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")

        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.up_nn(x)
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return self.act_fnc(self.bn2(x + skip))


class Encoder(nn.Module):
    """
    Encoder block
    Built for a 3x64x64 image and will result in a latent vector of size z x 1 x 1
    As the network is fully convolutional it will work for images LARGER than 64
    For images sized 64 * n where n is a power of 2, (1, 2, 4, 8 etc) the latent feature map size will be z x n x n

    When in .eval() the Encoder will not sample from the distribution and will instead output mu as the encoding vector
    and log_var will be None
    """
    def __init__(self, channels, ch=1, z_dim=64):
        assert ch * 8 == z_dim, f'z_dim must be 8  * {ch}'
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv1d(channels, ch, 7, 1, 3)
        self.res_down_block1 = ResDown(ch, 2 * ch)
        self.res_down_block2 = ResDown(2 * ch, 4 * ch)
        self.res_down_block3 = ResDown(4 * ch, 8 * ch)
        self.res_down_block4 = ResDown(8 * ch, 16 * ch)
        # self.conv_mu = nn.Conv1d(16 * ch, z_dim, 4, 1, padding_mode='circular')
        # self.conv_log_var = nn.Conv1d(16 * ch, z_dim, 4, 1, padding_mode='circular')
        self.act_fnc = nn.ELU()
        self.z_dim = z_dim
        

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x):
        x = self.act_fnc(self.conv_in(x))
        x = self.res_down_block1(x)  # 32
        x = self.res_down_block2(x)  # 16
        x = self.res_down_block3(x)  # 8
        x = self.res_down_block4(x)  # 4
        # print('before avg pool: ', x.shape)
        x = F.adaptive_avg_pool1d(x, 1)
        # print('after avg pool: ', x.shape)
        # x = x.view(x.size(0), 1, -1)
        # print('after fc: ', x.shape)
        mu = x[:, :self.z_dim]  # 1
        log_var = x[:, self.z_dim:]  # 1
        # print('mu: ', mu.shape)
        # print('log_var: ', log_var.shape)

        if self.training:
            x = self.reparameterize(mu, log_var)
        else:
            x = mu

        return x, mu, log_var


class Decoder(nn.Module):
    """
    Decoder block
    Built to be a mirror of the encoder block
    """

    def __init__(self, channels, ch=1, z_dim=64):
        super(Decoder, self).__init__()
        self.conv_t_up = nn.ConvTranspose1d(z_dim, ch * 16, 4, 1)
        self.res_up_block1 = ResUp(ch * 16, ch * 8, scale_factor=3)
        self.res_up_block2 = ResUp(ch * 8, ch * 4, scale_factor=5)
        self.res_up_block3 = ResUp(ch * 4, ch * 2, scale_factor=4)
        self.res_up_block4 = ResUp(ch * 2, ch, scale_factor=4)
        self.conv_out = nn.Conv1d(ch, channels, 3, 1, 1)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.act_fnc(self.conv_t_up(x))  # 4
        x = self.res_up_block1(x)  # 8
        x = self.res_up_block2(x)  # 16
        x = self.res_up_block3(x)  # 32
        x = self.res_up_block4(x)  # 64
        x = torch.tanh(self.conv_out(x))

        return x 


class VAE1d(nn.Module):
    """
    VAE network, uses the above encoder and decoder blocks
    """
    def __init__(self, channel_in=1, ch=2, z_dim=64):
        super(VAE1d, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image 
        z = the number of channels of the latent representation
        (for a 64x64 image this is the size of the latent vector)
        """
        
        self.encoder = Encoder(channel_in, ch=ch, z_dim=z_dim)
        self.decoder = Decoder(channel_in, ch=ch, z_dim=z_dim)

    def forward(self, x):
        encoding, mu, logvar = self.encoder(x)
        recon_img = self.decoder(encoding)
        return recon_img, mu, logvar
    
    def encode(self, x):
        mean, logvar = self.encoder(x)
        return self.reparameterize(mean, logvar)
    
    def decode(self, z):
        return self.decoder(z)
    
    
    
if __name__ == '__main__':
    model = VAE1d(z_dim=64, ch=8)
    img = torch.randn(16, 1, 960)
    recon_img, mu, log_var = model(img)
    print(recon_img.shape, mu.shape, log_var.shape)
    
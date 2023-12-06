# Adaptive batch normalization
import torch
import torch.nn as nn

class AdaBN1d(nn.Module):
    def __init__(self, num_features: int, 
        num_classes: int, 
        epsilon=1e-5, 
        momentum=0.1, 
        # affine=True, 
        # track_running_stats=True
    ):
        super(AdaBN1d, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        
        # mean and standard deviation
        self.mu = torch.zeros(1, num_features)
        self.sigma = torch.ones(1, num_features)
        
        self.epsilon = epsilon
        
        # exponential moving average
        self.momentum = momentum
        self.it_count = 0 # iteration count
        
        # trainable parameters
        self.beta = nn.Parameter(torch.zeros(1, num_features))
        self.gamma = nn.Parameter(torch.ones(1, num_features))
        
        self.batch_size = 0
        
        def forward(self, x: torch.Tensor):
            # [N, C, L]  or [N, C]
            self.it_count += 1
            
            if self.training:
                if self.batch_size == 0:
                    # first iteration, save batch size
                    self.batch_size = x.shape[0]
                
                # only keep the channel dim
                reduction_dims = (0, 2) if len(x.shape) == 3 else 0 # dims to be reduced
                batch_mu = x.mean(dim=reduction_dims).unsqueeze(0) # [1, num_features]
                batch_sigma = x.std(dim=reduction_dims).unsqueeze(0) # [1, num_features]
                
                x_normalized = (x - batch_mu) / (batch_sigma + self.epsilon) # [batch_size, num_features]
                x_bn = self.gamma * x_normalized + self.beta # [batch_size, num_features]
                
                # update mean and standard deviation
                self.mu = self.momentum * batch_mu + (1 - self.momentum) * self.mu
                self.sigma = self.momentum * batch_sigma + (1 - self.momentum) * self.sigma
                
                
            else: # inference
                x_normalized = (x - self.mu) / (self.sigma + self.epsilon) # [batch_size, num_features]
                x_bn = self.gamma * x_normalized + self.beta
                
            return x_bn
        
        
if __name__ == '__main__':
    adabn = AdaBN1d(16, 512)
    x = torch.randn(32, 16, 240)
    y = adabn(x)
    print(y.shape)
                 
                
                
        
        
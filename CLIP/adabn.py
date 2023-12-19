# Adaptive batch normalization
import torch
import torch.nn as nn

class AdaBN1d(nn.Module):
    def __init__(self, num_features: int, 
        num_classes: int = 1000, 
        epsilon=1e-5, 
        momentum=0.1, 
        # affine=True, 
        # track_running_stats=True
    ):
        super(AdaBN1d, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        
        # mean and standard deviation
        self.mu = torch.zeros(num_classes, 1, num_features, 1).to('cuda')
        self.sigma = torch.ones(num_classes, 1, num_features, 1).to('cuda')
        
        self.epsilon = epsilon
        
        # exponential moving average
        self.momentum = momentum
        self.it_count = 0 # iteration count
        
        # trainable parameters
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1))
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1))
        # TODO: try beta / gamma shape of [num_classes, 1, num_features, 1]
        
        self.batch_size = 0
        
    # @profile   
    def forward(self, x, class_id: int):
        # [N, C, L]  or [N, C]
        assert len(x.shape) == 3 or len(x.shape) == 2, "Input tensor must be 3d or 2d"
        assert class_id < self.num_classes, f"Class id {class_id} must be less than num_classes {self.num_classes}"
        self.it_count += 1
        
        is_3d = (len(x.shape) == 3)
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        
        if self.training:
            if self.batch_size == 0:
                # first iteration, save batch size
                self.batch_size = x.shape[0]
            
            # only keep the channel dim
            reduction_dims = (0, 2) if len(x.shape) == 3 else 0 # dims to be reduced
            batch_mu = x.mean(dim=reduction_dims).unsqueeze(0) # [1, num_features]
            batch_sigma = x.std(dim=reduction_dims).unsqueeze(0) # [1, num_features]
            
            if len(x.shape) == 3:
                batch_mu = batch_mu.unsqueeze(-1)
                batch_sigma = batch_sigma.unsqueeze(-1)
            
            # batch_mu and batch_sigma MUST BE DETACHED FROM THE COMPUTATION GRAPH!!!
            # use detach() to detach the tensor from the computation graph
            # otherwise, the computation graph will be retained and the memory will be used up
            x_normalized = (x - batch_mu.detach()) / (batch_sigma.detach() + self.epsilon) # [batch_size, num_features]
            x_bn = self.gamma * x_normalized + self.beta # [batch_size, num_features]
            
            # update mean and standard deviation
            self.mu[class_id] = self.momentum * batch_mu.detach() + (1 - self.momentum) * self.mu[class_id]
            self.sigma[class_id] = self.momentum * batch_sigma.detach() + (1 - self.momentum) * self.sigma[class_id]
            
            
        else: # inference
            x_normalized = (x - self.mu[class_id].squeeze(0)) / (self.sigma[class_id].squeeze(0) + self.epsilon) # [batch_size, num_features]
            x_bn = self.gamma * x_normalized + self.beta
            
        return x_bn if is_3d else x_bn.squeeze(-1)
    
        
if __name__ == '__main__':
    adabn = AdaBN1d(16, 300)
    x = torch.randn(32, 16, 960)
    y = adabn(x, 0)
    print(y.shape)
                 
                
                
        
        
    
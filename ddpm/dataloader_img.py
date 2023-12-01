import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


IMG_SIZE = 256
# batch_size = 128

def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.StanfordCars(root="/home/zzy/zyzhao/diffusion/1d_ddpm", download=True, 
                                         transform=data_transform, split='train')

    test = torchvision.datasets.StanfordCars(root="/home/zzy/zyzhao/diffusion/1d_ddpm", download=True, 
                                         transform=data_transform, split='test')
    return torch.utils.data.ConcatDataset([train, test])

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))
    
def get_dataloader(batch_size: int):
    data = load_transformed_dataset()
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader


if __name__ == "__main__":
    # data = load_transformed_dataset()
    # dataloader = DataLoader(data, batch_size=64, shuffle=True, drop_last=True)
    
    # # Simulate forward diffusion
    # image = next(iter(dataloader))[0]
    # forward_diffusion_sample = ForwardProcess(300)
    # plt.figure(figsize=(15,15))
    # plt.axis('off')
    # T = 300
    # num_images = 10
    # stepsize = int(T/num_images)

    # for idx in range(0, T, stepsize):
    #     t = torch.Tensor([idx]).type(torch.int64)
    #     plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
    #     img, noise = forward_diffusion_sample.forward(image, t)
    #     show_tensor_image(img)
    # plt.savefig("forward_diffusion.png")
    pass
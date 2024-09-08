import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from diffusers import AutoencoderKL
import matplotlib.pyplot as plt
from typing import Tuple


class ImageFolder(ImageFolder):
    def __init__(self, root : str, transform : v2.Transform = None, target_transform : v2.Transform = None) -> None:
        super(ImageFolder, self).__init__(root, transform)
        self.num_classes = len(self.classes)

    def __getitem__(self, idx):
        image, label = super(ImageFolder, self).__getitem__(idx)
        one_hot_label = F.one_hot(torch.tensor(label), num_classes = self.num_classes).to(torch.float32)
        return image, one_hot_label



def get_loaders(root : str, 
                split_ratio : tuple = (0.8, 0.2),
                batch_size : int = 256,
                shuffle : bool = True,
                transform : v2.Transform = None, 
                target_transform : v2.Transform = None) -> Tuple[DataLoader, DataLoader]:
    dataset = ImageFolder(root, transform = transform, target_transform = target_transform)
    train_ratio = split_ratio[0]
    val_ratio = split_ratio[1]
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = shuffle)

    return train_loader, val_loader


def visualize(images, labels):
    plt.figure(figsize=(10, 10))
    for idx, (image, label) in enumerate(zip(images, labels)):
        if idx >= 9:
            break
        plt.subplot(3, 3, idx+1)
        image = image.permute(1, 2, 0)
        plt.imshow(image)
        plt.title(torch.argmax(label).item())
        plt.axis("off")
    plt.show()


def get_AutoEncoder(model_name : str = "CompVis/stable-diffusion-v1-4", subfolder : str = "vae"):
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    return vae


def get_latent_vector(vae, images_batch):
    z = vae.encode(images_batch).latent_dist.sample() 
    return z

def decode_images(vae, latent_vectors_batch):
    reconstructed_images_batch = vae.decode(latent_vectors_batch).sample
    return reconstructed_images_batch





"""
root_dir = "data"
split_ratio = (0.7, 0.3)
batch_size = 6
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
train_loader, val_loader = get_loader(root = root_dir, split_ratio = split_ratio, batch_size = batch_size, transform = transform)
images, labels = next(iter(train_loader))

vae = get_AutoEncoder()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
vae = vae.to(device)
images = images.to(device)
print(images.shape)

latent_vectors = get_latent_vector(vae, images)
print(latent_vectors.shape)

reconstructed_images = decode_images(vae, latent_vectors)
print(reconstructed_images.shape)
"""





"""
input_image = images[0].unsqueeze(0).to(device) 
z = vae.encode(input_image).latent_dist.sample()
reconstructed_image = vae.decode(z).sample
reconstructed_image = reconstructed_image.squeeze(0).permute(1, 2, 0).cpu().detach()

reconstructed_image = (reconstructed_image + 1.0) / 2.0  

plt.figure(figsize=(10, 10))

plt.subplot(1, 3, 1)
plt.imshow(input_image.squeeze(0).permute(1, 2, 0).cpu())
plt.title("Original Image")
plt.axis("off")


plt.subplot(1, 3, 2)
plt.imshow(reconstructed_image)
plt.title("Reconstructed Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(z.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
plt.title("Latent Representation")
plt.axis("off")

plt.show()
"""

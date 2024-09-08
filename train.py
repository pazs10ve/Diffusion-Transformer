import torch
from utils import get_loaders
from utils import get_AutoEncoder 
from utils import get_latent_vector as encode
from utils import decode_images as decode
from model import get_model
from torchvision.transforms import v2
import torch.optim as optim
import matplotlib.pyplot as plt

tmax = 1000

def gaussian_nll_loss(predicted_noise, true_noise, covariance):
    variance = torch.exp(covariance) 
    loss = 0.5 * torch.sum(((predicted_noise - true_noise.permute(0, 2, 3, 1)) ** 2) / variance + torch.log(variance))
    return loss



def train_one_epoch(model, train_loader, autoencoder, optimizer, device):
    running_train_loss = 0
    model.train()

    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f'{batch_idx + 1} / {len(train_loader)}')
        images, labels = images.to(device), labels.to(device)

        latent_vectors = encode(autoencoder, images)
        images = images.detach()

        timesteps = torch.randint(0, tmax, (images.size(0),), device=device)

        noisy_latent_vectors, noise = model.forward_diffusion(latent_vectors, timesteps)
        noisy_latent_vectors, noise = noisy_latent_vectors.to(device), noise.to(device)
        predicted_noise, covariance = model(noisy_latent_vectors, labels, timesteps)

        loss = gaussian_nll_loss(predicted_noise, noise, covariance)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    epoch_train_loss = running_train_loss / len(train_loader)
    return epoch_train_loss

    
def validate_one_epoch(model, val_loader, autoencoder, device):
    running_val_loss = 0
    model.train()

    for batch_idx, (images, labels) in enumerate(val_loader):
        print(f'{batch_idx + 1} / {len(val_loader)}')
        images, labels = images.to(device), labels.to(device)

        latent_vectors = encode(autoencoder, images)
        images = images.to(device)

        timesteps = torch.randint(0, tmax, (images.size(0),), device = device)
        noisy_latent_vectors, noise = model.forward_diffusion(latent_vectors, timesteps)
        predicted_noise, covariance = model(noisy_latent_vectors, labels, timesteps)

        loss = gaussian_nll_loss(predicted_noise, noise, covariance)

        running_val_loss += loss.item()

    epoch_val_loss = running_val_loss / len(val_loader)
    return epoch_val_loss    



def train(model, train_loader, val_loader, autoencoder, optimizer, device, num_epochs, save_path="dit_model.pth"):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, autoencoder, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, autoencoder, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch + 1} / {num_epochs} --> Training Loss : {train_loss}, Validation Loss : {val_loss}')
        
        torch.save(model.state_dict(), f'{save_path}_epoch_{epoch + 1}.pth')

    plot_loss_histories(train_losses, val_losses)


def plot_loss_histories(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    
    plt.plot(train_losses, label="Training Loss", color='blue', linestyle='-')
    plt.plot(val_losses, label="Validation Loss", color='red', linestyle='--')
    
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


DiT_s_config = {
    'batch_size' : 16,
    'input_dim' : 32,
    'label_dim' : 1000,
    'channels' : 4,
    'patch_size' : 2,
    'cond_dim' : 384,
    'hidden_dim' : 384,
    'num_heads' : 6,
    'num_layers' : 12
}


DiT_B_config = {
    'batch_size' : 16,
    'input_dim' : 32,
    'label_dim' : 1000,
    'channels' : 4,
    'patch_size' : 4,
    'cond_dim' : 768,
    'hidden_dim' : 768,
    'num_heads' : 12,
    'num_layers' : 12
}


DiT_L_config = {
    'batch_size' : 16,
    'input_dim' : 32,
    'label_dim' : 1000,
    'channels' : 4,
    'patch_size' : 8,
    'cond_dim' : 1024,
    'hidden_dim' : 1024,
    'num_heads' : 16,
    'num_layers' : 24
}


DiT_XL_config = {
    'batch_size' : 16,
    'input_dim' : 32,
    'label_dim' : 1000,
    'channels' : 4,
    'patch_size' : 8,
    'cond_dim' : 1152,
    'hidden_dim' : 1152,
    'num_heads' : 16,
    'num_layers' : 28
}



root = "data"
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DiT_s_model = get_model(DiT_s_config).to(device)
train_loader, val_loader = get_loaders(root = root, split_ratio = (0.8, 0.2), batch_size = DiT_s_config['batch_size'], transform = transform)
autoencoder = get_AutoEncoder().to(device)
lr = 1e-4
optimizer = optim.AdamW(DiT_s_model.parameters(), lr = lr)
num_epochs = 1
DiT_s_model.betas = DiT_s_model.betas.to(device)
train(DiT_s_model, train_loader, val_loader, autoencoder, optimizer, device, num_epochs, save_path="dit_model.pth")

"""
root = "data"
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DiT_s_model = get_model(DiT_s_config).to(device)
train_loader, val_loader = get_loaders(root = root, split_ratio = (0.8, 0.2), batch_size = DiT_s_config['batch_size'], transform = transform)
autoencoder = get_AutoEncoder().to(device)
lr = 1e-4
optimizer = optim.AdamW(DiT_s_model.parameters(), lr = lr)
num_epochs = 1
DiT_s_model.betas = DiT_s_model.betas.to(device)
train(DiT_s_model, train_loader, val_loader, autoencoder, optimizer, device, num_epochs, save_path="dit_model.pth")
"""
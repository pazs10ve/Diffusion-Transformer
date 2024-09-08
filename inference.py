import torch
import torch.nn.functional as F
from utils import get_AutoEncoder 
from model import get_model
import matplotlib.pyplot as plt
from tqdm import tqdm



def load_model(model_class, checkpoint_path, device):
    model = model_class().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model



def infer(model, autoencoder, num_samples, class_label, device, timesteps=1000):
    model.eval()
    autoencoder.eval()
    
    class_one_hot = torch.tensor([class_label], device=device)
    class_one_hot = F.one_hot(class_one_hot, num_classes=1000).to(torch.float32).repeat(num_samples, 1)
    
    noise = torch.randn(num_samples, 4, model.input_dim, model.input_dim, device=device)
    noise = noise.to(device)
    class_one_hot = class_one_hot.to(device)
    
    for timestep in tqdm(range(timesteps), desc="Generating Images"):
        with torch.no_grad():
            noise, _ = model(noise, class_one_hot, torch.tensor([timestep/timesteps], device=device))
            noise = noise.permute(0, 3, 1, 2)
    with torch.no_grad():     
        decoded_images = autoencoder.decode(noise).sample.cpu()
        return decoded_images
    



def show_images(images):
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DiT_s_config = {
    'batch_size' : 16,
    'input_dim' : 32,
    'label_dim' : 1000,
    'channels' : 4,
    'patch_size' : 4,
    'cond_dim' : 384,
    'hidden_dim' : 384,
    'num_heads' : 6,
    'num_layers' : 12
}

"""
model = get_model(DiT_s_config).to(device)
model = load_model(model, model_checkpoint_path, device)
autoencoder = get_AutoEncoder().to(device)
num_samples = 2
class_label = 782 
sampled_images = infer(model, autoencoder, num_samples, class_label, device, timesteps=10000)
show_images(sampled_images)

"""

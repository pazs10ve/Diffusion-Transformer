# Diffusion-Transformer
This project is based on the paper "Diffusion-Transformer" which demonstrates the capability of the DiT model in the Latent Diffusion Models (LDMs) framework. In this framework, diffusion models are trained within the latent space of Variational Autoencoders (VAEs). The DiT model replaces the traditional U-Net backbone with a transformer architecture, effectively leveraging the strengths of transformers in image generation tasks.

# Components
## Patchify
The input to DiT is a spatial representation z. For example, a 256x256x3 image results in z having the shape 32x32x4. The first layer of DiT, called "patchify", converts this spatial input into a sequence of tokens. Each token has a dimension d, achieved by linearly embedding each patch in the input. Following the "patchify" layer, standard Vision Transformer (ViT) frequency-based positional embeddings (sine-cosine version) are applied to all input tokens.

## DiT Block
The core of the DiT model is the DiT Block, which includes multi-head self-attention mechanisms and feed-forward layers. The DiT Block processes the input tokens, incorporating positional information and class conditioning to generate high-quality image features.

## DiT Model
The DiT model integrates multiple DiT Blocks to form the complete architecture. It receives tokenized image patches and class information, and applies a series of transformer layers to produce high-fidelity image representations.

## Training
### Optimizer: AdamW
### Learning Rate: Constant at 1e-4
### Weight Decay: None
### Batch Size: 256
### Data Augmentation: Horizontal flips
The model is trained with a constant learning rate of 1e-4 using the AdamW optimizer. No weight decay is applied, and the only data augmentation technique employed is horizontal flipping.

## Evaluation
The model's performance is evaluated using Frechet Inception Distance (FID), a metric that assesses the quality and diversity of generated images by comparing their statistics to those of real images.

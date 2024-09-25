import torch
from torchvision.models import vgg11
from torch.nn import Conv2d
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange


class VGG(torch.nn.Module):
    
    """
    CNN model for image classification
    Adapted first layer of vgg11 to match the 13 bands of the multispectral image


    Args:
        num_bands (int): Number of bands in the multispectral image
        num_classes (int): Number of classes
    """

    def __init__(self, num_bands, num_classes):
        super().__init__()
        self.model = vgg11(num_classes=num_classes)
        self.model.features[0] = Conv2d(num_bands, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self,x):
        return self.model(x)

class ViT(torch.nn.Module):

    """
    ViT model for image classification
    comments correpsond to a 8x8 patch pattern with a 13x64x64 input image
    with batchsize 2


    Args:
        image_size (int): Size of the image (image assumed to be square)
        num_channels (int): Number of channels in the image
        patch_size (int): Size of the patch
        num_classes (int): Number of classes in the dataset
        embed_dim (int): Dimension of the embedding
        num_heads (int): Number of heads in the multihead attention
        num_layers (int): Number of layers in the transformer
        dropout (float): Dropout rate
    """

    def __init__(self,
                 image_size: int,
                 num_channels: int,
                 patch_size: int,
                 num_classes: int,
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float):

        super(ViT, self).__init__()

        self.image_size = image_size
        self.number_of_channels = num_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Calculate the number of patches
        # here we put a patch with all 13 bands into one token
        # if you dont want to do this check out a example for a differnt implementation here:
        # https://github.com/JonathanPrexl/SenPa-MAE/blob/main/src/model.py
        assert image_size % patch_size == 0, "img shape not divisible by patch size"
        self.num_patches = (image_size // patch_size)**2 # how many patches in total eg. 64 for patchsize 8 
        self.sqrt_num_patches = image_size // patch_size # how many patches in one dimension eg. 8 for patchsize 8

        # Patch embedding
        # turn an image into a sequence of patches
        # eg. 13x64x64 with patchsize 8 -> 64x832 (832 = 13x8x8)
        self.patchify = Rearrange('b c (h h1) (w w1) -> b (h1 w1) (h w c)',
                                  h1=self.sqrt_num_patches,
                                  w1=self.sqrt_num_patches)

        # linear embedding of patches 
        # eg. 64x83 -> 64x768
        self.linEmmbedingLayer = nn.Linear(self.patch_size**2 * self.number_of_channels, self.embed_dim) 

        # Positional encoding
        # one more for cls token
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.embed_dim)) # torch.Size([1, 17, 768])

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim)) # torch.Size([1, 1, 768])

        # Transformer encoder 
        # batch first true must be set for "standart" order !!!!!!!!!
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.embed_dim, num_heads, dropout=dropout, batch_first=True),
            num_layers
        )

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        return None


    def forward(self, x):

        assert x.shape[1:] == (self.number_of_channels,self.image_size,self.image_size), "Input shape not correct"

        # Patch embedding
        x = self.patchify(x) # torch.Size([2, 64, 832])

        # lin embedding
        x = self.linEmmbedingLayer(x) # torch.Size([2, 64, 768])

        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # torch.Size([1, 1, 768]) --> torch.Size([2, 1, 768])
        x = torch.cat([cls_token, x], dim=1) # torch.Size([2, 65, 768]) with cls token at position 0

        # Add positional encoding
        x += self.pos_embedding # torch.Size([2, 65, 768])

        # Transformer encoder
        x = self.transformer(x) # torch.Size([2, 65, 768])

        # Classification head
        px = self.head(x[:, 0]) # torch.Size([2, 768]) --> torch.Size([2, 10])

        return px, x


if __name__ == "__main__":

    import hydra
    import omegaconf
    from torchinfo import summary

    cfg = omegaconf.OmegaConf.load("./configs/vit_8x8.yaml")

    model = hydra.utils.instantiate(cfg.model)
    example_batch = torch.zeros(2, 13, 64, 64)
    example_output_shape = model(example_batch).shape
    print(f"With input shape {example_batch} the model outputs a shape {example_output_shape}.")
    summary(model)
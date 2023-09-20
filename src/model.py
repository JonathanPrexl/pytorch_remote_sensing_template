# Author: Jonathan Prexl
# Copyied from:
# https://github.com/IcarusWizard/MAE/
# small changes and comments added, otherwise fully copied!


import torch
import timm
import numpy as np

import torch.nn as nn

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):

    """
    Example:
    x = torch.zeros(3,2,10)
    x[1]=1
    x[2]=2

    ps=PatchShuffle(0.33)
    patches, forward_indexes, backward_indexes = ps(x)

    forward_indexes[:,0]
    >> tensor([2, 0, 1]) # on pos 0 der becomes the 2th element, on pos 1 the 0th elemend etc.

    patches[:,0]
    >> tensor([[2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    backward_indexes[:,0] # just an argsort of forward index. I guess to give it some kind of order. 
    >> tensor([1, 2, 0])
    """

    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio
        assert ratio > 5, "this would be a very small ratio"

    def forward(self, patches : torch.Tensor):

        T, B, C = patches.shape # T = number of patches
        remain_T = int(T * self.ratio//100)

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

class ChannelPatchShuffle(torch.nn.Module):

    """
    channels allways end up in the first dimensions
    img = torch.zeros(2,3,32,32)
    img[:,1]=1
    img[:,2]=2
    patches = rearrange(img, 'b c (h h1) (w w1) -> b (c h1 w1) (h w)', h1=2, w1=2)

    patches[0,1] = all zero
    patches[0,2] = all zero
    patches[0,3] = all zero
    patches[0,4] = all one
    ...
    patches[0,4] = all two

    wheras (not the change in order for after -> )

    patches = rearrange(img, 'b c (h h1) (w w1) -> b (h1 w1 c) (h w)', h1=2, w1=2)
    patches[0,0] = all zero
    patches[0,1] = all one
    patches[0,2] = all two

    so we take the second way here!

    So given nC channels we have choose one of the channels for each the intervalls
    [0...nC]
    [nC...2nC]
    [2nC...3nC]

    so for 10 channels we need one number from 0 to 10, one from 10 to 20 and so on.
    The rest of the list should then fill up so all indices are somewhere.
    idx_to_take = np.arange(0,num_patches*num_channels,num_channels)+np.random.randint(0,num_channels,num_patches)
    indexes = idx_to_take + np.delete(np.arange(num_patches*num_channels),idx_to_take)

    """

    def __init__(self, ratio, num_patches_per_ax, num_channels, keep_one_channel_per_patch, keep_one_of_the_ten_meter_channels) -> None:
        super().__init__()
        self.ratio = ratio
        self.num_patches_per_ax = num_patches_per_ax
        self.num_patches = num_patches_per_ax**2
        self.num_channels = num_channels
        self.keep_one_channel_per_patch = keep_one_channel_per_patch
        self.keep_one_of_the_ten_meter_channels = keep_one_of_the_ten_meter_channels
        if self.keep_one_of_the_ten_meter_channels:
            assert self.num_channels==10, "to much hard coded stuff here, keep it to 10 S2 bands"
        
        assert ratio > 5, "this would be a very small ratio"
        assert ratio < 301, "That is more then 3 channels per patch"
        if self.keep_one_of_the_ten_meter_channels:
            assert self.keep_one_channel_per_patch, "otherwise makes no sense"

    def custom_random_indices(self):

        """ here we make sure we """
        if self.keep_one_of_the_ten_meter_channels:
            # unmask either b, g, r or ir
            idx_to_take = np.arange(0,self.num_patches*self.num_channels,self.num_channels)+np.random.choice([0,1,2,6],self.num_patches)
        else:
            # unmask any
            idx_to_take = np.arange(0,self.num_patches*self.num_channels,self.num_channels)+np.random.randint(0,self.num_channels,self.num_patches)


        rest = np.delete(np.arange(self.num_patches*self.num_channels),idx_to_take)
        np.random.shuffle(rest)
        forward_indexes = np.concatenate([idx_to_take,rest])
        backward_indexes = np.argsort(forward_indexes)
        return forward_indexes, backward_indexes

    def forward(self, patches : torch.Tensor): 

        iT, iB, iC = patches.shape # iT = number of patches

        if self.keep_one_channel_per_patch:
            indexes = [self.custom_random_indices() for _ in range(iB)]
        else:
            indexes = [random_indexes(iT) for _ in range(iB)]

        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)

        # ratio of one would mean of each patch one channel is visible
        if self.keep_one_channel_per_patch:
            T = self.num_patches
        else:
            T = self.num_patches*self.num_channels
            assert T == iT

        remain_T = int(T * self.ratio//100)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

class MAE_Encoder_Default(torch.nn.Module):

    def __init__(self,
                image_size,
                patch_size,
                emb_dim,
                num_layer,
                num_head,
                num_channels,
                mask_ratio,
                ) -> None:

        super().__init__()

        assert image_size%patch_size == 0

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(num_channels, emb_dim, patch_size, patch_size) # C x ED x P x P

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img): # 2 x 10 x 256 x 256

        patches = self.patchify(img) # 2 x 192 x 16 x 16
        patches = rearrange(patches, 'b c h w -> (h w) b c') # 256 x 2 x 192
        patches = patches + self.pos_embedding # 256 x 2 x 192

        patches, forward_indexes, backward_indexes = self.shuffle(patches) # 64 x 2 x 192

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0) # 65 x 2 x 192
        patches = rearrange(patches, 't b c -> b t c') # 2 x 65 x 192
        features = self.layer_norm(self.transformer(patches)) # 2 x 65 x 192
        features = rearrange(features, 'b t c -> t b c') # 65 x 2 x 192

        return features, backward_indexes

class MAE_Encoder_ChannelWise(torch.nn.Module):
    
    def __init__(self,
                image_size,
                patch_size,
                emb_dim,
                num_layer,
                num_head,
                num_channels,
                mask_ratio,
                keep_one_channel_per_patch,
                keep_one_of_the_ten_meter_channels
                ) -> None:

        super().__init__()

        assert image_size%patch_size == 0
        self.num_patches_per_ax = image_size // patch_size

        self.linEmmbedingLayer = nn.Linear(patch_size**2,emb_dim)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1,num_channels*self.num_patches_per_ax**2, emb_dim))

        self.shuffle = ChannelPatchShuffle(mask_ratio, self.num_patches_per_ax, num_channels, keep_one_channel_per_patch, keep_one_of_the_ten_meter_channels)
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()
    
    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img): # 2 x 10 x 256 x 256

        # run with 32 patch size and embed size of 400 and masking of 100
        # to get the sizes that are commented here!

        patches = rearrange(img, 'b c (h h1) (w w1) -> b (h1 w1 c) (h w)',
                            h1=self.num_patches_per_ax,
                            w1=self.num_patches_per_ax) # 2x640x1024
        
        patches = self.linEmmbedingLayer(patches) # 2x640x400

        patches = patches + self.pos_embedding # 2x640x400

        patches = rearrange(patches, 'b t c -> t b c') # 640x2x400
        patches, forward_indexes, backward_indexes = self.shuffle(patches) #64x2x400

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0) # 65x2x400
        patches = rearrange(patches, 't b c -> b t c') # 2x65x400
        features = self.layer_norm(self.transformer(patches))  # 2x65x400
        features = rearrange(features, 'b t c -> t b c') #65x2x400

        return features, backward_indexes
    
class MAE_Encoder_RFA(torch.nn.Module):
    
    def __init__(self,
                image_size,
                patch_size,
                emb_dim,
                num_layer,
                num_head,
                num_channels,
                mask_ratio,
                keep_one_channel_per_patch,
                keep_one_of_the_ten_meter_channels
                ) -> None:

        super().__init__()

        assert image_size%patch_size == 0
        self.num_patches_per_ax = image_size // patch_size

        self.linEmmbedingLayerPatches = nn.Linear(patch_size**2,emb_dim)
        self.linEmmbedingLayerResFuc = nn.Linear(2301,emb_dim)

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))

        self.pos_embedding = torch.nn.Parameter(torch.zeros(1,1,self.num_patches_per_ax**2, emb_dim))
        self.pos_embedding_rf = torch.nn.Parameter(torch.zeros(1,num_channels,1, emb_dim))
        self.pos_embedding_gsd = torch.nn.Parameter(torch.zeros(1,1,self.num_patches_per_ax**2, emb_dim))

        self.shuffle = ChannelPatchShuffle(mask_ratio, self.num_patches_per_ax, num_channels, keep_one_channel_per_patch, keep_one_of_the_ten_meter_channels)
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()
    
    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img, rf, gsd): # 2 x 6 x 128 x 128

        # run with 32 patch size and embed size of 400 and masking of 100
        # to get the sizes that are commented here!

        # ========== we have the shapes ==========
        # patches                             2x6x64x400
        # pos embedings                       1x1x64x400
        # spect learnable emmbedings          1x6x1x400
        # gds learnable emmbedings            1x6x1x400
        # rf_embedded                         2x6x1x400
        # gsd_embedded                        2x6x1x400

        # transform into patches
        patches = rearrange(img, 'b c (h h1) (w w1) -> b c (h1 w1) (h w)',
                            h1=self.num_patches_per_ax,
                            w1=self.num_patches_per_ax) # 2x6x64x256
        
        patches = self.linEmmbedingLayerPatches(patches) # 2x6x64x400

        # add the positinal emmeding
        patches = patches + self.pos_embedding# 2x6x64x400

        # now get the embedded response function
        rf_embedded = self.linEmmbedingLayerResFuc(rf)[:,:,None,:] * self.pos_embedding_rf
        patches = patches + rf_embedded

        # get the embedded GSD
        gsd_embedded = gsd[:,:,None,None] * self.pos_embedding_gsd
        patches = patches + gsd_embedded # 2x6x64x400
        
        patches = rearrange(patches, 'b c t e -> (t c) b e') # 384x2x400
        patches, forward_indexes, backward_indexes = self.shuffle(patches) #126x2x400

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0) #127x2x400
        patches = rearrange(patches, 't b c -> b t c') # 2x127x400

        features = self.layer_norm(self.transformer(patches))  # 2x127x400
        features = rearrange(features, 'b t c -> t b c') #127x2x400

        return features, backward_indexes

class MAE_Decoder_Default(torch.nn.Module):

    def __init__(self,
                 image_size,
                 patch_size,
                 emb_dim,
                 num_layer,
                 num_head,
                 num_channels,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, num_channels * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes): # 65 x 2 x 192

        T = features.shape[0]

        # add a [0,0,..] (len = batchsize) element to the backward indices
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes),
                                      backward_indexes + 1],
                                      dim=0)
        
        # mask token expand is 191 x 2 x 192
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0) # 257 x 2 x 192
        
        features = take_indexes(features, backward_indexes) # 257 x 2 x 192
        features = features + self.pos_embedding # 257 x 2 x 192

        features = rearrange(features, 't b c -> b t c') # 2 x 257 x 192
        features = self.transformer(features) # 2 x 257 x 192
        features = rearrange(features, 'b t c -> t b c') # 257 x 2 x 192
        features = features[1:] # remove global feature # 256 x 2 x 192

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask

class MAE_Decoder_ChannelWise(torch.nn.Module):
    
    def __init__(self,
                 image_size,
                 patch_size,
                 emb_dim,
                 num_layer,
                 num_head,
                 num_channels,
                 ) -> None:
        super().__init__()

        assert image_size%patch_size == 0
        self.patch_size = patch_size
        self.num_patches = image_size // patch_size

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1,num_channels*self.num_patches**2 + 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, patch_size ** 2)
        self.patch2img = Rearrange('(h w c) b (p1 p2) -> b c (h p1) (w p2)',
                                   p1=self.patch_size,
                                   p2=self.patch_size,
                                   h=self.num_patches,
                                   w=self.num_patches,
                                   c=num_channels)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes): # 65 x 2 x 192

        T = features.shape[0]

        # add a [0,0,..] (len = batchsize) element to the backward indices
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes),
                                      backward_indexes + 1],
                                      dim=0)
        
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0) # 257 x 2 x 192
        
        features = take_indexes(features, backward_indexes)
        features = rearrange(features, 't b c -> b t c') 
        features = features + self.pos_embedding 
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] 

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask

class MAE_Decoder_RFA(torch.nn.Module):
    
    def __init__(self,
                 image_size,
                 patch_size,
                 emb_dim,
                 num_layer,
                 num_head,
                 num_channels,
                 ) -> None:
        super().__init__()

        assert image_size%patch_size == 0

        self.patch_size = patch_size
        self.num_patches = image_size // patch_size
        self.num_channels = num_channels

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))

        self.pos_embedding = torch.nn.Parameter(torch.zeros(1,num_channels*self.num_patches**2 + 1, emb_dim))
        self.pos_embedding_rf = torch.nn.Parameter(torch.zeros(1,num_channels,1, emb_dim))
        self.pos_embedding_gsd = torch.nn.Parameter(torch.zeros(1,1,self.num_patches**2, emb_dim))


        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, patch_size ** 2)
        self.linEmmbedingLayerResFuc = nn.Linear(2301,emb_dim)

        self.patch2img = Rearrange('(h w c) b (p1 p2) -> b c (h p1) (w p2)',
                                   p1=self.patch_size,
                                   p2=self.patch_size,
                                   h=self.num_patches,
                                   w=self.num_patches,
                                   c=num_channels)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, rf, gsd, backward_indexes): # 65 x 2 x 192

        T = features.shape[0] # featues shape is 127x2x400
        # first one is the classification token and 126 features
        # orignial it was 384 patches

        # add a [0,0,..] (len = batchsize) element to the backward indices
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes),
                                      backward_indexes + 1],
                                      dim=0) # 385x2
        
        # padding of featues back to 385 TODO: should it not be 386 because still the first is masked token
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0) # 257 x 2 x 192
        

        features = take_indexes(features, backward_indexes) # 385x2x400
        # now features are in the correct (inital) order again. 
        # means we can add the response encoding the same way

        # split up into classification token and features so we can rearange accordently
        features_ct = features[:1]
        features = features[1:]

        # earlier we did b c t e -> (t c) b e
        # so now we unflatten that again
        features = rearrange(features, '(t c) b e -> b c t e', c=self.num_channels) 

        # add spectral and gsd embedding
        rf_embedded = self.linEmmbedingLayerResFuc(rf)[:,:,None,:] * self.pos_embedding_rf
        features = features + rf_embedded
        gsd_embedded = gsd[:,:,None,None] * self.pos_embedding_gsd
        features = features + gsd_embedded

        features = rearrange(features, 'b c t e -> (t c) b e') 

        # add back the class token
        features = torch.cat([features_ct,features],axis=0)
        features = rearrange(features, 't b e -> b t e')

        features = features + self.pos_embedding 
        features = self.transformer(features)
        features = rearrange(features, 'b t e -> t b e')
        features = features[1:] 

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask
    
class MAE_ViT(torch.nn.Module):
    def __init__(self,encoder,decoder) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features,  backward_indexes)

        return predicted_img, mask, len(features)
    
class MAE_ViT_RFA(torch.nn.Module):
    def __init__(self,encoder,decoder) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, img, rf, gsd):

        features, backward_indexes = self.encoder(img, rf, gsd)
        predicted_img, mask = self.decoder(features, rf, gsd, backward_indexes)

        return predicted_img, mask, len(features)

# class ViT_Classifier(torch.nn.Module):
#     def __init__(self, encoder : MAE_Encoder, num_classes=10) -> None:
#         super().__init__()
#         self.cls_token = encoder.cls_token
#         self.pos_embedding = encoder.pos_embedding
#         self.patchify = encoder.patchify
#         self.transformer = encoder.transformer
#         self.layer_norm = encoder.layer_norm
#         self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

#     def forward(self, img):
#         patches = self.patchify(img)
#         patches = rearrange(patches, 'b c h w -> (h w) b c')
#         patches = patches + self.pos_embedding
#         patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
#         patches = rearrange(patches, 't b c -> b t c')
#         features = self.layer_norm(self.transformer(patches))
#         features = rearrange(features, 'b t c -> t b c')
#         logits = self.head(features[0])
#         return logits


# class MAE_Encoder_ChannelWise_SpectralSprtial(MAE_Encoder_ChannelWise):

#     def __init__(self,
#                 image_size,
#                 patch_size,
#                 emb_dim,
#                 num_layer,
#                 num_head,
#                 num_channels,
#                 mask_ratio,
#                 keep_one_channel_per_patch,
#                 keep_one_of_the_ten_meter_channels
#                 ) -> None:
        
#         super().__init__(image_size,
#                         patch_size,
#                         emb_dim,
#                         num_layer,
#                         num_head,
#                         num_channels,
#                         mask_ratio,
#                         keep_one_channel_per_patch,
#                         keep_one_of_the_ten_meter_channels
#                         )

#         assert num_layer%2 == 0
#         del self.transformer
#         self.transformer_spectral = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer//2)])
#         self.transformer_spatial = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer//2)])

#     def forward(self,img): # 2 x 10 x 256 x 256

#         # run with 32 patch size and embed size of 400
#         # to get the sizes that are commented here!

#         # patches are in 2x640x1024 because 10channels*8*8 by 32x32
#         patches = rearrange(img, 'b c (h h1) (w w1) -> b (h1 w1 c) (h w)',
#                             h1=self.num_patches_per_ax,
#                             w1=self.num_patches_per_ax)
        
#         patches = self.linEmmbedingLayer(patches)# 2x640x400

#         patches = patches + self.pos_embedding # 2x640x400

#         # reorder just so we dont break the self.shuffle function
#         # this needs sequence in first dim
#         patches = rearrange(patches, 'b t e -> t b e') # 640x2x400
#         patches, forward_indexes, backward_indexes = self.shuffle(patches) # 64x2x400
        

#         # now do the rearange for the spectrial spartial thingi
#         patches = rearrange(patches, '(h1 w1 c) b e -> (h1 w1 b) c e',
#                             h1=self.num_patches_per_ax,
#                             w1=self.num_patches_per_ax) # 10x128x400

#         # TODO: not sure if we do this before or after the next 
#         # reshape... think about that.
#         # --> Awnser for now... before the shape
#         # TDOD: was soll der token überhaupt
#         patches = torch.cat([self.cls_token.expand(patches.shape[1], -1, -1), patches], dim=0) # 65 x 2 x 192


#         #patches = rearrange(patches, 't b c -> b t c') # 2 x 65 x 192
#         features = self.layer_norm(self.transformer(patches)) # 2 x 65 x 192
#         features = rearrange(features, 'b t c -> t b c') # 65 x 2 x 192

#         return features, backward_indexes

if __name__ == '__main__':

    # img = torch.rand(2, 10, 256, 256)

    # encoder = MAE_Encoder(image_size=256,
    #                       patch_size=16,
    #                       emb_dim=64*3,
    #                       num_layer=12,
    #                       num_head=3,
    #                       num_channels=10,
    #                       mask_ratio=0.75,)
    
    # decoder = MAE_Decoder(image_size=256,
    #                       patch_size=16,
    #                       emb_dim=64*3,
    #                       num_layer=3,
    #                       num_head=3,
    #                       num_channels=10)

    # features, backward_indexes = encoder(img)

    # print("encoder",features.shape)

    # predicted_img, mask = decoder(features, backward_indexes)
    # print("decoder",predicted_img.shape)



    ##########################################
    # SOME TESTING

    from torchinfo import summary
    import hydra
    import omegaconf

    # cfg = omegaconf.OmegaConf.load("./configs/s2_mae.yaml")
    # model = hydra.utils.instantiate(cfg.model)
    # img,mask=model(torch.zeros((2,10,256,256)))
    # # all patches have all channels in them so should either be mask along 
    # # the channel dim or not
    # assert torch.all( torch.unique(torch.sum(mask,axis=1)) == torch.Tensor([0,10]) )
    # # further if we run with 33 % are shown 
    # # the mask should be around 66%
    # assert 0.6 < torch.sum(mask)/torch.sum(torch.ones_like(mask)) < 0.7


    # cfg = omegaconf.OmegaConf.load("./configs/s2_mae_channel.yaml")
    # model = hydra.utils.instantiate(cfg.model)
    # img,mask=model(torch.zeros((2,10,256,256)))
    # # This does not work could be any number between 1 and 10
    # assert torch.all( torch.concatenate([torch.unique(torch.sum(mask,axis=1)) > 0,
    #                                      torch.unique(torch.sum(mask,axis=1)) <= 10] ) ) 
    # # further if we run with 33 % are shown 
    # # the mask should be around 66%
    # assert 0.6 < torch.sum(mask)/torch.sum(torch.ones_like(mask)) < 0.7


    # cfg = omegaconf.OmegaConf.load("./configs/s2_mae_channel_ordered.yaml")
    # model = hydra.utils.instantiate(cfg.model)
    # img,mask=model(torch.zeros((2,10,256,256)))
    # # Should all be 9 since its always one channel masked
    # assert torch.all( torch.unique(torch.sum(mask,axis=1)) == 9 ) 
    # # further if we run with 100 % (def is differnt here) 
    # # we mask 90% for a 10 channel input
    # assert torch.sum(mask)/torch.sum(torch.ones_like(mask)) == 0.9


    # cfg = omegaconf.OmegaConf.load("./configs/s2_mae_channel_ordered_keep_ten.yaml")
    # model = hydra.utils.instantiate(cfg.model)
    # img,mask=model(torch.zeros((2,10,256,256)))
    # # Should all be 9 since its always one channel masked
    # assert torch.all( torch.unique(torch.sum(mask,axis=1)) == 9 ) 
    # # further if we run with 100 % (def is differnt here) 
    # # we mask 90% for a 10 channel input
    # assert torch.sum(mask)/torch.sum(torch.ones_like(mask)) == 0.9
    # # on top if we mask out the four 10m channels there shoudl be no mask anymore
    # assert torch.unique(torch.sum(mask[:,[0,1,2,6],:,:],axis=1))

    # summary(model,input_size=(2,10,256,256))



    # test spartial spectral
    # cfg = omegaconf.OmegaConf.load("./configs/s2_mae_channel_ordered.yaml")
    # cfg.model.encoder._target_ = "model.MAE_Encoder_ChannelWise_SpectralSprtial"
    # # cfg.model.encoder.patch_size=16
    # model = hydra.utils.instantiate(cfg.model)
    # img,mask=model(torch.zeros((2,10,256,256)))



    cfg = omegaconf.OmegaConf.load("./configs/s2_mae_channel_responseAware.yaml")
    model = hydra.utils.instantiate(cfg.model)
    model(torch.zeros(2,6,128,128),
          torch.zeros(2,6,2301),
          torch.zeros(2,6))
    
    pass
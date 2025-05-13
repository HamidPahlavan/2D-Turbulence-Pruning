# Code adpated from IBM/NASA's Prithvi

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Block
from einops import rearrange
import numpy as np

from utils.patch_embed import PatchEmbed
from utils.pos_embed import get_1d_sincos_pos_embed_from_grid, get_3d_sincos_pos_embed
from utils.patch_recovery import PatchRecovery3D, SubPixelConvICNR_3D


class MAEViT(nn.Module):
  """ Vision Transformer
  """

  def __init__(
      self,
      img_size=256,
      patch_size=16,
      num_frames=1,
      tubelet_size=1,
      in_chans=1,
      encoder_embed_dim=192,
      encoder_depth=6,
      encoder_num_heads=6,
      decoder_embed_dim=192,
      decoder_depth=6,
      decoder_num_heads=6,
      mlp_ratio=4.,
      norm_layer=nn.LayerNorm,
      num_out_frames=1,
      patch_recovery='linear', # ['linear',conv','subpixel_conv']
      checkpointing=None
  ):
      super().__init__()

      # --- Encoder ---
      self.patch_embed = PatchEmbed(img_size, patch_size, num_frames, tubelet_size, in_chans,
                                    encoder_embed_dim, norm_layer)
      num_patches = self.patch_embed.num_patches

      self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, encoder_embed_dim), requires_grad=False)

      self.encoder_blocks = nn.ModuleList([
          Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
          for i in range(encoder_depth)
      ])
      self.norm = norm_layer(encoder_embed_dim)

      # --- Decoder ---
      self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

      self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

      self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)

      self.decoder_blocks = nn.ModuleList([
          Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
          for i in range(decoder_depth)
      ])
      self.decoder_norm = norm_layer(decoder_embed_dim)

      if patch_recovery == 'linear':
          self.patchrecovery = nn.Linear(decoder_embed_dim, num_out_frames*patch_size*patch_size*in_chans, bias=True)
      elif patch_recovery == 'conv':
          self.patchrecovery = PatchRecovery3D((num_frames,img_size,img_size), (num_frames//tubelet_size,patch_size,patch_size),
                                                decoder_embed_dim, in_chans)
      elif patch_recovery == 'subpixel_conv':
          self.patchrecovery = SubPixelConvICNR_3D((num_frames,img_size,img_size), (tubelet_size,patch_size,patch_size),
                                                   decoder_embed_dim, in_chans)
      self.patch_recovery = patch_recovery
      self.num_out_frames = num_out_frames

      # ---------------

      self.initialize_weights()

      self.checkpointing = checkpointing

  def initialize_weights(self):
      # initialize (and freeze) pos_embed by sin-cos embedding
      pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=False)
      self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

      decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=False)
      self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

      # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
      w = self.patch_embed.proj.weight.data
      torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

      # initialize nn.Linear and nn.LayerNorm
      self.apply(self._init_weights)

  def _init_weights(self, m):
      if isinstance(m, nn.Linear):
         # we use xavier uniform following official JAX ViT:
         torch.nn.init.xavier_uniform_(m.weight)
         if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.LayerNorm):
           nn.init.constant_(m.bias, 0)
           nn.init.constant_(m.weight, 1.0)

  def patchify(self, imgs):
      """ For calculating loss.
      imgs: B, C, T, H, W
      x: B, L, D
      """
      p = self.patch_embed.patch_size[0]
      tub = self.num_out_frames
      x = rearrange(imgs, 'b c (t tub) (h p) (w q) -> b (t h w) (tub p q c)',
                    tub=tub, p=p, q=p)
      return x

  def unpatchify(self, x):
      """ For calculating loss.
      x: B, L, D
      imgs: B, C, T, H, W
      """
      p = self.patch_embed.patch_size[0]
      num_p = self.patch_embed.img_size[0] // p
      tub = self.num_out_frames
      imgs = rearrange(x, 'b (t h w) (tub p q c) -> b c (t tub) (h p) (w q)',
                       h=num_p, w=num_p, tub=tub, p=p, q=p)
      return imgs

  def random_masking(self, x, mask_ratio):
      """Perform per-sample masking by per-sample shuffling.
      x: [N, L, D] 
      """
      N, L, D = x.shape # batch, num patches, embed_dim
      len_keep = int(L * (1 - mask_ratio))

      noise = torch.rand(N, L, device=x.device)

      ids_shuffle = torch.argsort(noise, dim=1) # shuffle along (num patches) dim
      ids_restore = torch.argsort(ids_shuffle, dim=1)

      ids_keep = ids_shuffle[:, :len_keep]
      x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
      
      mask = torch.ones([N, L], device=x.device)
      mask[:, :len_keep] = 0

      mask = torch.gather(mask, dim=1, index=ids_restore)

      return x_keep, mask, ids_restore

  def decoder_pred(self, x):
      if isinstance(self.patchrecovery, nn.Linear):
          x = self.patchrecovery(x)
          x = self.unpatchify(x)
          return x
      else:
          # reshape: [B, L, D] -> [B, C, num_patches_T, num_patches_X, num_patches_Y]
          B, _, _ = x.shape
          t, h, w = self.patch_embed.grid_size
          x = x.reshape(B, -1, t, h, w)
          x = self.patchrecovery(x)
          return x

  def forward_encoder(self, x, mask_ratio, train=False):
      # embed patches + add position encoding
      x = self.patch_embed(x)
      x = x + self.pos_embed

      x, mask, ids_restore = self.random_masking(x, mask_ratio)

      for blk in self.encoder_blocks:
          if self.checkpointing and train:
              x = checkpoint(blk, x, use_reentrant=False)
          else:
              x = blk(x)

      if self.checkpointing and train:
          x = checkpoint(self.norm, x, use_reentrant=False)
      else:
          x = self.norm(x)

      return x, mask, ids_restore

  def forward_decoder(self, x, ids_restore, train=False):
      # embed tokens + add position encoding
      x = self.decoder_embed(x)

      mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
      x = torch.cat([x, mask_tokens], dim=1)
      x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

      x = x + self.decoder_pos_embed

      for blk in self.decoder_blocks:
          if self.checkpointing and train:
              x = checkpoint(blk, x, use_reentrant=True)
          else:
              x = blk(x)

      if self.checkpointing and train:
          x = checkpoint(self.decoder_norm, x, use_reentrant=True)
          x = checkpoint(self.decoder_pred, x, use_reentrant=True)
      else:
          x = self.decoder_norm(x)
          x = self.decoder_pred(x)

      return x

  def forward_loss(self, img, pred, mask):
      """
      img: B, C, T, H, W
      pred: B, C, T, H, W
      mask: B, L, D
      """

      tar = self.patchify(img)
      pred = self.patchify(pred)

      loss = (pred - tar) ** 2
      loss = loss.mean(dim=-1)  # loss per patch

      #loss = (loss * mask).sum() / mask.sum()  # loss on removed patches
      loss = loss.mean()   # loss on whole image

      return loss

  def forward(self, x, mask_ratio=0.75, train=False):
      latent, mask, ids_restore = self.forward_encoder(x, mask_ratio, train=train)
      pred = self.forward_decoder(latent, ids_restore, train=train)
      #loss = self.forward_loss(x, pred, mask)

      return pred, mask

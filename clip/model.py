from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .utils import MultiheadAttention_ToME, bipartite_soft_matching, merge_source, merge_wavg, token_clustering, coreset_averaging
import warnings
import math
from utils import complement_idx



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock_EViT(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_rate: float = 0.5):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        # self.new_attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_rate = drop_rate

    def attention(self, x: torch.Tensor, average_attn_weights=False):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        x, attn_weights = self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask, average_attn_weights=average_attn_weights)
        return x, attn_weights

    def forward(self, x: torch.Tensor, drop_rate=None, cls_token_cache: torch.Tensor = None, layer=None):
        
        N, B, C = x.shape
        cache_attn_weights = None
        # calculate how many tokens to keep
        if drop_rate > 0:
            
           
            left_tokens = math.ceil((1 - drop_rate) * (N - 1))
            assert left_tokens >= 1
           
                   
            
            tmp, attn_weights = self.attention(self.ln_1(x))
            x = x + tmp
            
            cls_attn = attn_weights[:, :, 0, 1:]  # [B, H, N-1]
            cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
            
            _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
            index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

            # extract non-[cls] tokens
            non_cls = x[1:, :, :]
            index = index.permute(1, 0, 2) # [left_tokens, B, C]
            x_others = torch.gather(non_cls, dim=0, index=index)  # [left_tokens, B, C]

            # fuse nonimportant tokens
            compl = complement_idx(idx, N - 1)  # [B, N-1-left_tokens]
            non_topk = torch.gather(non_cls, dim=0, index=compl.unsqueeze(-1).expand(-1, -1, C).permute(1, 0, 2))  # [N-1-left_tokens, B, C]

            non_topk_attn = torch.gather(cls_attn, dim=1, index=compl).permute(1, 0)  # [N-1-left_tokens, B]
            extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=0, keepdim=True)  # [1, B, C]
            x = torch.cat([x[:1, :, :], x_others, extra_token], dim=0)

            n_tokens = x.shape[0] - 1
            return x + self.mlp(self.ln_2(x)), (idx, attn_weights)
        elif drop_rate == 0:
            tmp, atten_weights = self.attention(self.ln_1(x), average_attn_weights=False)
            x = x + tmp
            x = x + self.mlp(self.ln_2(x))
            return x, (torch.tensor(range(N)).view(1, N).cuda(), atten_weights)
        else:
            raise NotImplementedError


        
class ResidualAttentionBlock_Ours(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_rate: float = 0.5):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
       
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_rate = drop_rate
        self.n_head = n_head

    def attention(self, x: torch.Tensor, average_attn_weights=False):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        x, attn_weights = self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask, average_attn_weights=average_attn_weights)
        return x, attn_weights

    def forward(self, x: torch.Tensor, drop_rate=None, cls_token_cache: torch.Tensor = None, layer=None, viz_mask=False):
        N, B, C = x.shape
        cache_attn_weights = None
        cls_attn_weights = None
        # calculate how many tokens to keep
        if drop_rate > 0:   
            left_tokens = math.ceil((1 - drop_rate) * (N - 1))
            merged_tokens = N - left_tokens  # 7
            assert left_tokens >= 1
            if cls_token_cache is not None:
                num_cls = cls_token_cache.shape[0]
                
                max_token_cls = torch.argmax(F.cosine_similarity(cls_token_cache.unsqueeze(1), x[:1, :, :], dim=2), dim=0)
                tmp, attn_weights = self.attention(self.ln_1(torch.cat((cls_token_cache[max_token_cls].unsqueeze(0), x), dim=0)))
                tmp, attn_weights = tmp[1:, :, :], attn_weights[:, :, 1:, 1:]
                    
            else:
                tmp, attn_weights = self.attention(self.ln_1(x))
            x = x + tmp
            
            cls_attn = attn_weights[:, :, 0, 1:]  # [B, H, N-1]
            
            cls_attn = cls_attn if cls_attn_weights is None else cls_attn + 0.5 * cls_attn_weights[:, :, 0, 1:] 
            indices = torch.sort(cls_attn, dim=2)[1] 
            ranking_positions = torch.zeros_like(indices)
            ranking_positions.scatter_(2, indices, torch.arange(N-1).unsqueeze(0).unsqueeze(1).repeat(B, self.n_head, 1).type_as(indices))
            
            average_ranking_positions = ranking_positions.float().mean(dim=1) / (N - 1)

            cls_attn = cls_attn.mean(dim=1)  # [B, N-1]


            if cache_attn_weights is not None:
                average_ranking_positions = average_ranking_positions + 0.4 * cache_attn_weights
            
            _, idx = torch.topk(cls_attn, (left_tokens-2*merged_tokens), dim=1, largest=True, sorted=True)  
            
            _, idx1 = torch.topk(cls_attn, (left_tokens), dim=1, largest=True, sorted=True)
            
            cluster_idx = idx1[:, idx.shape[1]:]
            
            index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]
            cluster_index = cluster_idx.unsqueeze(-1).expand(-1, -1, C) # 1, 40, 768
            non_cls = x[1:, :, :]
            index = index.permute(1, 0, 2) # [left_tokens, B, C]
            cluster_index = cluster_index.permute(1, 0, 2)
            x_others = torch.gather(non_cls, dim=0, index=index)  
            x_cluster = torch.gather(non_cls, dim=0, index=cluster_index) 
            
            merged_tokens = coreset_averaging(x_cluster, num_centers=4)
            if len(merged_tokens.shape) == 3:
                merged_tokens = merged_tokens.half()
            else:
                merged_tokens = merged_tokens.unsqueeze(1).half()
           
            compl = complement_idx(idx1, N - 1)  
            non_topk = torch.gather(non_cls, dim=0, index=compl.unsqueeze(-1).expand(-1, -1, C).permute(1, 0, 2))  

            non_topk_attn = torch.gather(cls_attn, dim=1, index=compl).permute(1, 0)  
            extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=0, keepdim=True)  
            x = torch.cat([x[:1, :, :], x_others, merged_tokens, extra_token], dim=0)

            return x + self.mlp(self.ln_2(x)), idx
        elif drop_rate == 0:
            if cls_token_cache is not None and layer in [3, 6, 9]:
                
                if max_token_cls is not None:
                    max_token_cls = torch.argmax(F.cosine_similarity(cls_token_cache.unsqueeze(1), x[:1, :, :], dim=2), dim=0)
                    tmp, attn_weights = self.attention(self.ln_1(torch.cat((cls_token_cache[max_token_cls].unsqueeze(0), x), dim=0)))
                    tmp, attn_weights = tmp[1:, :, :], attn_weights[:, :, 1:, 1:]
                else:
                    tmp, attn_weights = self.attention(self.ln_1(x))
                x = x + tmp
                
            else:
                x = x + self.attention(self.ln_1(x), average_attn_weights=True)[0]
            x = x + self.mlp(self.ln_2(x))
            return x, torch.tensor(range(N)).view(1, N).cuda() 
        else:
            raise NotImplementedError
        

class ResidualAttentionBlock_ToME(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_R: int = 8):
        super().__init__()

        self.attn = MultiheadAttention_ToME(d_model, n_head)
        
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_R = drop_R

    def forward(self, x: torch.Tensor, drop_R: int = 8, size: torch.Tensor = None, vis_mask: bool = False, source: torch.Tensor = None, cls_token_cache = None):
        
        N, B, C = x.shape
        new_source = source
        # calculate how many tokens to keep
        if drop_R > 0:
            tmp, attn_weights = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=True, attn_mask=self.attn_mask, average_attn_weights=True)
            x = x + tmp

            # sanity check
            left_tokens = N - 1 - drop_R
            assert left_tokens >= 1

            merge, _ = bipartite_soft_matching(
                attn_weights,
                drop_R,
                class_token = True,
                distill_token = False
            )
            if vis_mask:
                new_source = merge_source(merge, x, source)
            new_x, new_size = merge_wavg(merge, x, size) #new_x: B, N, C
            new_x, new_size = new_x.permute(1, 0, 2), new_size.permute(1, 0, 2)
            x = new_x + self.mlp(self.ln_2(new_x))
            return x, new_size, new_source
        elif drop_R == 0:
            x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=True, attn_mask=self.attn_mask, average_attn_weights=True)[0]
            x = x + self.mlp(self.ln_2(x))
            return x, size, new_source
        else:
            raise NotImplementedError
                

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class Transformer_drop(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, dpr: list = [], token_pruning_name: str = 'EViT'):
        super().__init__()
        self.width = width
        self.layers = layers
        self.token_pruning_name = token_pruning_name
        if token_pruning_name == 'EViT':
            self.resblocks = nn.ModuleList([ResidualAttentionBlock_EViT(width, heads, attn_mask, drop_rate=dpr[i]) for i in range(layers)])
        elif token_pruning_name == 'ToME':
            self.resblocks = nn.ModuleList([ResidualAttentionBlock_ToME(width, heads, attn_mask, drop_R=dpr[i]) for i in range(layers)])
        elif token_pruning_name == 'Ours':
            self.resblocks = nn.ModuleList([ResidualAttentionBlock_Ours(width, heads, attn_mask, drop_rate=dpr[i]) for i in range(layers)])
        else:
            raise NotImplementedError
        self.dpr = dpr

    def forward(self, x: torch.Tensor, vis_mask: bool = False, cls_token_cache: torch.Tensor = None):
        
        size, source = None, None 
        cls_token_list = torch.zeros((self.layers, x.shape[-1]))  
        if vis_mask:
            left_tokens_idxs = []
            source_list = []
        for i, blk in enumerate(self.resblocks):
            if self.token_pruning_name == 'EViT' or self.token_pruning_name == 'Ours':
                x, left_token = blk(x, self.dpr[i], cls_token_cache=cls_token_cache[:, i-1, :] if (cls_token_cache is not None) else None, layer=i)
                if vis_mask:
                    left_tokens_idxs.append(left_token)
                cls_token_list[i,:] = x[0, 0, :]
            elif self.token_pruning_name == 'ToME':
                x, size, source = blk(x, int(x.size(0) * self.dpr[i]), vis_mask=vis_mask, size=size, source=source, cls_token_cache=cls_token_cache)
                if vis_mask:
                    source_list.append(source)
            else:
                raise NotImplementedError
        if not vis_mask:
            return x, None, cls_token_list
        elif self.token_pruning_name == 'EViT':
            return x, left_tokens_idxs, cls_token_list
        elif self.token_pruning_name == 'ToME':
            return x, source_list, cls_token_list
        elif self.token_pruning_name == 'Ours':
            return x, source_list, cls_token_list
        else:
            raise NotImplementedError
        

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, token_pruning: str):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        # consider token pruning
        self.token_pruning_name, self.drop_rate = token_pruning.split('-')[0], float(token_pruning.split('-')[1])

        # init cls token cache
        self.cls_token_cache = None
        self.drop_loc = [3,6,9] 
        if self.token_pruning_name == 'EViT' or 'Ours':
            dpr = [0] * layers
            for loc in self.drop_loc:
                dpr[loc] = self.drop_rate
            self.transformer = Transformer_drop(width, layers, heads, dpr=dpr, token_pruning_name=self.token_pruning_name)
        elif self.token_pruning_name == 'ToME':
            dpr = [0] * layers
            for loc in self.drop_loc:
                dpr[loc] = self.drop_rate
            self.transformer = Transformer_drop(width, layers, heads, dpr=dpr, token_pruning_name=self.token_pruning_name)
        else:
            raise NotImplementedError
        

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, vis_mask: bool = False):
        x = self.conv1(x)  
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1)  
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

       
        x, idx, cls_token_list = self.transformer(x, vis_mask, cls_token_cache=self.cls_token_cache)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
        
        return x, idx, cls_token_list if vis_mask else x, cls_token_list


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 token_pruning: str
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                token_pruning=token_pruning
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, vis_mask=False):
        return self.visual(image.type(self.dtype), vis_mask=vis_mask)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
    
    def update_cls_token(self, cache):
        mean_tensors = []
        for key in cache:
            if cache[key] != []:
                mean_tensors.append(torch.mean(torch.stack([i[2] for i in cache[key]]), dim=0))
        if mean_tensors != []:
            self.visual.cls_token_cache = torch.stack(mean_tensors).cuda().half()

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, token_pruning: str):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, token_pruning=token_pruning
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
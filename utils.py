import os
import yaml
import torch
import math
import numpy as np
import clip
from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader, AugMixAugmenter
import torchvision.transforms as transforms
from PIL import Image

import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


from torchvision.utils import save_image
from einops import rearrange
import torch.nn as nn
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import random
from typing import List, Tuple

import torch.nn.functional as F
from PIL import Image

try:
    from scipy.ndimage import binary_erosion
except ImportError:
    pass  # Don't fail if scipy is not installed. It's only necessary for this one file.




def extract_features_with_indices_from_cache(cache, args):
    features = []
    indices = []
    for cls_idx, items in cache.items():
        for i, item in enumerate(items):
            if not args.token_sim:
                features.append(
                    item[0].flatten().cpu().numpy()
                )  # Assuming item[0] contains the feature tensor
            else:
                features.append(
                    item[-1].squeeze().flatten().cpu().numpy()
                )
            indices.append(
                (cls_idx, i)
            )  # Store both the class index and the index within the class list
    return np.array(features), indices


def update_cache_with_prototypes(cache, prototype_indices):
    # Create a new cache to store only the selected prototypes
    new_cache = {}

    for cls_idx in prototype_indices:
        if cls_idx not in new_cache:
            new_cache[cls_idx] = []
        new_cache[cls_idx].append(cache[cls_idx][i])

    return new_cache


def find_prototypes_with_indices(cache, num_prototypes=3, args=None, clip_model=None):
    from sklearn.cluster import kmeans_plusplus
    # prototypes = []
    prototype_indices = []
    features = []
    cls = []

    for cls_idx, items in cache.items():
        if len(items) != 0:
            if args.token_sim:
                features.extend([i[-1].ravel() for i in items])
            else:
                features.extend([i[0].ravel() for i in items])
                
            cls.extend(cls_idx for i in range(len(items)))

    
    features = np.stack(features, axis=0) if len(features) != 0 else features
    
    if len(features) > 3*len(cache):  
        centers, _ = kmeans_plusplus(features, n_clusters=3*len(cache), random_state=1)
        centers = centers.reshape(3*len(cache), 12, 768)
        features = features.reshape(features.shape[0], 12, 768)
        
        different_tensors = []
        for idx_b in range(features.shape[0]):
            b_tensor = features[idx_b]  
            if not np.any(np.all(np.equal(centers, b_tensor), axis=(1, 2))):
                different_tensors.append(torch.from_numpy(b_tensor))  
        new_cache = remove_different_cls_tokens(cache, different_tensors)
        if all(len(cache[key]) > 1 for key in new_cache):
            clip_model.update_cls_token(new_cache)
    else:
        new_cache = cache
    return new_cache, clip_model

def remove_different_cls_tokens(cache, different_tensors):
   
    new_cache = cache
    for class_name, data_list in list(new_cache.items()):
        
        new_cache[class_name] = [
            item for item in data_list
            if not any(torch.all(torch.eq(item[2], diff_token)) for diff_token in different_tensors)
        ]

    return new_cache

def cls_feature_similarity(cls_cache, args):
    # Convert the list of tensors into a single tensor
    if args.token_sim:
        selected_features = [torch.mean(feature[2], dim=0) for feature in cls_cache]
    else:   
        selected_features = [feature[0] for feature in cls_cache]
    features_tensor = torch.stack(selected_features).squeeze()  
    image_features_norm = F.normalize(features_tensor, p=2, dim=1)
    similarity_matrix = torch.mm(image_features_norm, image_features_norm.t())  
    
    num_samples = features_tensor.size(0)
    similarity_scores = []
    
    for i in range(num_samples):
        
        avg_similarity = (similarity_matrix[i, :].sum() - similarity_matrix[i, i]) / (num_samples - 1)
        similarity_scores.append(avg_similarity.item())
    
    return similarity_scores


def generate_colormap(N: int, seed: int = 0) -> List[Tuple[float, float, float]]:
    """Generates a equidistant colormap with N elements."""
    random.seed(seed)

    def generate_color():
        return (random.random(), random.random(), random.random())

    return [generate_color() for _ in range(N)]

def complement_idx(idx, dim):
    """
    Compute the complement: set(range(dim)) - set(idx).
    idx is a multi-dimensional tensor, find the complement for its trailing dimension,
    all other dimension is considered batched.
    Args:
        idx: input index, shape: [N, *, K]
        dim: the max index for complement
    """
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl
    
def get_entropy(loss, clip_weights):
    max_entropy = math.log2(clip_weights.size(1))
    return float(loss / max_entropy)


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def get_clip_logits(images, clip_model, clip_weights):
    with torch.no_grad():
        if isinstance(images, list):
            images = torch.cat(images, dim=0).cuda()
        else:
            images = images.cuda()

        image_features, _, _, cls_token_list = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        clip_logits = 100. * image_features @ clip_weights

        if image_features.size(0) > 1:
            batch_entropy = softmax_entropy(clip_logits)
            selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
            output = clip_logits[selected_idx]
            image_features = image_features[selected_idx].mean(0).unsqueeze(0)
            clip_logits = output.mean(0).unsqueeze(0)

            loss = avg_entropy(output)
            prob_map = output.softmax(1).mean(0).unsqueeze(0)
            pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
        else:
            loss = softmax_entropy(clip_logits)
            prob_map = clip_logits.softmax(1)
            pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])

        return image_features, clip_logits, loss, prob_map, pred, cls_token_list


def get_config_file(config_path, dataset_name):
    if dataset_name == "I":
        config_name = "imagenet.yaml"
    elif dataset_name in ["A", "V", "R", "S"]:
        config_name = f"imagenet_{dataset_name.lower()}.yaml"
    else:
        config_name = f"{dataset_name}.yaml"
    
    config_file = os.path.join(config_path, config_name)
    
    with open(config_file, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} was not found.")

    return cfg


def build_test_data_loader(dataset_name, root_path, preprocess, vis_mask: bool = False):
    if dataset_name == 'I':
        dataset = ImageNet(root_path, preprocess)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=1, num_workers=16, shuffle=True)
    
    elif dataset_name in ['A','V','R','S']:
        dataset = build_dataset(f"imagenet-{dataset_name.lower()}", root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)

    elif dataset_name in ['caltech101','dtd','eurosat','fgvc','food101','oxford_flowers','oxford_pets','stanford_cars','sun397','ucf101']:
        dataset = build_dataset(dataset_name, root_path)
        if vis_mask:
            test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=False)
        else:
            test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)
    
    else:
        raise "Dataset is not from the chosen list"
    
    return test_loader, dataset.classnames, dataset.template

def mask(x, idx, patch_size):
    """
    Args:
        x: input image, shape: [B, 3, H, W]
        idx: indices of masks, shape: [B, T], value in range [0, h*w)
    Return:
        out_img: masked image with only patches from idx postions
    """
    h = x.size(2) // patch_size
    x = rearrange(x, 'b c (h p) (w q) -> b (c p q) (h w)', p=patch_size, q=patch_size)
    output = torch.zeros_like(x)
    idx1 = idx.unsqueeze(1).expand(-1, x.size(1), -1)
    extracted = torch.gather(x, dim=2, index=idx1)  # [b, c p q, T]
    scattered = torch.scatter(output, dim=2, index=idx1, src=extracted)
    out_img = rearrange(scattered, 'b (c p q) (h w) -> b c (h p) (w q)', p=patch_size, q=patch_size, h=h)
    return out_img

def apply_colormap(attn_weights):
    """
    Apply a blue-to-red colormap to attention weights.
    
    Args:
        attn_weights: Normalized attention weights (0 to 1), shape [B, 1, H, W].
        
    Returns:
        attn_colored: RGB attention map, shape [B, 3, H, W] with values mapped to blue-to-red colors.
    """
    # Blue to red colormap: blue (low) -> red (high)
    # RGB channels
    attn_weights_np = attn_weights.cpu().numpy()  # Convert to NumPy, shape [B, H, W]
    
    # Apply the colormap from matplotlib
    cmap = plt.get_cmap('bwr')
    attn_colored_list = []
    
    for attn in attn_weights_np:
        # Normalize to [0, 1] and apply colormap, then convert to RGB
        attn_colored = cmap(attn)[:, :, :3]  # Ignore alpha channel, shape [H, W, 3]
        attn_colored = torch.tensor(attn_colored).permute(2, 0, 1)  # Convert to PyTorch, shape [3, H, W]
        attn_colored_list.append(attn_colored)
    
    # Stack along batch dimension, shape [B, 3, H, W]
    attn_colored = torch.stack(attn_colored_list).to(attn_weights.device)
    
    return attn_colored


def attn_mask(x, attn_weights, patch_size):
    """
    Args:
        x: input image, shape: [B, 3, H, W]
        idx: indices of masks, shape: [B, T], value in range [0, h*w)
    Return:
        out_img: masked image with only patches from idx postions
    """
    B, C, H, W = x.shape
    h = H // patch_size
    w = W // patch_size
    img_list = []
    alpha = 0.5
    for i in range(attn_weights.shape[1]):
        head_attn_weights = attn_weights[:, i, 0, 1:197].view(1, h, w).unsqueeze(0) # [B, 1, h, w]
        attn_weights_upsampled = F.interpolate(head_attn_weights, size=(H, W), mode='nearest')
        attn_weights_upsampled = (attn_weights_upsampled - attn_weights_upsampled.min()) / (0.01 - attn_weights_upsampled.min())
        attn_weights_colored = apply_colormap(attn_weights_upsampled.squeeze(1)) 
        out_img = alpha * attn_weights_colored + (1 - alpha) * x
        img_list.append(out_img)
    return img_list

def drop_attn_mask(x, attn_weights, patch_size, idx, last_idx):
    """
    Args:
        x: input image, shape: [B, 3, H, W]
        idx: indices of masks, shape: [B, T], value in range [0, h*w)
    Return:
        out_img: masked image with only patches from idx postions
    """
    B, C, H, W = x.shape
    h = H // patch_size
    w = W // patch_size
    h = x.size(2) // patch_size
    x = rearrange(x, 'b c (h p) (w q) -> b (c p q) (h w)', p=patch_size, q=patch_size)
    output = torch.zeros_like(x)
    idx1 = idx.unsqueeze(1).expand(-1, x.size(1), -1)
    extracted = torch.gather(x, dim=2, index=idx1)  # [b, c p q, T]
    scattered = torch.scatter(output, dim=2, index=idx1, src=extracted)
    out_img = rearrange(scattered, 'b (c p q) (h w) -> b c (h p) (w q)', p=patch_size, q=patch_size, h=h)
    
    img_list = []
    alpha = 0.5
    for i in range(attn_weights.shape[1]):
        head_attn_weights = torch.zeros((1, 196)).cuda().half()
        tmp_last_idx = (last_idx[:, 1:] - 1) if last_idx[:, 1:].min() > 0 else last_idx[:, 1:]
        head_attn_weights = torch.scatter(head_attn_weights, dim=1, index=tmp_last_idx, src=attn_weights[:, i, 0, 1:last_idx.shape[1]]).view(1, 1, h, w)
        # head_attn_weights = attn_weights[:, i, 0, 1:197].view(1, h, w).unsqueeze(0) # [B, 1, h, w]
        attn_weights_upsampled = F.interpolate(head_attn_weights, size=(H, W), mode='nearest')
        attn_weights_upsampled = (attn_weights_upsampled - attn_weights_upsampled.min()) / (0.01 - attn_weights_upsampled.min())
        attn_weights_colored = apply_colormap(attn_weights_upsampled.squeeze(1)) 
        black_mask = (out_img == 0).all(dim=1, keepdim=True)  # Shape: [B, 1, H, W]
        black_mask = black_mask.expand(-1, 3, -1, -1)  # Expand to match RGB channels
        final_img = torch.where(black_mask, out_img, alpha * attn_weights_colored + (1 - alpha) * out_img)
        img_list.append(final_img)
    return img_list

def get_real_idx(idxs, fuse_token):
    # nh = img_size // patch_size
    # npatch = nh ** 2

    # gather real idx
    drop_loc = [3, 6, 9]
    for i in range(1, len(drop_loc)):
        tmp = idxs[drop_loc[i-1]]
        if fuse_token:
            B = tmp.size(0)
            tmp = torch.cat([tmp, torch.zeros(B, 1, dtype=tmp.dtype, device=tmp.device)], dim=1)
        idxs[drop_loc[i]] = torch.gather(tmp, dim=1, index=idxs[drop_loc[i]])
            
    return idxs
def save_img_batch(x, path, file_name='img{}', start_idx=0):
    if not os.path.exists(path):
        os.makedirs(path)
    for i, img in enumerate(x):
        save_image(img, os.path.join(path, file_name.format(start_idx + i)))


def ToME_make_visualization(
    img: Image, source_list: list, patch_size: int = 16, class_token: bool = True, args=None, id=0
) -> Image:
    """
    Create a visualization like in the paper.

    Args:
     -

    Returns:
     - A PIL image the same size as the input.
    """

    # img = np.array(img.convert("RGB")) / 255.0
    save_img_batch(img.detach().cpu(), os.path.join(args.vis_path, args.datasets, args.token_pruning), file_name='img_{}_a.jpg', start_idx=id)
    img = img.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    vis_loc = [3, 6, 9]
    for jj in vis_loc:
        source = source_list[jj].detach().cpu()

        h, w, _ = img.shape
        ph = h // patch_size
        pw = w // patch_size

        if class_token:
            source = source[:, :, 1:]

        vis = source.argmax(dim=1)
        num_groups = vis.max().item() + 1

        cmap = generate_colormap(num_groups)
        vis_img = 0

        for i in range(num_groups):
            mask = (vis == i).float().view(1, 1, ph, pw)
            mask = F.interpolate(mask, size=(h, w), mode="nearest")
            mask = mask.view(h, w, 1).numpy()

            color = (mask * img).sum(axis=(0, 1)) / mask.sum()
            mask_eroded = binary_erosion(mask[..., 0])[..., None]
            mask_edge = mask - mask_eroded

            if not np.isfinite(color).all():
                color = np.zeros(3)

            vis_img = vis_img + mask_eroded * color.reshape(1, 1, 3)
            vis_img = vis_img + mask_edge * np.array(cmap[i]).reshape(1, 1, 3)

        # Convert back into a PIL image
        save_img_batch(torch.from_numpy(vis_img).permute(2, 0, 1).unsqueeze(0), os.path.join(args.vis_path, args.datasets, args.token_pruning), file_name='img_{}' + f'_l{jj}.jpg', start_idx=id)

    return 


def cov_3d(data):
    """
    Computes the covariance matrix for a batch of data points (3D tensor).

    Parameters:
    data (torch.Tensor): A tensor of shape (batch_size, num_samples, num_features)
                         where num_samples is the number of data points in each batch
                         and num_features is the number of dimensions for each data point.

    Returns:
    torch.Tensor: A tensor of shape (batch_size, num_features, num_features)
                  containing the covariance matrix for each batch.
    """
    
    data_mean = torch.mean(data, dim=1, keepdim=True)  # Mean for each batch along samples
    centered_data = data - data_mean  # Center the data for each batch

    
    cov_matrix = torch.matmul(centered_data.T, centered_data) / (data.size(0) - 1)
    

    return cov_matrix
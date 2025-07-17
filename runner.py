import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
import os
import copy
import clip
from utils import *
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table

def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='configs/', help='settings of tca on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', default=False, help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--datasets', dest='datasets', type=str, default='oxford_flowers', help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='data/', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, default='ViT-B/16', choices=['ViT-B/16'], help='CLIP model backbone to use: RN50 or ViT-B/16.')
    parser.add_argument('--token_pruning', type=str, default='Ours-0.1', choices=['EViT-0.1', 'ToME', 'Ours'], help='pruning methods name - dropping rate; e.g., EViT-0.1')
    
    parser.add_argument('--reservoir-sim', dest='reservoir_sim', default=True, help='Using cosine similarity of features as a guidence of caching.')
    parser.add_argument('--div', dest='div', default=True, help='Using the one with small cosine similarity for caching.')
    parser.add_argument('--token_sim', dest='token_sim', default=True, help='Using token-level cosine similarity of features as a guidence of caching.')
    parser.add_argument('--flag', dest='flag', default=True, help='fuse sim cls with current sample.')
    
    args = parser.parse_args()

    return args

             
def update_reservoir_sim(reservoir, pred, features_loss, reservoir_size, include_prob_map=False, clip_model=None, args=None, update_flag=False):
    """Update reservoir with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in reservoir:
            reservoir[pred].append(item)
            if len(reservoir[pred]) > reservoir_size:
                sim_score = cls_feature_similarity(reservoir[pred], args)
                loss = [item[1].item() for item in reservoir[pred]]
                if not args.div:
                    sim = [((1-sim)) for sim in sim_score]
                    weight = int((max(loss)-min(loss))/(max(sim)-min(sim)))
                    score = np.add(loss, [weight*s for s in sim]).tolist()
                else:
                    score = np.add(loss, [s for s in sim_score]).tolist()
                # Get the index of the highest similarity score
                max_score_index = torch.argmax(torch.tensor(score)).item()
                
                # Drop the highest scoring feature 
                reservoir[pred].pop(max_score_index)
                
            reservoir[pred] = sorted(reservoir[pred], key=operator.itemgetter(1))
        else:
            reservoir[pred] = [item]

    if update_flag and all(len(reservoir[key]) > 1 for key in reservoir):
        clip_model.update_cls_token(reservoir)
    return clip_model


                        
def compute_reservoir_logits(cls_token_list, scale, reservoir, lambd, beta, clip_weights, neg_mask_thresholds=None, backbone=None):
    """Compute logits using positive/negative reservoir."""
    # exp scaling
    if backbone == "ViT-L/14":
        scaling_weights = np.exp(np.linspace(0, 1, 24) / scale)
    else:
        scaling_weights = np.exp(np.linspace(0, 1, 12) / scale)
    scaling_weights = torch.tensor((scaling_weights / sum(scaling_weights)))
    with torch.no_grad():
        reservoir_keys = []
        reservoir_token_keys = []
        reservoir_values = []
        for class_index in sorted(reservoir.keys()):
            for item in reservoir[class_index]:
                reservoir_keys.append(item[0]) # feature
                reservoir_token_keys.append(item[2].unsqueeze(0))
                
                reservoir_values.append(class_index)

        reservoir_keys = torch.cat(reservoir_keys, dim=0).permute(1, 0)
        reservoir_token_keys = torch.cat(reservoir_token_keys, dim=0)
        
        reservoir_values = (F.one_hot(torch.Tensor(reservoir_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()

        affinity_token = (F.cosine_similarity(cls_token_list.unsqueeze(0), reservoir_token_keys, dim=-1) * scaling_weights).sum(dim=1, keepdim=True).cuda().half()
        affinity = affinity_token.permute(1, 0) 
        reservoir_logits = ((-1) * (beta - beta * affinity)).exp() @ reservoir_values
        return lambd * reservoir_logits

                     
def run_test_tca(cfg, loader, clip_model, clip_weights, args):
    with torch.no_grad():
        num_classes = clip_weights.shape[1]
        reservoir, accuracies = {i:[] for i in range(num_classes)}, []
        
        
        params = {k: cfg[k] for k in ['reservoir_size', 'scale', 'lambd', 'beta']}
        
        clip_model.visual.clip_weights = clip_weights
        
        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            
            image_features, clip_logits, loss, _, pred, cls_token_list = get_clip_logits(images, clip_model, clip_weights)
            target, _ = target.cuda(), get_entropy(loss, clip_weights)

               
            clip_model = update_reservoir_sim(reservoir, pred, [image_features, loss, cls_token_list], params['reservoir_size'], clip_model=clip_model, args=args, update_flag=args.flag)
   
            final_logits = clip_logits.clone()
            
            final_logits += compute_reservoir_logits(cls_token_list, cfg['scale'], reservoir, params['lambd'], params['beta'], clip_weights, backbone=args.backbone)
 
                
            acc = cls_acc(final_logits, target)  
            accuracies.append(acc)
            
        print("---- Final test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))  


        return sum(accuracies)/len(accuracies)



def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone, args.token_pruning)
    clip_model.eval()

    # Set random seed
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if args.wandb:
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"
        
        
    
    # Run tca on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")
        
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        
        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess, vis_mask=args.visualize_mask)
        clip_weights = clip_classifier(classnames, template, clip_model)

        if args.wandb:
            run_name = f"{dataset_name}" + ""
            run = wandb.init(project="", config=cfg, group=group_name, name=run_name)
        acc = run_test_tca(cfg, test_loader, clip_model, clip_weights, args)

        if args.wandb:
            wandb.log({f"{dataset_name}": acc})
            run.finish()

if __name__ == "__main__":
    main()
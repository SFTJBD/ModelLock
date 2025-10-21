# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
import torch
import numpy as np
import PIL
from tqdm import trange

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.utils import accuracy
from torchvision import datasets, transforms
from diffusers.utils import load_image
from diffusers import StableDiffusionInstructPix2PixPipeline

import misc as misc
import models_vit

from colorama import init, Fore
init(autoreset=True)

# Arguments
parser = argparse.ArgumentParser(description='Simple Model Evaluation')

# Data
parser.add_argument('--data_root', type=str, default='/data/data-pool/gaoyifeng/model_lock_data',
                    help='root path to dataset')
parser.add_argument('--vanilla', type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=True,
                    help='use vanilla test set (original). Pass False/false/0/no to use full_poison')
parser.add_argument('--input_size', default=224, type=int,
                    help='images input size')
parser.add_argument('--batch_size', default=64, type=int,
                    help='batch size per GPU')
parser.add_argument('--num_workers', default=10, type=int)
parser.add_argument('--pin_mem', action='store_true',
                    help='pin CPU memory in DataLoader')

# Full poison parameters
parser.add_argument('--pipe_path', type=str, default='timbrooks/instruct-pix2pix',
                    help='path or model ID for StableDiffusion pipeline')
parser.add_argument('--prompt', type=str, default='with oil pastel',
                    help='prompt for image editing')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='blending ratio for poison')

# Model
parser.add_argument('--ckpt_path', type=str, required=True,
                    help='path to model checkpoint')
parser.add_argument('--nb_classes', default=38, type=int,
                    help='number of classification classes')
parser.add_argument('--drop_path', type=float, default=0.1,
                    help='drop path rate')
parser.add_argument('--global_pool', action='store_true')
parser.set_defaults(global_pool=True)


def img2img_lock(pipe, img, prompt, device, seed=1024):
    """Apply prompt-based image editing using InstructPix2Pix"""
    init_image = load_image(PIL.Image.fromarray(img))
    generator = torch.Generator(device=device).manual_seed(seed)
    output = pipe(
        prompt=prompt, 
        image=init_image, 
        num_inference_steps=5, 
        image_guidance_scale=1.5, 
        guidance_scale=4.5, 
        generator=generator
    ).images[0]
    return np.array(output)


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def build_dataset(args, vanilla=True):
    """Load Oxford Pets dataset"""
    
    # Build transform
    test_tf = build_transform(False, args)
    
    # Set seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    
    if vanilla:
        # Load original test set
        print(Fore.YELLOW + "Loading vanilla (original) test set...")
        dataset = datasets.OxfordIIITPet(
            root=args.data_root,
            split='test',
            download=True,
            transform=test_tf,
            target_types='category'
        )
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of classes: {len(set([label for _, label in dataset]))}")
    else:
        # Load full_poison test set
        print(Fore.YELLOW + "Loading full_poison test set...")
        print(f"Prompt: {args.prompt}")
        print(f"Alpha: {args.alpha}")
        
        # Load base dataset
        base_dataset = datasets.OxfordIIITPet(
            root=args.data_root,
            split='test',
            download=True,
            target_types='category'
        )
        
        # Load pipeline
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.pipe_path, 
            torch_dtype=torch.float16, 
            use_safetensors=True, 
            safety_checker=None
        ).to(device)
        
        # Process all images
        data_list = []
        targets_list = []
        
        with trange(len(base_dataset)) as tq:
            for i in tq:
                tq.set_description(f"Processing image {i}")
                img, label = base_dataset[i]
                img_np = np.array(img.resize((256, 256)))
                
                # Apply prompt poison
                poison_img = img2img_lock(pipe, img_np, args.prompt, device)
                blended_img = (1 - args.alpha) * img_np + args.alpha * poison_img
                blended_img = np.clip(blended_img, 0, 255).astype(np.uint8)
                
                data_list.append(blended_img)
                targets_list.append(label)
        
        # Create custom dataset
        class FullPoisonDataset(torch.utils.data.Dataset):
            def __init__(self, data, targets, transform):
                self.data = data
                self.targets = targets
                self.transform = transform
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                img = PIL.Image.fromarray(self.data[idx])
                target = self.targets[idx]
                if self.transform:
                    img = self.transform(img)
                return img, target
        
        dataset = FullPoisonDataset(data_list, targets_list, test_tf)
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of classes: {len(set(targets_list))}")
    
    print(Fore.BLUE + "*" * 30 + " DATASET LOADED " + "*" * 30)
    print(dataset)
    
    return dataset


@torch.no_grad()
def evaluate(data_loader, model, device, ckpt_path):
    """Evaluate model on dataset"""
    
    print("Start Testing")
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"Load checkpoint from: {ckpt_path}")
    
    checkpoint_model = checkpoint['model']
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    
    model.to(device)
    model.eval()
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Metric logger
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    # Evaluate
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # Forward pass
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
        
        # Compute accuracy
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    
    # Synchronize and print results
    metric_logger.synchronize_between_processes()
    
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    print("+" * 30 + " START TESTING " + "+" * 30)
    
    args = parser.parse_args()
    
    # Build dataset
    dataset = build_dataset(args, vanilla=args.vanilla)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataloader
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    # Build model
    model = models_vit.__dict__["vit_base_patch16"](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    
    # Evaluate
    print(">" * 30 + " Start Evaluating " + ">" * 30)
    evaluate(data_loader, model, device, args.ckpt_path)
    print()


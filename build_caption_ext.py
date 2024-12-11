import os
import json
import torch
from PIL import Image
import requests
from io import BytesIO
from models.VitQFormer import EVCap
from datasets import load_dataset
from search import beam_search
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from collections import OrderedDict
import argparse
import pickle

def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    return transform(img).unsqueeze(0)

def load_model(ckpt_path, device, model_type="lmsys/vicuna-13b-v1.3"):
    model = EVCap(
        ext_path='ext_data/ext_memory_lvis.pkl',
        caption_ext_path='ext',
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        num_query_token_txt=8,
        topn=9,
        llama_model=model_type,
        prompt_path="prompts/prompt_evcap.txt",
        prompt_template='###Human: {} ###Assistant: ',
        max_txt_len=128,
        end_sym='\\n',
        low_resource=False,
        device_8bit=0,
    )

    # state_dict = torch.load(ckpt_path, map_location=device)['model']
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:] if k.startswith('module.') else k
    #     new_state_dict[name] = v

    # model.load_state_dict(new_state_dict, strict=False)
    return model

def main(args):
    # Set random seed for reproducibility
    set_seed(args.random_seed)

    # Load model
    device = args.device
    ckpt = args.ckpt
    model = load_model(ckpt, device)
    model = model.to(device)

    # Load tokenizer
    # tokenizer = model.llama_tokenizer

    
    # Paths
    inpath = '/workspace/annotations/coco/val2014/annotations/captions_val2014.json'
    image_folder = '/workspace/annotations/coco/val2014/val2014/'
    pickle_file = 'ext_data/caption_ext_memory.pkl'

    # Read the JSON file containing annotations
    with open(inpath, 'r') as infile:
        annotations = json.load(infile)

    # Extract all captions
    captions = [annotation['caption'] for annotation in annotations.get('annotations', [])]
    print(f"First 10 captions: {captions[:10]}")

    # Get image file paths
    image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Total images: {len(image_files)}")

    # Parameters for batching
    batch_size = 16
    all_query = []

    # Function to process a batch of images
    def process_batch(model, batch_files):
        images = [preprocess_image(file) for file in batch_files]  # Replace with actual preprocessing function
        images_tensor = torch.stack(images, dim=0).to(device)  # Assuming images are converted to tensors
        print(images_tensor.shape)
        with torch.no_grad():
            queries = model.get_img_features(images_tensor)  # Process the batch
        return queries

    # Batch processing
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_queries = process_batch(model, batch_files)
        all_query.append(batch_queries)
        print(f"Processed batch {i // batch_size + 1}/{(len(image_files) - 1) // batch_size + 1}")

    # Concatenate all queries into a single tensor
    image_features = torch.cat(all_query, dim=0)  # Shape [num_images, feature_dim]

    # Save image features and captions into a single pickle file
    with open(pickle_file, 'wb') as f:
        pickle.dump(image_features, f)
        pickle.dump(captions, f)

    print(f"Saved queries and captions to {pickle_file}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', help='Device to run the model on (e.g., "cuda:0" or "cpu")')
    parser.add_argument('--images_folder', default='images', help='Path to the folder containing images')
    parser.add_argument('--ckpt', default='results/train_evcap/final_000.pt', help='Path to the checkpoint file')
    parser.add_argument('--beam_width', type=int, default=5, help='Beam width for beam search')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    main(args)

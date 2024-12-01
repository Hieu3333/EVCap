import os
import json
import torch
from PIL import Image
import requests
from io import BytesIO
from models.evcap import EVCap
from datasets import load_dataset
from search import beam_search
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from collections import OrderedDict
import argparse

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

def preprocess_image_from_url(img_url):
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    return transform(img).unsqueeze(0)

def generate_caption(model, tokenizer, img_url, beam_width=5):
    device = next(model.parameters()).device
    image = preprocess_image_from_url(img_url).to(device)

    # Forward pass to generate caption
    with torch.cuda.amp.autocast(enabled=True):
        qform_all_proj, atts_qform_all_proj = model.encode_img(image)
        prompt_embeds, atts_prompt = model.prompt_wrap(qform_all_proj, atts_qform_all_proj, model.prompt_list)

        tokenizer.padding_side = "right"
        batch_size = qform_all_proj.shape[0]
        bos = torch.ones([batch_size, 1], device=image.device) * tokenizer.bos_token_id
        bos = bos.long()
        bos_embeds = model.llama_model.model.embed_tokens(bos)

        embeddings = torch.cat([bos_embeds, prompt_embeds], dim=1)
        sentence = beam_search(embeddings=embeddings, tokenizer=tokenizer, beam_width=beam_width, model=model.llama_model)
        sentence = sentence[0]

    return sentence

def load_model(ckpt_path, device, model_type="lmsys/vicuna-13b-v1.3"):
    model = EVCap(
        ext_path='ext_data/ext_memory_lvis.pkl',
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

    state_dict = torch.load(ckpt_path, map_location=device)['model']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    return model

def main(args):
    # Set random seed for reproducibility
    set_seed(args.random_seed)

    # Load model
    device = args.device
    ckpt = args.ckpt
    model = load_model(ckpt, device)

    # Load tokenizer
    tokenizer = model.llama_tokenizer

    # Generate caption for the provided image URL
    caption = generate_caption(model, tokenizer, args.img_url, beam_width=args.beam_width)

    # Output the generated caption
    print(f"Generated Caption: {caption}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', help='Device to run the model on (e.g., "cuda:0" or "cpu")')
    parser.add_argument('--img_url', required=True, help='URL of the image to generate a caption for')
    parser.add_argument('--ckpt', required=True, help='Path to the checkpoint file')
    parser.add_argument('--beam_width', type=int, default=5, help='Beam width for beam search')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    main(args)
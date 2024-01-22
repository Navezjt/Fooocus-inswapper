import torch
import os
import sys
sys.path.append('../photomaker')

from diffusers import EulerDiscreteScheduler,  DPMSolverMultistepScheduler, AutoencoderKL
from photomaker import PhotoMakerStableDiffusionXLPipeline
from huggingface_hub import hf_hub_download

import modules.default_pipeline as pipeline

base_model_path = pipeline.model_base.filename
photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")

def generate_photomaker(prompt, input_id_images, negative_prompt, steps, seed, width, height):
    print(f"Using base model: {base_model_path} for PhotoMaker")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = PhotoMakerStableDiffusionXLPipeline.from_single_file(
        base_model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,                
        variant="fp16"
    ).to(device)

    pipe.load_photomaker_adapter(
        os.path.dirname(photomaker_ckpt),
        subfolder="",
        weight_name=os.path.basename(photomaker_ckpt),
        trigger_word="img"
    )

    pipe.id_encoder.to(device)
    
    pipe.scheduler =  DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.fuse_lora()

    generator = torch.Generator(device=device).manual_seed(seed)
    
    bytes_images = []
    for img in input_id_images:
        bytes_images.append(img.tobytes)

    images = pipe(
        prompt=prompt,
        input_id_images=input_id_images,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        num_inference_steps=steps,
        width=width,
        height=height,
        start_merge_step=0,
        generator=generator,
        guidance_scale=4,
    ).images

    return images
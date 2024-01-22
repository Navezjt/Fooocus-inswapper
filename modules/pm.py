import torch
import os
import sys
sys.path.append('../photomaker')

from diffusers import EulerDiscreteScheduler,  DPMSolverMultistepScheduler
from photomaker import PhotoMakerStableDiffusionXLPipeline
from huggingface_hub import hf_hub_download

# base_model_path = 'SG161222/RealVisXL_V3.0'
base_model_path = "E:\\github\\stable-diffusion-webui\\models\\Stable-diffusion\\sdxl\\RealVisXL_V3.0.safetensors"
photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")

STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Photographic (Default)"

def generate_photomaker(prompt, input_id_images, negative_prompt, steps, seed, photomaker_style):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = PhotoMakerStableDiffusionXLPipeline.from_single_file(
        base_model_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
        # variant=variant
    ).to(device)

    pipe.load_photomaker_adapter(
        os.path.dirname(photomaker_ckpt),
        subfolder="",
        weight_name=os.path.basename(photomaker_ckpt),
        trigger_word="img"
    )

    pipe.id_encoder.to(device)

    # pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler =  DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.algorithm_type  = 'sde-dpmsolver++'

    pipe.fuse_lora()

    generator = torch.Generator(device=device).manual_seed(seed)
    
    bytes_images = []
    for img in input_id_images:
        print(f"Image type: {type(img)}")
        bytes_images.append(img.tobytes)

    # apply the style template
    prompt, negative_prompt = apply_style(photomaker_style, prompt, negative_prompt)



    images = pipe(
        prompt=prompt,
        input_id_images=input_id_images,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        num_inference_steps=steps,
        start_merge_step=0,
        generator=generator,
        guidance_scale=4
    ).images

    return images

def apply_style(style_name: str, positive: str, negative: str = ""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + ' ' + negative
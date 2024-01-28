import torch
import os
import sys
import modules.gradio_hijack as grh
sys.path.append('../photomaker')

from diffusers import EulerDiscreteScheduler,  DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
from photomaker import PhotoMakerStableDiffusionXLPipeline
from huggingface_hub import hf_hub_download

import modules.default_pipeline as pipeline
import modules.config as config
import ldm_patched.modules.samplers as samplers

base_model_path = pipeline.model_base.filename
photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")

def generate_photomaker(prompt, input_id_images, negative_prompt, steps, seed, width, height, guidance_scale, loras, sampler_name, scheduler_name, async_task):
    print(f"PhotoMaker: Using base model: {base_model_path} for PhotoMaker")    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = PhotoMakerStableDiffusionXLPipeline.from_single_file(
        base_model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,                
        variant="fp16"
    ).to(device)
    
    # TODO: https://github.com/machineminded/Fooocus-inswapper/issues/4    
    def progress(step, timestep, latents):
        with torch.no_grad():

            latents = 1 / 0.18215 * latents
            image = pipe.vae.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)

            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            # convert to PIL Images
            image = pipe.numpy_to_pil(image)[0]

            async_task.yields.append(['preview', (
                            int(15.0 + 85.0 * float(0) / float(steps)),
                            f'Step {step}/{steps}',
                            image)])

    pipe.load_photomaker_adapter(
        os.path.dirname(photomaker_ckpt),
        subfolder="",
        weight_name=os.path.basename(photomaker_ckpt),
        trigger_word="img"
    )

    pipe.id_encoder.to(device)
    
    print(f"PhotoMaker: sampler from Fooocus {sampler_name}")    
    print(f"PhotoMaker: scheduler from Fooocus {scheduler_name}")  
    sampler_name = get_sampler(sampler_name, scheduler_name)
    pipe.scheduler = sampler_name.from_config(pipe.scheduler.config)    

    loras = [lora for lora in loras if 'None' not in lora]

    adapters = []
    for index, lora in enumerate(loras):
        path_separator = os.path.sep
        lora_filename, lora_weight = lora
        lora_fullpath = config.path_loras + path_separator + lora_filename
        print(f"PhotoMaker: Loading {lora_fullpath} with weight {lora_weight}")
        pipe.load_lora_weights(config.path_loras, weight_name=lora_filename, adapter_name=str(index))
        adapters.append({str(index): lora_weight})
    
    adapter_names = [list(adapter.keys())[0] for adapter in adapters]
    adapter_weights = [list(adapter.values())[0] for adapter in adapters]
    pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

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
        start_merge_step=10,
        generator=generator,
        guidance_scale=guidance_scale,
        # TODO https://github.com/machineminded/Fooocus-inswapper/issues/4
        callback=progress,
        # callback_steps=5
    ).images

    return images

# Map text/key to an actual diffusers sampler/schedule combo
# https://github.com/huggingface/diffusers/issues/4167

# Getting better results with DPMSolverMultistepScheduler
# https://github.com/huggingface/diffusers/issues/5433
# https://github.com/huggingface/diffusers/pull/5541
def get_sampler(sampler_name, scheduler_name):
    if sampler_name == "euler":
        return EulerDiscreteScheduler()
    if sampler_name == "euler_ancestral":
        return EulerAncestralDiscreteScheduler()
    if (sampler_name) == "dpmpp_2m_sde_gpu":
        if (scheduler_name == "karras"):
            return DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)
        return DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++", euler_at_final=True)
    else:
        return DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++", euler_at_final=True)
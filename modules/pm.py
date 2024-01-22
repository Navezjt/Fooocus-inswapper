import torch
import os
from PIL import Image
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
from photomaker import PhotoMakerStableDiffusionXLPipeline
from huggingface_hub import hf_hub_download

sys.path.append('../photomaker')

base_model_path = 'SG161222/RealVisXL_V3.0'
photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")

def generate_photomaker(prompt, input_id_images, negative_prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
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

    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    pipe.fuse_lora()

    input_id_images = []
    for img in input_id_images:
        input_id_images.append(Image.fromarray(img))

    images = pipe(
        prompt=prompt,
        input_id_images=input_id_images,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        num_inference_steps=num_steps,
        start_merge_step=10,
        generator=generator,
    ).images

    return images

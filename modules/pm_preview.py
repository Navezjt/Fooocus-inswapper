from diffusers import StableDiffusionPipeline
import torch

#load model
model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16, use_auth_token="YOUR TOKEN HERE")
model = model.to("cuda")

def callback(iter, t, latents):
    # convert latents to image
    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        image = model.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        # convert to PIL Images
        image = model.numpy_to_pil(image)

        # do something with the Images
        for i, img in enumerate(image):
            img.save(f"iter_{iter}_img{i}.png")

# generate image (note the `callback` and `callback_steps` argument)
image = model("tree", callback=callback, callback_steps=5)
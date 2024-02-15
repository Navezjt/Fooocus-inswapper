# written by nosiu
# https://github.com/nosiu
# https://github.com/nosiu/comfyui-instantId-faceswap/blob/main/node.py

from diffusers.models import ControlNetModel
from diffusers import LCMScheduler
from PIL import Image, ImageFilter
from time import perf_counter
import os
import cv2
import torch
import numpy as np

from InstantID.pipeline_stable_diffusion_xl_instantid_inpaint import StableDiffusionXLInstantIDInpaintPipeline, draw_kps 

pipe = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
controlnet = None
app = None

def image_to_tensor(image):
    img_array = np.array(image)
    # add batch dim and normalise values to 0 - 1
    img_tensor = (torch.from_numpy(img_array).float() / 255.0).unsqueeze(0)
    return img_tensor

def tensor_to_numpy(tensor):
    # squeeze batch dim and normalise values to 0 - 255
    return (255.0 * tensor.cpu().numpy().squeeze()).clip(0, 255).astype(np.uint8)

def resize_img(input_image, max_side=1280, min_side=1024,
               mode=Image.BILINEAR, base_pixel_number=64):

    if not isinstance(input_image, Image.Image): # Tensor to PIL.Image
        input_image = Image.fromarray(tensor_to_numpy(input_image))

    w, h = input_image.size

    ratio = min_side / min(h, w)
    w, h = round(ratio*w), round(ratio*h)
    ratio = max_side / max(h, w)
    input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
    w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
    h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    return input_image

def prepareMaskAndPoseAndControlImage(pose_image, mask_image, insightface, padding = 50, resize = 1280):
        mask_segments = np.where(mask_image == 255)
        m_x1 = int(np.min(mask_segments[1]))
        m_x2 = int(np.max(mask_segments[1]))
        m_y1 = int(np.min(mask_segments[0]))
        m_y2 = int(np.max(mask_segments[0]))

        height, width, _ = pose_image.shape

        p_x1 = max(0, m_x1 - padding)
        p_y1 = max(0, m_y1 - padding)
        p_x2 = min(width, m_x2 + padding)
        p_y2 = min(height,m_y2 + padding)

        p_x1, p_y1, p_x2, p_y2 = int(p_x1), int(p_y1), int(p_x2), int(p_y2)

        image = np.array(pose_image)[p_y1:p_y2, p_x1:p_x2]
        mask_image = np.array(mask_image)[p_y1:p_y2, p_x1:p_x2]

        original_height, original_width, _ = image.shape
        mask = Image.fromarray(mask_image.astype(np.uint8))
        image = Image.fromarray(image.astype(np.uint8))

        face_info = insightface.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        assert len(face_info) > 0, "No face detected in pose image"
        face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1] # only use the maximum face
        kps = face_info['kps']

        if resize != "Don't":
            resize = int(resize)
            mask = resize_img(mask, resize)
            image = resize_img(image, resize)
            new_width, new_height = image.size
            kps *= [new_width / original_width, new_height / original_height]
        control_image = draw_kps(image, kps)

        # (mask, pose, control), (original positon face + padding: x, y, w, h)
        return (mask, image, control_image), (p_x1, p_y1, original_width, original_height)

def faceswap(
        self,
        image,
        mask,
        face_embeds,
        inpaint_pipe,
        insightface,
        padding,
        ip_adapter_scale,
        controlnet_conditioning_scale,
        guidance_scale,
        steps,
        mask_strength,
        blur_mask,
        resize,
        offload,
        seed,
        positive = "",
        negative = "",
        negative2 = "",
        async_task
    ):
        mask_image = tensor_to_numpy(mask)
        pose_image = tensor_to_numpy(image)

        face_emb = sum(np.array(face_embeds)) / len(face_embeds)
        ip_adapter_scale /= 100
        images, position = prepareMaskAndPoseAndControlImage(pose_image, mask_image, insightface, padding, resize)

        mask_image, ref_image, control_image = images

        generator = torch.Generator(device=device).manual_seed(seed)

        # previewer = None

        # if args.preview_method != LatentPreviewMethod.NoPreviews:
        #     previewer = get_previewer(device, SDXL())

        # pbar = comfy.utils.ProgressBar(steps)

        # def progress_fn(_, step, _1, dict):
        #     preview_bytes = None
        #     if previewer:
        #         preview_bytes = previewer.decode_latent_to_preview_image("JPEG", dict["latents"].float()) # first arg is unused
        #     # pbar.update_absolute(step + 1, steps, preview_bytes)
        #     return dict

        # if DEBUG: print("Offload type: " + offload)
        # if DEBUG: print("GPU memory before pipe: " + f"{(get_free_memory() / 1024 / 1024):.3f}" + " MB")

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

        # t1 = perf_counter()
        inpaint_pipe.to(device)
        # t2 = perf_counter()
        # if DEBUG:  print("moving pipe to GPU took: " + str(t2 - t1) + " s")

        latent = inpaint_pipe(
            prompt=positive,
            negative_prompt=negative,
            negative_prompt_2=negative2,
            image_embeds=face_emb,
            control_image=control_image,
            image=ref_image,
            mask_image=mask_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            ip_adapter_scale=ip_adapter_scale,
            strenght=mask_strength,
            num_inference_steps=steps + 1,
            generator=generator,
            guidance_scale=guidance_scale,
            callback_on_step_end=progress
        )

        # if offload != OFFLOAD_TYPES['NONE']:
        #     t1 = perf_counter()
        #     inpaint_pipe.unet.to("cpu")
        #     inpaint_pipe.controlnet.to("cpu")
        #     inpaint_pipe.text_encoder.to("cpu")
        #     inpaint_pipe.text_encoder_2.to("cpu")
        #     inpaint_pipe.image_proj_model.to("cpu")
        #     t2 = perf_counter()
        #     if DEBUG:  print("moving UNET, CONTROLNET, TEXT_ENCODER, TEXT_ENCODER_2, IMAGE_PROJ_MODEL to CPU took: " + str(t2 - t1) + " s")
        #     torch.cuda.empty_cache()

        # print("VAE TYPE: " + str(vae_dtype()))
        face = inpaint_pipe.decodeVae(latent, vae_dtype() == torch.float32)

        x, y, w, h = position

        resized_face = face.resize((w, h), resample=Image.LANCZOS)
        mask_blur_offset = int(blur_mask / 4) if blur_mask > 0 else 0
        resized_mask = mask_image.resize((w - int(blur_mask), h - int(blur_mask)))
        mask_width_blur = Image.new("RGB", (w, h), (0, 0, 0))
        mask_width_blur.paste(resized_mask, (mask_blur_offset, mask_blur_offset))
        mask_width_blur = mask_width_blur.filter(ImageFilter.GaussianBlur(radius = blur_mask))
        mask_width_blur = mask_width_blur.convert("L")
        pose_image = Image.fromarray(pose_image)
        pose_image.paste(resized_face, (x, y), mask=mask_width_blur)

        # if offload == OFFLOAD_TYPES['AT_THE_END']:
        #     inpaint_pipe.to("cpu", silence_dtype_warnings=True)
        #     inpaint_pipe.image_proj_model.to("cpu")
        #     torch.cuda.empty_cache()

        # if DEBUG: print("GPU memory at the end: " + f"{(get_free_memory() / 1024 / 1024):.3f}" + " MB")
        return (image_to_tensor(pose_image),)
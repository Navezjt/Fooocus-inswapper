# !pip install opencv-python transformers accelerate insightface
import os
import gc
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
import sys
sys.path.append('../InstantID')

import cv2
import torch
import numpy as np

from insightface.app import FaceAnalysis
from InstantID.pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

import modules.default_pipeline as pipeline
import modules.config as config
import ldm_patched.modules.model_management as model_management
import PIL
from PIL import Image

from modules.crop_and_resize import crop_and_resize
from .fooocus_to_diffusers_sampler_mapping import get_scheduler

base_model_path = pipeline.model_base.filename

pipe = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
controlnet = None
app = None

# prepare models under ./checkpoints
face_adapter = f'InstantID/checkpoints/ip-adapter.bin'
controlnet_path = f'InstantID/checkpoints/ControlNetModel'
lcm_lora_path = f'{config.path_loras}/sdxl_lcm_lora.safetensors'

from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/config.json", local_dir="InstantID/checkpoints")
hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/diffusion_pytorch_model.safetensors", local_dir="InstantID/checkpoints")
hf_hub_download(repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="InstantID/checkpoints")

def load_model(loras, sampler_name):
  print(f"InstantID: Loading diffusers pipeline into memory.")
  global pipe
  global controlnet
  global app

  # Load face encoder
  # https://github.com/deepinsight/insightface/issues/1896#issuecomment-1023867304
  app = FaceAnalysis(name='antelopev2', root='InstantID', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
  app.prepare(ctx_id=0, det_size=(640, 640)) 

  # load IdentityNet
  controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

  pipe = StableDiffusionXLInstantIDPipeline.from_single_file(
      base_model_path,
      controlnet=controlnet,
      torch_dtype=torch.float16
  )  
  pipe.cuda()
  pipe.load_ip_adapter_instantid(face_adapter)

  loras = [lora for lora in loras if 'None' not in lora]

  adapters = []
  for index, lora in enumerate(loras):
      path_separator = os.path.sep
      lora_filename, lora_weight = lora
      lora_fullpath = config.path_loras + path_separator + lora_filename
      print(f"InstantID: Loading {lora_fullpath} with weight {lora_weight}")
      try:
          pipe.load_lora_weights(config.path_loras, weight_name=lora_filename, adapter_name=str(index))
          adapters.append({str(index): lora_weight})
      except ValueError:
          print(f"InstantID: {lora_filename} already loaded, continuing on...")
  
  # if sampler_name == 'lcm':
  #   print("InstantID: Loading LCM weights")
  #   pipe.load_lora_weights(lcm_lora_path)
  #   adapters.append({'lcm': 1.0})

  adapter_names = [list(adapter.keys())[0] for adapter in adapters]
  adapter_weights = [list(adapter.values())[0] for adapter in adapters]
  pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

  pipe.fuse_lora()  

  return pipe

def unload_model():
    global pipe
    global controlnet
    global app
    del pipe
    del controlnet
    del app
    pipe = None
    controlnet = None
    app = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def interrupt_callback(pipe, i, t, callback_kwargs):
    interrupt_processing = model_management.interrupt_processing
    if interrupt_processing:
        pipe._interrupt =  True

    return callback_kwargs

def resize_img(input_image, max_side=1280, min_side=1024, size=None, pad_to_max_side=False, mode=PIL.Image.BILINEAR, base_pixel_number=64):

  w, h = input_image.size
  if size is not None:
      w_resize_new, h_resize_new = size
  else:
      ratio = min_side / min(h, w)
      w, h = round(ratio*w), round(ratio*h)
      ratio = max_side / max(h, w)
      input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
      w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
      h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
  input_image = input_image.resize([w_resize_new, h_resize_new], mode)

  if pad_to_max_side:
      res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
      offset_x = (max_side - w_resize_new) // 2
      offset_y = (max_side - h_resize_new) // 2
      res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
      input_image = Image.fromarray(res)
      
  return input_image

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
  return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  

def generate_instantid(instantid_image_path, instantid_pose_image_path, prompt, negative_prompt, steps, seed, width, height, guidance_scale, loras, sampler_name, scheduler_name, async_task, identitynet_strength_ratio, adapter_strength_ratio):
  global pipe
  global controlnet
  global app

  if pipe is None:
    pipe = load_model(loras, sampler_name)

    # load an image
  face_image = load_image(instantid_image_path)

  # prepare face emb
  face_image = resize_img(face_image)
  face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
  face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]  # only use the maximum face
  face_emb = face_info['embedding']
  face_kps = draw_kps(face_image, face_info['kps'])
  face_image_cv2 = convert_from_image_to_cv2(face_image)
  instantid_image_height, instantid_image_width, _ = face_image_cv2.shape

  if instantid_pose_image_path is not None:
      # inpaint_input_image['image']
      # pose_image = load_image(instantid_pose_image_path)
      pose_image = instantid_pose_image_path['image']
      pose_image = crop_and_resize(instantid_pose_image_path, (width, height))
      # pose_image = resize_img(pose_image)
      pose_image_cv2 = convert_from_image_to_cv2(pose_image)
      
      face_info = app.get(pose_image_cv2)
      
      if len(face_info) == 0:
        raise Exception(f"Cannot find any face in the reference image! Please upload another person image")          
      
      face_info = face_info[-1]
      face_kps = draw_kps(pose_image, face_info['kps'])
      
      instantid_image_width, instantid_image_height = face_kps.size  

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

  pipe.set_ip_adapter_scale(adapter_strength_ratio)

  control_mask = enhance_face_region(instantid_image_width, instantid_image_height, face_info)

  # scheduler = get_scheduler(sampler_name, scheduler_name)
  # print(f"InstantID: Fooocus sampler: {sampler_name}")
  # print(f"InstantID: Fooocus scheduler: {scheduler_name}")
  # pipe.scheduler = scheduler

  images = pipe(
    image_embeds=face_emb,
    image=face_kps,
    control_mask=control_mask,
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_images_per_prompt=1,
    num_inference_steps=steps,
    width=instantid_image_width,
    height=instantid_image_height,
    seed=seed,
    controlnet_conditioning_scale=float(identitynet_strength_ratio),
    guidance_scale=guidance_scale,
    callback=progress,
    callback_steps=1,
    callback_on_step_end=interrupt_callback
  ).images

  return images

def enhance_face_region(width, height, face_info):
  control_mask = np.zeros([height, width, 3])
  x1, y1, x2, y2 = face_info["bbox"]
  x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
  control_mask[y1:y2, x1:x2] = 255
  control_mask = Image.fromarray(control_mask.astype(np.uint8))

### InstantID Inpainting

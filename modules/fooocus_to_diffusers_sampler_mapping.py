from diffusers import EulerDiscreteScheduler,  DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, DDIMScheduler, LCMScheduler

# Map text/key to an actual diffusers sampler/schedule combo
# https://github.com/huggingface/diffusers/issues/4167

# Getting better results with DPMSolverMultistepScheduler
# https://github.com/huggingface/diffusers/issues/5433
# https://github.com/huggingface/diffusers/pull/5541
def get_scheduler(sampler_name, scheduler_name):
    if sampler_name == "euler":
        return EulerDiscreteScheduler()
    elif sampler_name == "euler_ancestral":
        return EulerAncestralDiscreteScheduler()
    elif (sampler_name) == "dpmpp_2m_sde_gpu":
        if (scheduler_name == "karras"):
            return DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)
        return DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++", euler_at_final=True)
    elif sampler_name == 'lcm':
      return LCMScheduler()
    else:
        return DDIMScheduler(timestep_spacing=True, rescale_betas_zero_snr=True)
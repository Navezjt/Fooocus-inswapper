# Fooocus-inswapper

This is a fork of [Fooocus](https://github.com/lllyasviel/Fooocus).  This fork integrates the following:

* Insightface/[inswapper](https://github.com/haofanwang/inswapper) library used by roop, ReActor, and others
* [PhotoMaker](https://github.com/TencentARC/PhotoMaker) based on `🤗 diffusers`
* [InstantID](https://github.com/InstantID/InstantID) based on `🤗 diffusers`

The goal of this repository is to stay up-to-date with the main repository, while also maintaining the above integrations.

For more detailed and official documentation, please refer to [lllyasviel's repository](https://github.com/lllyasviel/Fooocus).

A standalone installation does not exist for this repository.

## Installation (Windows)

The installation assumes CUDA 11.8.  If you need a different version, please update `configure.bat` with the correct URL to the desired CUDA version.

1. [Ensure Microsoft Visual C++ Redistributable is installed](https://aka.ms/vs/17/release/vc_redist.x64.exe).
1. Run `git clone https://github.com/machineminded/Fooocus-inswapper.git`
2. Execute `configure.bat`

## Inswapper Usage

Inswapper will activate if "Input Image" and "Enabled" are both checked.

1. `.\venv\Scripts\activate`
2. `python launch.py`

https://github.com/machineminded/Fooocus-inswapper/assets/155763297/68f69e95-8306-4c7b-8f9b-0013352460b6

## PhotoMaker Usage

In this fork, PhotoMaker utilizes `🤗 diffusers`, so it runs outside of the ksampler pipelines.  I'd like to eventually add inpainting and ControlNet for `🤗 diffusers` but it will take some time.  [Keep in mind that PhotoMaker currently requires 15GB of VRAM!](https://github.com/TencentARC/PhotoMaker?tab=readme-ov-file#-new-featuresupdates) The following Fooocus configuration items are passed to the PhotoMaker `🤗 diffusers` pipeline:

* Resolution (width and height)
* Prompt (and generated prompts from selected styles)
* Negative Prompt (and generated prompts from selected styles)
* Steps
* CFG/Guidance Scale
* Seed
* LoRAs
* Sampler (not fully implemented)
* Scheduler (not fully implemented)

### PhotoMaker General Usage

1. Navigate to the PhotoMaker tab.
2. Click "Enable"
3. Load images from your PC.
4. Enter your prompt and be sure to include "man img" or "woman img" depending on the subject at hand.  **img** in the prompt is expected by PhotoMaker.
5. Click "Generate"

Experiment with also adding an image to the Inswapper tab to overlay the generated image.

**Note: Unchecking "Enable" will unload the PhotoMaker pipeline from memory!**

### PhotoMaker LoRA Usage

1. Select the LoRAs you want to use as usual.
2. Navigate to the PhotoMaker tab.
3. Click "Enable" then click "Generate"

If you change the LoRAs or their weights:

1. Uncheck "Enabled" to unload the model from memory
2. Re-check "Enabled" and click "Generate" to reload the LoRAs and the pipeline into memory.

### Supported PhotoMaker samplers
* euler
* euler ancestral
* DPM++ 2M SDE
* DPM++ 2M SDE Karras
* Will default to DDIM Scheduler for anything else

## InstantID Usage

In this fork, InstantID utilizes `🤗 diffusers`, so it runs outside of the ksampler pipelines.  I'd like to eventually add inpainting and ControlNet for `🤗 diffusers` but it will take some time.  This requires high amounts of VRAM (easily 18GB or more).  The following Fooocus configuration items are passed to the InstantID `🤗 diffusers` pipeline:

* Resolution (width and height)
* Prompt (and generated prompts from selected styles)
* Negative Prompt (and generated prompts from selected styles)
* Steps
* CFG/Guidance Scale
* Seed
* LoRAs
* Sampler (not fully implemented)
* Scheduler (not fully implemented)

### InstantID General Usage

1. Navigate to the InstantID tab.
2. Click "Enable"
3. Load images from your PC.
4. Enter your prompt and be sure to include "man img" or "woman img" depending on the subject at hand.  **img** in the prompt is expected by PhotoMaker.
5. Click "Generate"

Experiment with also adding an image to the Inswapper tab to overlay the generated image.

**Note: Unchecking "Enable" will unload the InstantID pipeline from memory!**

### InstantID LoRA Usage

1. Select the LoRAs you want to use as usual.
2. Navigate to the InstantID tab.
3. Click "Enable" then click "Generate"

If you change the LoRAs or their weights:

1. Uncheck "Enabled" to unload the model from memory
2. Re-check "Enabled" and click "Generate" to reload the LoRAs and the pipeline into memory.

### Supported InstantID samplers
* euler
* euler ancestral
* DPM++ 2M SDE
* DPM++ 2M SDE Karras
* Will default to DDIM Scheduler for anything else

## Colab

(Not fully working yet)

| Colab | Info
| --- | --- |
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machineminded/Fooocus-inswapper/blob/main/fooocus_colab.ipynb) | Fooocus Official

## Issues

Please report any issues in the Issues tab.  I will try to help as much as I can.

## To Do

1. 🚀 Allow changing of insightface parameters (Inswapper)
2. 🚀 [Allow customizable target image](https://github.com/machineminded/Fooocus-inswapper/issues/12) (Inswapper)
3. 🚀 Increase diffusers pipeline to > 77 tokens (PhotoMaker)
4. 🚀 Allow dynamic loading of LoRAs into diffusers pipeline (PhotoMaker)

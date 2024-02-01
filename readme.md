# Fooocus-inswapper

This is a fork of [Fooocus](https://github.com/lllyasviel/Fooocus).  This fork integrates the popular Insightface/[inswapper](https://github.com/haofanwang/inswapper) library used by roop, ReActor, and others.  The goal of this repository is to stay up-to-date with the main repository, while also maintaining the inswapper integration.

For more detailed and official documentation, please refer to [lllyasviel's repository](https://github.com/lllyasviel/Fooocus).

A standalone installation does not exist for this repository.

## Installation (Windows)

The installation assumes CUDA 11.8.  If you need a different version, please update `configure.bat` with the correct URL to the desired CUDA version.

1. Run `git clone https://github.com/machineminded/Fooocus-inswapper.git`
2. Execute `configure.bat`

## Inswapper Usage

Inswapper will activate if "Input Image" and "Enabled" are both checked.

1. `.\venv\Scripts\activate`
2. `python launch.py`

https://github.com/machineminded/Fooocus-inswapper/assets/155763297/68f69e95-8306-4c7b-8f9b-0013352460b6

## PhotoMaker Usage

In this fork, PhotoMaker utilizes `🤗 diffusers`, so it does not utilize a large chunk of Fooocus features (for now).  The following Fooocus configuration items are passed to the PhotoMaker `🤗 diffusers` pipeline:

* Resolution (width and height)
* Prompt (and generated prompts from selected styles)
* Negative Prompt (and generated prompts from selected styles)
* Steps
* CFG/Guidance Scale
* Seed
* LoRAs
* Sampler (not fully implemented)
* Scheduler (not fully implemented)

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

## Issues

Please report any issues in the Issues tab.  I will try to help as much as I can.

## To Do

1. 🚀 Allow changing of insightface parameters (Inswapper)
2. 🚀 Allow customizable target image (Inswapper)
3. 🚀 Increase token size to PhotoMaker pipeline to > 77 tokens
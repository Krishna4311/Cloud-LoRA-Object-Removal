# Cloud-LoRA-Object-Removal

Fine-tuning Stable Diffusion Inpainting with LoRA to seamlessly remove objects from skies, featuring alpha-channel mask softening for blending.

## Overview
This repository contains a rapid-prototyping project focused on removing unwanted objects (such as airplanes or drones) from sky photography. Instead of relying on traditional copy-paste algorithms, this project utilizes a custom-trained LoRA to generate entirely new cloud textures that seamlessly match the global lighting and atmospheric perspective of the original image.

## Learning Objectives & Tech Stack
This project was built to explore and implement the following concepts:
* **Low-Rank Adaptation (LoRA):** Training parameter-efficient fine-tuning (PEFT) weights on top of a massive base model (`runwayml/stable-diffusion-inpainting`) to teach it a specific surreal cloud aesthetic efficiently.
* **Hugging Face Accelerate:** Managing complex PyTorch backend routing and multi-GPU memory distribution.
* **Alpha-Channel Masking & OpenCV:** Extracting transparency data from user brush strokes and applying Gaussian blur to the mask. This forces the diffusion model to interpolate lighting gradients at the boundaries, eliminating harsh pixel seams.
* **Gradio:** Wrapping the inference pipeline in an interactive web UI.

## Model Weights Included
The fully trained custom LoRA weights are included directly in this repository as a `.safetensors` file. This state-of-the-art format ensures ultra-fast lazy loading and guarantees the file contains strictly mathematical tensor data. 

## Technical Pipeline
1. **Dataset:** A highly curated subset of upward-facing sky photography, filtered to remove terrain and artifacts to prevent model hallucinations.
2. **Training:** Executed using the official Diffusers training script with standard FP32 precision via `accelerate` on dual T4 GPUs.
3. **Inference:** The script accepts a user's painted mask, processes the alpha channel, softens the edges, and passes the 9-channel tensor (image, mask, masked-image) into the LoRA-infused U-Net for final generation.

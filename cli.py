"""
Copyright 2024 Khej-Trhyk (aka 308K) © All rights reserved.
Copyleft 2024 Khej-Trhyk (aka 308K) 🄯 All wrongs reserved.
This file is part of FLUX.1-dev-demo.
FLUX.1-dev-demo is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
FLUX.1-dev-demo is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with FLUX.1-dev-demo. If not, see <https://www.gnu.org/licenses/>. 
"""
import argparse
import json
import torch
from diffusers import FluxPipeline
import matplotlib.pyplot as plt
import numpy as np
import random

with open('config.json', 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)

if not config['device']:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = config['device']

MAX_SEED = np.iinfo(np.int32).max

def generate_image(prompt, height, width, guidance_scale, num_inference_steps, max_sequence_length, output_path):
    pipe = FluxPipeline.from_pretrained(config['FLUX.1-dev_local_path'], torch_dtype=torch.bfloat16)
    if config['save_VRAM']:
        pipe.enable_model_cpu_offload()

    image = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=max_sequence_length,
        generator=torch.Generator(device).manual_seed(0)
    ).images[0]
    
    if output_path:
        image.save(output_path)
    
    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an image using FluxPipeline")
    parser.add_argument("prompt", type=str, help="The prompt for image generation")
    parser.add_argument("-H", "--height", type=int, default=config['height'], help="The height of the generated image")
    parser.add_argument("-W", "--width", type=int, default=config['width'], help="The width of the generated image")
    parser.add_argument("--guidance_scale", type=float, default=config['guidance_scale'], help="The guidance scale for image generation")
    parser.add_argument("--steps", type=int, default=config['inference_steps'], help="The number of inference steps")
    parser.add_argument("--max_sequence_length", type=int, default=config['max_sequence_length'], help="The maximum sequence length")
    parser.add_argument("-S", "--save", nargs='?', const=True, help="The path to save the generated image")

    args = parser.parse_args()

    if args.save is True:
        args.save = f"flux-dev-{random.randint(0, 1000000)}.png"
    if args.save and not args.save.endswith(".png"):
        args.save += ".png"
    if args.guidance_scale <= 0:
        raise ValueError("Guidance scale must be greater than 0")
    if args.steps <= 0:
        raise ValueError("Number of inference steps must be greater than 0")
    if args.max_sequence_length <= 0:
        raise ValueError("Maximum sequence length must be greater than 0")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("Height and width must be greater than 0")

    generate_image(
        args.prompt,
        args.height,
        args.width,
        args.guidance_scale,
        args.steps,
        args.max_sequence_length,
        args.save
    )
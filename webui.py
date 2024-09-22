"""
Copyright 2024 Khej-Trhyk (aka 308K) © All rights reserved.
Copyleft 2024 Khej-Trhyk (aka 308K) 🄯 All wrongs reserved.
This file is part of FLUX.1-dev-demo.
FLUX.1-dev-demo is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
FLUX.1-dev-demo is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with FLUX.1-dev-demo. If not, see <https://www.gnu.org/licenses/>. 
"""
import gradio as gr
import torch
import numpy as np
import random
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import  DiffusionPipeline, FlowMatchEulerDiscreteScheduler, AutoencoderTiny, AutoencoderKL
from live_preview_helpers import calculate_shift, retrieve_timesteps, flux_pipe_call_that_returns_an_iterable_of_images
import json

with open('config.json', 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)

if not config['device']:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = config['device']
dtype = torch.bfloat16

torch.cuda.empty_cache()

taef1 = AutoencoderTiny.from_pretrained(config['taef1_local_path'], torch_dtype=dtype).to(device)
good_vae = AutoencoderKL.from_pretrained(config['FLUX.1-dev_local_path'], subfolder="vae", torch_dtype=dtype).to(device)
pipe = DiffusionPipeline.from_pretrained(config['FLUX.1-dev_local_path'], torch_dtype=dtype, vae=taef1).to(device)
if config['save_VRAM']:
    pipe.enable_model_cpu_offload()
torch.cuda.empty_cache()

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

pipe.flux_pipe_call_that_returns_an_iterable_of_images = flux_pipe_call_that_returns_an_iterable_of_images.__get__(pipe)

def infer(prompt, seed=42, randomize_seed=False, width=256, height=256, guidance_scale=3.5, num_inference_steps=14, progress=gr.Progress(track_tqdm=True)):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)
    
    for img in pipe.flux_pipe_call_that_returns_an_iterable_of_images(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
            output_type="pil",
            good_vae=good_vae,
        ):
            yield img, seed

examples = [
    ["a tiny astronaut hatching from an egg on the moon", 42, True, 256],
    ["a cat holding a sign that says hello world", 42, True, 256],
    ["an anime illustration of a wiener schnitzel", 42, True, 256],
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 520px;
}
"""

interface = gr.Interface(
    fn=infer,
    inputs=[
        gr.components.Textbox(lines=2, placeholder="Enter prompt here..."),
        gr.components.Slider(0, MAX_SEED, step=1, value=42, label="Seed"),
        gr.components.Checkbox(label="Randomize Seed"),
        gr.components.Slider(256, MAX_IMAGE_SIZE, step=256, value=config['width'], label="Width"),
        gr.components.Slider(256, MAX_IMAGE_SIZE, step=256, value=config['height'], label="Height"),
        gr.components.Slider(0.5, 10.0, step=3.5, value=config['guidance_scale'], label="Guidance Scale"),
        gr.components.Slider(1, 100, step=14, value=config['inference_steps'], label="Number of Inference Steps"),
    ],
    outputs=[
        gr.components.Image(type="pil", label="Generated Image"),
        gr.components.Textbox(label="Seed Used")
    ],
    examples=examples,
    css=css,
    live=True
)

interface.launch(server_name="localhost", server_port=8888)
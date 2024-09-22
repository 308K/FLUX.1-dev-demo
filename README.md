# FLUX.1-dev-demo
This project is an image generation tool using the model FLUX.1-dev, featuring both a Command Line Interface (CLI) and a Web User Interface (WebUI).
### Installation
1. Clone this repository:
```bash
git clone https://github.com/308K/FLUX.1-dev-demo
cd FLUX.1-dev-demo
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Download the pre-trained models [FLUX.1-dev-demo](https://huggingface.co/black-forest-labs/FLUX.1-dev) and [teaf1](https://huggingface.co/madebyollin/taef1) and place it in the ./models directory.
### Configuration
Edit the `config.json` file to set your preferred default values and paths for the models.
Example:
```json
{
    "device": "cuda", 
    "save_VRAM": true,
    "FLUX.1-dev_local_path": "./models/FLUX.1-dev",
    "taef1_local_path": "./models/taef1",
    "height": 512,
    "width": 512,
    "guidance_scale": 3.5,
    "inference_steps": 14,
    "max_sequence_length": 128
}
```
### Usage
#### CLI
Generate an image using the CLI:
```bash
python cli.py "your prompt" -H 256 -W 256 --guidance_scale 3.5 --steps 14 --max_sequence_length 128 -s 42 -S "output.png"
```
Parameter descriptions:
* `prompt`: The prompt for image generation.
* `-H`, `--height`: The height of the generated image (default 256).
* `-W`, `--width`: The width of the generated image (default 256).
* `--guidance_scale`: The guidance scale for image generation (default 3.5).
* `--steps`: The number of inference steps (default 14).
* `--max_sequence_length`: The maximum sequence length (default 128).
* `-s`, `--seed`: The seed for random number generation (default random).
* `-S`, `--save`: The path to save the generated image (optional).

#### WebUI
1. Start the WebUI:
```bash
python webui.py
```
2. Open a browser and navigate to `http://localhost:8888`.
3. Enter the prompt and other parameters in the interface, then click the generate button.
### License
![GPLv3 logo](https://www.gnu.org/graphics/gplv3-88x31.png)
This project is licensed under the GPLv3 License. See the LICENSE file for details.
```
Copyright 2024 Khej-Trhyk (aka 308K) © All rights reserved.

Copyleft 2024 Khej-Trhyk (aka 308K) 🄯 All wrongs reserved.

     
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>. 
```
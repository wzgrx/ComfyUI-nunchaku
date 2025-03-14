<div align="center" id="nunchaku_logo">
  <img src="https://raw.githubusercontent.com/mit-han-lab/nunchaku/96615bd93a1f0d2cf98039fddecfec43ce34cc96/assets/nunchaku.svg" alt="logo" width="220"></img>
</div>
<h3 align="center">
<a href="http://arxiv.org/abs/2411.05007"><b>Paper</b></a> | <a href="https://hanlab.mit.edu/projects/svdquant"><b>Website</b></a> | <a href="https://hanlab.mit.edu/blog/svdquant"><b>Blog</b></a> | <a href="https://svdquant.mit.edu"><b>Demo</b></a> | <a href="https://huggingface.co/collections/mit-han-lab/svdquant-67493c2c2e62a1fc6e93f45c"><b>HuggingFace</b></a> | <a href="https://modelscope.cn/collections/svdquant-468e8f780c2641"><b>ModelScope</b></a>
</h3>

This repository is the ComfyUI node for [**Nunchaku**](https://github.com/mit-han-lab/nunchaku), an efficient inference engine for 4-bit diffusion models quantized by [SVDQuant](http://arxiv.org/abs/2411.05007).  Please check [DeepCompressor](https://github.com/mit-han-lab/deepcompressor) for the quantization library.

Check [here](https://github.com/mit-han-lab/nunchaku/issues/149) to join our user groups on [**Slack**](https://join.slack.com/t/nunchaku/shared_invite/zt-3170agzoz-NgZzWaTrEj~n2KEV3Hpl5Q) and [**WeChat**](https://github.com/mit-han-lab/nunchaku/blob/main/assets/wechat.jpg?raw=true) for discussions! If you have any questions, encounter issues, or are interested in contributing to the codebase, feel free to share your thoughts there!

# Nunchaku ComfyUI Node

![comfyui](assets/comfyui.jpg)
## Installation

Please first install `nunchaku` following the instructions in [README.md](https://github.com/mit-han-lab/nunchaku?tab=readme-ov-file#installation). 

**[Optional]** You need to install `image_gen_aux` if you use `FluxDepthPreprocessor` node:

```shell
pip install git+https://github.com/asomoza/image_gen_aux.git
```

### ComfyUI-CLI

```shell
pip install comfy-cli  # install the comfyui-cli
comfy install # install comfyui
comfy node registry-install nunchaku
```

### ComfyUI-Manager

1. Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI/tree/master) with 

   ```shell
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   pip install -r requirements.txt
   ```

2. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) with the following commands then restart ComfyUI:

   ```shell
   cd custom_nodes
   git clone https://github.com/ltdrdata/ComfyUI-Manager comfyui-manager
   ```

3. Launch ComfyUI

   ```shell
   cd ..  # navigate back to the root of ComfyUI
   python main.py
   ```

4. Open the Manager, search `nunchaku` in the Custom Nodes Manager and then install it.


### Manual Installation
1. Set up the dependencies for [ComfyUI](https://github.com/comfyanonymous/ComfyUI/tree/master) with the following commands:

   ```shell
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   pip install -r requirements.txt
   ```

2. Clone this repo to `custom_nodes` directory under ComfyUI:

   ```shell
   cd custom_nodes
   https://github.com/mit-han-lab/ComfyUI-nunchaku
   ```

## Usage

1. **Set Up ComfyUI and SVDQuant**:

     * Nunchaku workflows can be found at [`workflows`](./workflows). You can place them in `user/default/workflows` in ComfyUI root directory to load them. For example:

       ```shell
       cd ComfyUI
       
       # Copy workflow configurations
       mkdir -p user/default/workflows
       cp custom_nodes/ComfyUI-nunchaku/workflows/* user/default/workflows/
       ```

     * Install missing nodes (e.g., comfyui-inpainteasy) following [this tutorial](https://github.com/ltdrdata/ComfyUI-Manager?tab=readme-ov-file#support-of-missing-nodes-installation).

2. **Download Required Models**: Follow [this tutorial](https://comfyanonymous.github.io/ComfyUI_examples/flux/) to download the required models into the appropriate directories. You can use the commands below:

   ```shell
   huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir models/text_encoders
   huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --local-dir models/text_encoders
   huggingface-cli download black-forest-labs/FLUX.1-schnell ae.safetensors --local-dir models/vae
   ```

3. **Run ComfyUI**: From ComfyUI’s root directory, execute the `python main.py` to start the application. If you use `comfy-cli`, you can simply run `comfy launch`.

4. **Select the Nunchaku Workflow**: Choose one of the Nunchaku workflows (workflows that start with `nunchaku-`) to get started. For the FLUX.1-Fill workflow, you can use the built-in MaskEditor tool to add mask on top of an image.

5. All the 4-bit models can be accessible at our [HuggingFace](https://huggingface.co/collections/mit-han-lab/svdquant-67493c2c2e62a1fc6e93f45c) or [ModelScope](https://modelscope.cn/collections/svdquant-468e8f780c2641) collection. Except [`svdq-flux.1-t5`](https://huggingface.co/mit-han-lab/svdq-flux.1-t5), please download the **entire** model folder to `models/diffusion_models`.

## Nunchaku Nodes

* **Nunchaku Flux DiT Loader**: A node for loading the FLUX diffusion model. 

  * `model_path`: Specifies the model location. You need to manually download the model folder from our [HuggingFace](https://huggingface.co/collections/mit-han-lab/svdquant-67493c2c2e62a1fc6e93f45c) or [ModelScope](https://modelscope.cn/collections/svdquant-468e8f780c2641) collection. For example, you can runn the following command to download it:

    ```shell
    huggingface-cli download mit-han-lab/svdq-int4-flux.1-dev --local-dir models/diffusion_models/svdq-int4-flux.1-dev
    ```

     After downloading, specify the corresponding folder name as the `model_path`.

  * `cpu_offload`: Enables CPU offloading for the transformer model. While this may reduce GPU memory usage, it can slow down inference. When set to `auto`, it will detect your GPU memory. If you have enough GPU memory (>14GiB), it will disable. Otherwise, it will enable. **Memory usage will be further optimized in node v0.2**.

  * `device_id`: Indicates the GPU ID for running the model.

* **Nunchaku FLUX LoRA Loader**: A node for loading LoRA modules for SVDQuant FLUX models.

  * Place your LoRA checkpoints in the `models/loras` directory. These will appear as selectable options under `lora_name`. Meanwhile, the [example Ghibsky LoRA](https://huggingface.co/aleksa-codes/flux-ghibsky-illustration) is included and will automatically download from our Hugging Face repository when used.
  * `lora_format` specifies the LoRA format. Supported formats include:
    * `auto`: Automatically detects the appropriate LoRA format.
    * `diffusers` (e.g., [aleksa-codes/flux-ghibsky-illustration](https://huggingface.co/aleksa-codes/flux-ghibsky-illustration))
    * `comfyui` (e.g., [Shakker-Labs/FLUX.1-dev-LoRA-Children-Simple-Sketch](https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-Children-Simple-Sketch))
    * `xlab` (e.g., [XLabs-AI/flux-RealismLora](https://huggingface.co/XLabs-AI/flux-RealismLora))
    * `svdquant` (e.g., [mit-han-lab/svdquant-lora-collection](https://huggingface.co/mit-han-lab/svdquant-lora-collection)).
  
  * `base_model_name` specifies the path to the quantized base model. If `lora_format` is already set to `svdquant`, this option has no use. You can set it to the same value as `model_path` in the above **SVDQuant Flux DiT Loader**.
  * `lora_strength` specfies your LoRA strength.
  * `save_converted_lora`: whether you want to save your converted LoRA to disk to save the conversion time for the next time if your LoRA is not in SVDQuant format. If enabled, the converted LoRA will be saved in the same folder and named `svdq-{precision}-{name}.safetensors`.
  * **Note**: Currently, **only one LoRA** can be loaded at a time. Multiple LoRA support will be added in node v0.2.
* **Nunchaku Text Encoder Loader**: A node for loading the text encoders.

  * For FLUX, use the following files:

    - `text_encoder1`: `t5xxl_fp16.safetensors`
    - `text_encoder2`: `clip_l.safetensors`

  * `t5_min_length`: Sets the minimum sequence length for T5 text embeddings. The default in `DualCLIPLoader` is hardcoded to 256, but for better image quality in SVDQuant, use 512 here.

  * `use_4bit_t5`: Specifies whether you need to use our quantized 4-bit T5 to save GPU memory. Choose `INT4` to use the INT4 text encoder.
  
  
     * `int4_model`: Specifies the INT4 T5 location. This option is only used when `use_4bit_t5` is enabled. You can download the model folder to `models/text_encoders` for [HuggingFace](https://huggingface.co/mit-han-lab/svdq-flux.1-t5) or [ModelScope](https://modelscope.cn/models/Lmxyy1999/svdq-flux.1-t5). For example, you can run the following command:
      
        ```shell
        huggingface-cli download mit-han-lab/svdq-flux.1-t5 --local-dir models/text_encoders/svdq-flux.1-t5
        ```
      
         After downloading, specify the corresponding folder name as the `int4_model`.
  

* **FLUX.1 Depth Preprocessor**: A node for loading the depth estimation model and output the depth map. `model_path` specifies the model location. You can manually download the repository from [HuggingFace](https://huggingface.co/LiheYoung/depth-anything-large-hf) to `models/checkpoints`. For example, you can run the following command example:

  ```shell
  huggingface-cli download LiheYoung/depth-anything-large-hf --local-dir models/checkpoints/depth-anything-large-hf
  ```

  


import json
import os

import comfy.model_patcher
import folder_paths
import torch
from comfy.ldm.common_dit import pad_to_patch_size
from comfy.supported_models import Flux, FluxSchnell
from diffusers import FluxTransformer2DModel
from einops import rearrange, repeat
from nunchaku import NunchakuFluxTransformer2dModel
from torch import nn


class ComfyFluxForwardWrapper(nn.Module):
    def __init__(self, model: NunchakuFluxTransformer2dModel, config):
        super(ComfyFluxForwardWrapper, self).__init__()
        self.model = model
        self.dtype = next(model.parameters()).dtype
        self.config = config

    def forward(self, x, timestep, context, y, guidance, control=None, transformer_options={}, **kwargs):
        assert control is None  # for now
        bs, c, h, w = x.shape
        patch_size = self.config["patch_size"]
        x = pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = (h + (patch_size // 2)) // patch_size
        w_len = (w + (patch_size // 2)) // patch_size
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(
            0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype
        ).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(
            0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype
        ).unsqueeze(0)
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        out = self.model(
            hidden_states=img,
            encoder_hidden_states=context,
            pooled_projections=y,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance if self.config["guidance_embed"] else None,
        ).sample

        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:, :, :h, :w]
        return out


class NunchakuFluxDiTLoader:
    @classmethod
    def INPUT_TYPES(s):
        prefixes = folder_paths.folder_names_and_paths["diffusion_models"][0]
        local_folders = set()
        for prefix in prefixes:
            if os.path.exists(prefix) and os.path.isdir(prefix):
                local_folders_ = os.listdir(prefix)
                local_folders_ = [
                    folder
                    for folder in local_folders_
                    if not folder.startswith(".") and os.path.isdir(os.path.join(prefix, folder))
                ]
                local_folders.update(local_folders_)
        model_paths = sorted(list(local_folders))
        ngpus = torch.cuda.device_count()
        return {
            "required": {
                "model_path": (model_paths, {"tooltip": "The SVDQuant quantized FLUX.1 models."}),
                "cpu_offload": (
                    ["auto", "enable", "disable"],
                    {
                        "default": "auto",
                        "tooltip": "Whether to enable CPU offload for the transformer model. 'auto' will enable it if the GPU memory is less than 14G.",
                    },
                ),
                "device_id": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": ngpus - 1,
                        "step": 1,
                        "display": "number",
                        "lazy": True,
                        "tooltip": "The GPU device ID to use for the model.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Flux DiT Loader"

    def load_model(self, model_path: str, cpu_offload: str, device_id: int, **kwargs) -> tuple[FluxTransformer2DModel]:
        device = f"cuda:{device_id}"
        prefixes = folder_paths.folder_names_and_paths["diffusion_models"][0]
        for prefix in prefixes:
            if os.path.exists(os.path.join(prefix, model_path)):
                model_path = os.path.join(prefix, model_path)
                break

        # Check if the device_id is valid
        if device_id >= torch.cuda.device_count():
            raise ValueError(f"Invalid device_id: {device_id}. Only {torch.cuda.device_count()} GPUs available.")

        # Get the GPU properties
        gpu_properties = torch.cuda.get_device_properties(device_id)
        gpu_memory = gpu_properties.total_memory / (1024**2)  # Convert to MB
        gpu_name = gpu_properties.name
        print(f"GPU {device_id} ({gpu_name}) Memory: {gpu_memory} MB")

        # Check if CPU offload needs to be enabled
        if cpu_offload == "auto":
            if gpu_memory < 14336:  # 14GB threshold
                cpu_offload_enabled = True
                print("VRAM < 14GiB，enable CPU offload")
            else:
                cpu_offload_enabled = False
                print("VRAM > 14GiB，disable CPU offload")
        elif cpu_offload == "enable":
            cpu_offload_enabled = True
            print("Enable CPU offload")
        else:
            cpu_offload_enabled = False
            print("Disable CPU offload")

        capability = torch.cuda.get_device_capability(0)
        sm = f"{capability[0]}{capability[1]}"
        precision = "fp4" if sm == "120" else "int4"
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            model_path, precision=precision, offload=cpu_offload_enabled
        )
        transformer = transformer.to(device)

        if os.path.exists(os.path.join(model_path, "comfy_config.json")):
            config_path = os.path.join(model_path, "comfy_config.json")
        else:
            default_config_root = os.path.join(os.path.dirname(__file__), "configs")
            config_name = os.path.basename(model_path).replace("svdq-int4-", "").replace("svdq-fp4-", "")
            config_path = os.path.join(default_config_root, f"{config_name}.json")
            assert os.path.exists(config_path), f"Config file not found: {config_path}"

        print(f"Loading configuration from {config_path}")
        comfy_config = json.load(open(config_path, "r"))
        model_class_name = comfy_config["model_class"]
        model_config = eval(model_class_name)(comfy_config["model_config"])
        model_config.set_inference_dtype(torch.bfloat16, None)
        model_config.custom_operations = None
        model = model_config.get_model({})
        model.diffusion_model = ComfyFluxForwardWrapper(transformer, config=comfy_config["model_config"])
        model = comfy.model_patcher.ModelPatcher(model, device, device_id)
        return (model,)

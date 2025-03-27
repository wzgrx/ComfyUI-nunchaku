import copy
import logging

import folder_paths
from ..models.flux import ComfyFluxWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NunchakuFluxLoraLoader")


class NunchakuFluxLoraLoader:
    def __init__(self):
        self.cur_lora_name = "None"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                "lora_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the diffusion model. This value can be negative.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    FUNCTION = "load_lora"
    TITLE = "Nunchaku FLUX.1 LoRA Loader"

    CATEGORY = "Nunchaku"
    DESCRIPTION = (
        "LoRAs are used to modify the diffusion model, "
        "altering the way in which latents are denoised such as applying styles. "
        "You can link multiple LoRA nodes."
    )

    def load_lora(self, model, lora_name: str, lora_strength: float):
        model_wrapper = model.model.diffusion_model
        assert isinstance(model_wrapper, ComfyFluxWrapper)

        transformer = model_wrapper.model
        model_wrapper.model = None
        ret_model = copy.deepcopy(model)  # copy everything except the model
        ret_model_wrapper = ret_model.model.diffusion_model
        assert isinstance(ret_model_wrapper, ComfyFluxWrapper)

        model_wrapper.model = transformer
        ret_model_wrapper.model = transformer

        ret_model_wrapper.loras.append((folder_paths.get_full_path_or_raise("loras", lora_name), lora_strength))
        return (ret_model,)

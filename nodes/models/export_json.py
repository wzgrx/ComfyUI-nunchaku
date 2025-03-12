import json


def main():
    config = {
        "model_class": "Flux",
        "dit_config": {
            "image_model": "flux",
            "patch_size": 2,
            "out_channels": 16,
            "vec_in_dim": 768,
            "context_in_dim": 4096,
            "hidden_size": 3072,
            "mlp_ratio": 4.0,
            "num_heads": 24,
            "depth": 19,
            "depth_single_blocks": 38,
            "axes_dim": [16, 56, 56],
            "theta": 10000,
            "qkv_bias": True,
            "guidance_embed": True,
            "disable_unet_model_creation": True,
            "in_channels": 64,
        },
    }
    with open("configs/flux.1-fill-dev.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

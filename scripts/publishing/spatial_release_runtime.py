"""Portable inference for spatial feature predictors; no robot policy/RGB decoder."""
import hashlib
import json
from pathlib import Path


PACKAGE_KIND = "shiftwm_real_video_spatial_v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_package(directory, device="cpu"):
    """Strictly restore an inference-only package without any training paths."""
    import torch
    from shiftwm.real_video_spatial.model import from_config
    directory = Path(directory)
    if directory.is_symlink() or (directory / 'package_manifest.json').is_symlink():
        raise ValueError('Inference package must contain independent regular files')
    manifest = json.loads((directory / "package_manifest.json").read_text())
    if (manifest.get("format_version") != 1 or manifest.get("package_kind") != PACKAGE_KIND
            or set(manifest.get("files", {})) != {"model.pt", "config.json"}):
        raise ValueError("Require the complete spatial inference-only package")
    for name, expected in manifest["files"].items():
        if (directory / name).is_symlink() or sha(directory / name) != expected:
            raise ValueError("Spatial model file identity differs: " + name)
    state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    config = json.loads((directory / "config.json").read_text())
    if (state.get("config") != config or config.get("package_kind") != PACKAGE_KIND
            or type(state.get("epoch")) is not int or not 1 <= state["epoch"] <= 30):
        raise ValueError("Spatial package metadata differs")
    model = from_config(config)
    model.load_state_dict(state["state_dict"], strict=True)
    for key, value in model.package_config.items():
        if config.get(key) != value:
            raise ValueError("Reconstructed spatial configuration differs: " + key)
    counts = {"total": sum(p.numel() for p in model.parameters()),
              "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if config["metadata"].get("parameter_counts") != counts:
        raise ValueError("Spatial parameter counts differ")
    return model.to(device).eval(), state


def encode_rgb(images, encoder_directory, device="cpu", bf16=False):
    """Encode RGB float [...,3,H,W] in [0,1] into raw [...,6144] features.

    Exact resize/normalization/pooling order matches the registered extractor.
    The original GPU cache uses BF16; CPU FP32 is not bitwise cache-equivalent.
    No images are decoded/downloaded by this helper; pass observed frames only.
    """
    import torch
    from torch.nn import functional as F
    from transformers import Dinov2Model
    encoder_directory = Path(encoder_directory)
    provenance = json.loads((encoder_directory / "provenance.json").read_text())
    if (provenance.get("repo") != "facebook/dinov2-small"
            or provenance.get("revision") != "ed25f3a31f01632728cabb09d1542f84ab7b0056"):
        raise ValueError("Wrong frozen DINO encoder")
    rows = provenance.get('files', [])
    if (len(rows) != 4 or {r.get('file') for r in rows} !=
            {'config.json','preprocessor_config.json','model.safetensors','README.md'}):
        raise ValueError('Incomplete frozen encoder inventory')
    if encoder_directory.is_symlink() or (encoder_directory/'provenance.json').is_symlink():
        raise ValueError('Encoder must contain independent regular files')
    for row in provenance["files"]:
        name = Path(row["file"])
        if (name.is_absolute() or len(name.parts) != 1 or (encoder_directory/name).is_symlink()
                or (encoder_directory/name).stat().st_size != row.get('bytes')
                or sha(encoder_directory / name) != row["sha256"]):
            raise ValueError("Encoder source identity differs")
    if (images.ndim < 4 or images.shape[-3] != 3 or not images.is_floating_point()
            or not torch.isfinite(images).all() or (images < 0).any() or (images > 1).any()):
        raise ValueError("RGB inputs must be finite float [...,3,H,W] in [0,1]")
    if bf16 and torch.device(device).type != "cuda":
        raise ValueError("Registered BF16 encoding requires an allocated CUDA device")
    encoder = Dinov2Model.from_pretrained(encoder_directory, local_files_only=True).to(device).eval().requires_grad_(False)
    shape = images.shape[:-3]
    pixels = images.reshape(-1, *images.shape[-3:]).to(device).float()
    pixels = F.interpolate(pixels, (224,224), mode="bilinear", align_corners=False, antialias=True)
    mean = pixels.new_tensor([.485,.456,.406])[None,:,None,None]
    std = pixels.new_tensor([.229,.224,.225])[None,:,None,None]
    with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=bf16, dtype=torch.bfloat16):
        patches = encoder(pixel_values=(pixels - mean) / std).last_hidden_state[:,1:]
    if tuple(patches.shape[1:]) != (256,384):
        raise ValueError("Unexpected frozen DINO patch layout")
    features = F.adaptive_avg_pool2d(patches.float().transpose(1,2).reshape(-1,384,16,16),(4,4)).flatten(1)
    if not torch.isfinite(features).all():
        raise ValueError("Nonfinite encoded features")
    return features.reshape(*shape,6144)

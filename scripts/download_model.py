"""Download opt-6.7b with retry/resume — separate from inference."""

from huggingface_hub import snapshot_download

model_id = "facebook/opt-6.7b"
print(f"Downloading {model_id}...")
local_dir = snapshot_download(
    model_id,
    resume_download=True,
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
)
print(f"Done! Saved to: {local_dir}")

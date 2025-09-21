import subprocess
import os
import modal

# ------------------------------------------------------------
# Modal app: WAN VACE Long Video Generation
# ------------------------------------------------------------

# Base image with essentials
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "ffmpeg",
        "build-essential",
        "cmake",
        "wget",
    )
    .pip_install(
        # Core deps
        "opencv-python-headless",
        "imageio[ffmpeg]",
        "moviepy",
        "fastapi[standard]==0.115.4",
        "comfy-cli==1.5.1",
    )
    # Install ComfyUI with NVIDIA
    .run_commands(
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.59"
    )
)

# Custom nodes + deps
image = image.run_commands(
    # VideoHelperSuite
    "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
    # KJNodes
    "git clone https://github.com/kijai/ComfyUI-KJNodes.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-KJNodes",
    # Essentials
    "git clone https://github.com/cubiq/ComfyUI_essentials.git /root/comfy/ComfyUI/custom_nodes/ComfyUI_essentials",
    # Easy-Use
    "git clone https://github.com/yolain/ComfyUI-Easy-Use.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-Easy-Use",
    # Custom Scripts
    "git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-Custom-Scripts",
    # LayerStyle
    "git clone https://github.com/chflame163/ComfyUI_LayerStyle.git /root/comfy/ComfyUI/custom_nodes/ComfyUI_LayerStyle",
    # Frame Interpolation
    "git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation",
    # Logic
    "git clone https://github.com/theUpsider/ComfyUI-Logic.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-Logic",
    # Comfyroll
    "git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git /root/comfy/ComfyUI/custom_nodes/ComfyUI_Comfyroll_CustomNodes",
    # rgthree
    "git clone https://github.com/rgthree/rgthree-comfy.git /root/comfy/ComfyUI/custom_nodes/rgthree-comfy",

    # Install deps for VHS
    "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && pip install -r requirements.txt",

    # Install deps for Frame Interpolation (uses install.py)
    "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation && python install.py"
)

# ------------------------------------------------------------
# Hugging Face downloads
# ------------------------------------------------------------

def hf_download():
    from huggingface_hub import hf_hub_download
    
    # Ensure dirs exist
    os.makedirs("/root/comfy/ComfyUI/models/vae", exist_ok=True)
    os.makedirs("/root/comfy/ComfyUI/models/text_encoders", exist_ok=True)
    os.makedirs("/root/comfy/ComfyUI/models/diffusion_models", exist_ok=True)
    os.makedirs("/root/comfy/ComfyUI/models/loras", exist_ok=True)

    # VAE
    vae_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/vae/wan_2.1_vae.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {vae_model} /root/comfy/ComfyUI/models/vae/wan_2.1_vae.safetensors",
        shell=True,
        check=True,
    )

    # Text encoder (fp8 + optional fp16)
    t5_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {t5_model} /root/comfy/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        shell=True,
        check=True,
    )

    try:
        t5_model_fp16 = hf_hub_download(
            repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            filename="split_files/text_encoders/umt5_xxl_fp16.safetensors",
            cache_dir="/cache",
        )
        subprocess.run(
            f"ln -sf {t5_model_fp16} /root/comfy/ComfyUI/models/text_encoders/umt5_xxl_fp16.safetensors",
            shell=True,
            check=True,
        )
    except:
        print("umt5_xxl_fp16.safetensors not found, using fp8 version")

    # Diffusion models
    wan_dir = "/root/comfy/ComfyUI/models/diffusion_models/wan2_2"
    os.makedirs(wan_dir, exist_ok=True)

    high_noise_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {high_noise_model} {os.path.join(wan_dir, 'wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors')}",
        shell=True,
        check=True,
    )

    low_noise_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {low_noise_model} {os.path.join(wan_dir, 'wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors')}",
        shell=True,
        check=True,
    )

    # LoRA dirs (manual placement)
    for d in [
        "/root/comfy/ComfyUI/models/loras/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1",
        "/root/comfy/ComfyUI/models/loras/Wan2_2/pusa_wan2-2_v1",
        "/root/comfy/ComfyUI/models/loras/Wan2_2",
    ]:
        os.makedirs(d, exist_ok=True)

    # RIFE
    try:
        rife_model = hf_hub_download(
            repo_id="AlexWortega/RIFE",
            filename="rife47.pth",
            cache_dir="/cache",
        )
        rife_dir = "/root/comfy/ComfyUI/models/rife"
        os.makedirs(rife_dir, exist_ok=True)
        subprocess.run(
            f"ln -sf {rife_model} {os.path.join(rife_dir, 'rife47.pth')}",
            shell=True,
            check=True,
        )
    except:
        print("RIFE model not found, frame interpolation may not work")

# ------------------------------------------------------------
# Volumes + image with HF
# ------------------------------------------------------------

vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

image = (
    image.pip_install("huggingface_hub[hf_transfer]>=0.34.0,<1.0")
    .run_function(
        hf_download,
        volumes={"/cache": vol},
    )
)

# ------------------------------------------------------------
# Modal app
# ------------------------------------------------------------

app = modal.App(name="wan-vace-long-video", image=image)

@app.function(
    max_containers=1,
    gpu="L40S",
    volumes={"/cache": vol},
    timeout=7200,
    memory=32768,
)
@modal.concurrent(max_inputs=2)
@modal.web_server(8000, startup_timeout=180)
def ui():
    import time
    p = subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)
    # Block to keep server alive
    while True:
        time.sleep(3600)

@app.function(
    gpu="L40S",
    volumes={"/cache": vol},
    timeout=7200,
    memory=32768,
)
def generate_long_video(
    image_path: str,
    travel_prompts: str,
    negative_prompt: str = "静态，过曝，细节模糊不清，字幕，风格，作品，画作，画面，静止",
    width: int = 832,
    height: int = 480,
    fps: int = 16,
    video_seconds: int = 30,
    sampler_length: int = 121,
    overlap_frames: int = 10,
    steps: int = 8,
    seed: int = -1
):
    import random
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    total_frames = fps * video_seconds
    print(f"Generating {video_seconds}s video ({total_frames} frames) at {fps} FPS")
    print(f"Using {sampler_length} frames per segment with {overlap_frames} overlap")
    # TODO: implement ComfyUI workflow programmatically
    pass

if __name__ == "__main__":
    pass

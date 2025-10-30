import modal

cuda_version = "12.8.1"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

gptneox_commit = "d12c771198388980ee054617e537665f044e0584"
tokensmith_commit = "aaf7364b921f397581dafb6c79ca3e618885128a"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install(
        "git",
        "build-essential",        # gcc/g++/make
        "python3-dev",            # Python headers for C extensions
        "libatlas-base-dev",      # BLAS support
        "libopenblas-dev",
        "liblapack-dev",
    )
    .run_commands(
        "pip install --upgrade pip setuptools wheel cython numpy",
        gpu="A100-40GB"
    )
    .run_commands(
        "pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 "
        "--index-url https://download.pytorch.org/whl/cu121",
        gpu="A100-40GB"
    )
    .run_commands(
        f"git clone https://github.com/EleutherAI/gpt-neox.git && "
        f"cd gpt-neox && "
        f"git checkout {gptneox_commit} && "
        "pip install -r requirements/requirements.txt",
        gpu="A100-40GB"
    )
    .run_commands(
        f"git clone https://github.com/aflah02/tokensmith.git && "
        f"cd tokensmith && "
        f"git checkout {tokensmith_commit} && "
        'pip install ".[all]"'
    )
    # Add Jupyter Notebook dependencies
    .run_commands(
        "pip install tokengrams jupyter notebook jupyterlab ipywidgets matplotlib seaborn ipykernel",
    )
)

app = modal.App("tokensmith-backend", image=image)

@app.function()
def run():
    pass
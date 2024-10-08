import os
import tempfile

import imageio
import numpy as np
import rembg
import torch

# from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from einops import rearrange
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torchvision.transforms import v2
from tqdm import tqdm

from src.utils.camera_util import (
    FOV_to_intrinsics,
    get_circular_camera_poses,
    get_zero123plus_input_cameras,
)
from src.utils.infer_util import images_to_video, remove_background, resize_foreground
from src.utils.mesh_util import save_glb, save_obj
from src.utils.train_util import instantiate_from_config

if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
else:
    device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device1 = device0

# Define the cache directory for model files
model_cache_dir = "./ckpts/"
os.makedirs(model_cache_dir, exist_ok=True)


def get_render_cameras(
    batch_size=1, M=120, radius=2.5, elevation=10.0, is_flexicubes=False
):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = (
            FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        )
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def images_to_video(images, output_path, fps=30):
    # images: (N, C, H, W)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []
    for i in range(images.shape[0]):
        frame = (
            (images[i].permute(1, 2, 0).cpu().numpy() * 255)
            .astype(np.uint8)
            .clip(0, 255)
        )
        assert (
            frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3]
        ), f"Frame shape mismatch: {frame.shape} vs {images.shape}"
        assert (
            frame.min() >= 0 and frame.max() <= 255
        ), f"Frame value out of range: {frame.min()} ~ {frame.max()}"
        frames.append(frame)
    imageio.mimwrite(output_path, np.stack(frames), fps=fps, codec="h264")


###############################################################################
# Configuration.
###############################################################################

seed_everything(0)

config_path = "configs/instant-mesh-large.yaml"
config = OmegaConf.load(config_path)
config_name = os.path.basename(config_path).replace(".yaml", "")
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True if config_name.startswith("instant-mesh") else False

device = torch.device("cuda")

# load reconstruction model
print("Loading reconstruction model ...")
model_ckpt_path = hf_hub_download(
    repo_id="TencentARC/InstantMesh",
    filename="instant_mesh_large.ckpt",
    repo_type="model",
    cache_dir=model_cache_dir,
)
model = instantiate_from_config(model_config)
state_dict = torch.load(model_ckpt_path, map_location="cpu")["state_dict"]
state_dict = {
    k[14:]: v
    for k, v in state_dict.items()
    if k.startswith("lrm_generator.") and "source_camera" not in k
}
model.load_state_dict(state_dict, strict=True)

model = model.to(device1)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device1, fovy=30.0)
model = model.eval()

print("Loading Finished!")


def check_input_image(input_image, input_image2):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(input_image, input_image2, do_remove_background):
    rembg_session = rembg.new_session() if do_remove_background else None
    # resize the input image to 256 while keeping the aspect ratio using pil
    input_image = input_image.resize((256, 256))
    input_image2 = input_image2.resize((256, 256))
    if do_remove_background:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)

        input_image2 = remove_background(input_image2, rembg_session)
        input_image2 = resize_foreground(input_image2, 0.85)

    return input_image, input_image2


def generate_mvs(input_image, input_image2, sample_steps, sample_seed):
    seed_everything(sample_seed)

    # put these 2 image in one row
    combined_image = Image.new(
        "RGBA", (input_image.width, input_image.height + input_image2.height)
    )
    combined_image.paste(input_image, (0, 0))
    combined_image.paste(input_image2, (0, input_image.height))
    # save the combined image
    combined_image.save("combined_image.png")
    print(combined_image.size)

    # sampling
    generator = torch.Generator(device=device0)
    z123_image = pipeline(
        combined_image,
        num_inference_steps=sample_steps,
        generator=generator,
    ).images[0]

    show_image = np.asarray(z123_image, dtype=np.uint8)
    show_image = torch.from_numpy(show_image)  # (960, 640, 3)
    show_image = rearrange(show_image, "(n h) (m w) c -> (n m) h w c", n=3, m=2)
    show_image = rearrange(show_image, "(n m) h w c -> (n h) (m w) c", n=2, m=3)
    show_image = Image.fromarray(show_image.numpy())

    return z123_image, show_image


def make_mesh(mesh_fpath, planes):
    mesh_basename = os.path.basename(mesh_fpath).split(".")[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    mesh_glb_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.glb")

    with torch.no_grad():
        # get mesh

        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=False,
            **infer_config,
        )

        vertices, faces, vertex_colors = mesh_out
        vertices = vertices[:, [1, 2, 0]]

        save_glb(vertices, faces, vertex_colors, mesh_glb_fpath)
        save_obj(vertices, faces, vertex_colors, mesh_fpath)

        print(f"Mesh saved to {mesh_fpath}")

    return mesh_fpath, mesh_glb_fpath


def rgba_to_rgb(image):
    """
    Convert an RGBA image to RGB, replacing transparent pixels (alpha=0) with white.

    Parameters:
    image (numpy.ndarray): The input RGBA image as a NumPy array with shape (height, width, 4).

    Returns:
    numpy.ndarray: The output RGB image as a NumPy array with shape (height, width, 3).
    """
    # Separate the RGBA channels
    r, g, b, a = image[:, :, 0], image[:, :, 1], image[:, :, 2], image[:, :, 3]

    # Normalize alpha to [0, 1] range
    alpha_factor = a / 255.0

    # Create a white background (255, 255, 255)
    white_background = np.ones_like(image[:, :, :3]) * 255

    # Blend the image with the white background based on alpha
    rgb_image = (1 - alpha_factor[:, :, np.newaxis]) * white_background + (
        alpha_factor[:, :, np.newaxis]
    ) * image[:, :, :3]

    # Convert to uint8 and return

    return rgb_image.astype(np.uint8)


def make3d(images):
    images = np.asarray(images, dtype=np.float32) / 255.0
    print(images.shape)
    images = images[:, :, 0:3]  # remove alpha channel
    images = (
        torch.from_numpy(images).permute(2, 0, 1).contiguous().float()
    )  # (3, 960, 640)
    images = rearrange(
        images, "c (n h) (m w) -> (n m) c h w", n=3, m=2
    )  # (6, 3, 320, 320)

    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device1)
    render_cameras = get_render_cameras(
        batch_size=1, radius=4.5, elevation=20.0, is_flexicubes=IS_FLEXICUBES
    ).to(device1)

    images = images.unsqueeze(0).to(device1)
    images = v2.functional.resize(
        images, (320, 320), interpolation=3, antialias=True
    ).clamp(0, 1)

    mesh_fpath = tempfile.NamedTemporaryFile(suffix=".obj", delete=False).name
    print(mesh_fpath)
    mesh_basename = os.path.basename(mesh_fpath).split(".")[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    video_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.mp4")

    with torch.no_grad():
        # get triplane
        planes = model.forward_planes(images, input_cameras)

        # get video
        chunk_size = 20 if IS_FLEXICUBES else 1
        render_size = 384

        frames = []
        for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
            if IS_FLEXICUBES:
                frame = model.forward_geometry(
                    planes,
                    render_cameras[:, i : i + chunk_size],
                    render_size=render_size,
                )["img"]
            else:
                frame = model.synthesizer(
                    planes,
                    cameras=render_cameras[:, i : i + chunk_size],
                    render_size=render_size,
                )["images_rgb"]
            frames.append(frame)
        frames = torch.cat(frames, dim=1)

        images_to_video(
            frames[0],
            video_fpath,
            fps=30,
        )

        print(f"Video saved to {video_fpath}")

    mesh_fpath, mesh_glb_fpath = make_mesh(mesh_fpath, planes)

    return video_fpath, mesh_fpath, mesh_glb_fpath


import gradio as gr

_HEADER_ = """
<h2><b>Official 🤗 Gradio Demo</b></h2><h2><a href='https://github.com/TencentARC/InstantMesh' target='_blank'><b>InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models</b></a></h2>

**InstantMesh** is a feed-forward framework for efficient 3D mesh generation from a single image based on the LRM/Instant3D architecture.

Code: <a href='https://github.com/TencentARC/InstantMesh' target='_blank'>GitHub</a>. Techenical report: <a href='https://arxiv.org/abs/2404.07191' target='_blank'>ArXiv</a>.

❗️❗️❗️**Important Notes:**
- Our demo can export a .obj mesh with vertex colors or a .glb mesh now. If you prefer to export a .obj mesh with a **texture map**, please refer to our <a href='https://github.com/TencentARC/InstantMesh?tab=readme-ov-file#running-with-command-line' target='_blank'>Github Repo</a>.
- The 3D mesh generation results highly depend on the quality of generated multi-view images. Please try a different **seed value** if the result is unsatisfying (Default: 42).
"""

_CITE_ = r"""
If InstantMesh is helpful, please help to ⭐ the <a href='https://github.com/TencentARC/InstantMesh' target='_blank'>Github Repo</a>. Thanks! [![GitHub Stars](https://img.shields.io/github/stars/TencentARC/InstantMesh?style=social)](https://github.com/TencentARC/InstantMesh)
---
📝 **Citation**

If you find our work useful for your research or applications, please cite using this bibtex:
```bibtex
@article{xu2024instantmesh,
  title={InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models},
  author={Xu, Jiale and Cheng, Weihao and Gao, Yiming and Wang, Xintao and Gao, Shenghua and Shan, Ying},
  journal={arXiv preprint arXiv:2404.07191},
  year={2024}
}
```

📋 **License**

Apache-2.0 LICENSE. Please refer to the [LICENSE file](https://huggingface.co/spaces/TencentARC/InstantMesh/blob/main/LICENSE) for details.

📧 **Contact**

If you have any questions, feel free to open a discussion or contact us at <b>bluestyle928@gmail.com</b>.
"""

with gr.Blocks() as demo:
    gr.Markdown(_HEADER_)
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    image_mode="RGBA",
                    sources="upload",
                    width=256,
                    height=256,
                    type="pil",
                    elem_id="content_image",
                )
            with gr.Row():
                submit = gr.Button("Generate", elem_id="generate", variant="primary")
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    output_video = gr.Video(
                        label="video",
                        format="mp4",
                        width=379,
                        autoplay=True,
                        interactive=False,
                    )

            with gr.Row():
                with gr.Tab("OBJ"):
                    output_model_obj = gr.Model3D(
                        label="Output Model (OBJ Format)",
                        # width=768,
                        interactive=False,
                    )
                    gr.Markdown(
                        "Note: Downloaded .obj model will be flipped. Export .glb instead or manually flip it before usage."
                    )
                with gr.Tab("GLB"):
                    output_model_glb = gr.Model3D(
                        label="Output Model (GLB Format)",
                        # width=768,
                        interactive=False,
                    )
                    gr.Markdown(
                        "Note: The model shown here has a darker appearance. Download to get correct results."
                    )

            with gr.Row():
                gr.Markdown(
                    """Try a different <b>seed value</b> if the result is unsatisfying (Default: 42)."""
                )

    gr.Markdown(_CITE_)
    mv_images = gr.State()

    submit.click(fn=check_input_image, inputs=[input_image]).success(
        fn=make3d,
        inputs=[input_image],
        outputs=[output_video, output_model_obj, output_model_glb],
    )

demo.queue(max_size=10)
demo.launch(server_name="0.0.0.0", server_port=43839)

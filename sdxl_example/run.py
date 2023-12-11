import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

base = StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype=torch.float16,
    variant='fp16',
    use_safetensors=True,
).to('cuda')
refiner: StableDiffusionXLImg2ImgPipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-refiner-1.0',
    text_encoder_2=base.text_encoder_2,
    torch_dtype=torch.float16,
    vae=base.vae,
    variant='fp16',
    use_safetensors=True,
).to('cuda')
svd: StableVideoDiffusionPipeline = StableVideoDiffusionPipeline.from_pretrained(
    'stabilityai/stable-video-diffusion-img2vid-xt',
    torch_dtype=torch.float16,
    variant='fp16',
    use_safetensors=True,
).to('cuda')


def sdxl_base(
        prompt: str,
        high_noise_frac=0.8,
):
    return base(
        prompt=prompt,
        prompt_2=prompt,
        num_images_per_prompt=2,
        output_type='latent',
        denoising_end=high_noise_frac,
        width=1024,
        height=1024,
    ).images


def sdxl_refiner(
        prompt,
        images,
        high_noise_frac=0.8,
):
    return refiner(
        prompt=prompt,
        image=images,
        num_images_per_prompt=2,
        denoising_start=high_noise_frac,
        width=1024,
        height=1024,
    ).images


def sdxl_video(
        images,
        num_frames=10,
        fps=20,
):
    images = [img.resize((1024, 576)) for img in images]
    return svd(
        image=images,
        num_frames=num_frames,
        fps=fps,
        decode_chunk_size=8,
    ).frames


def run_inference(
        prompt: str,
):
    latents = sdxl_base(prompt)
    images = sdxl_refiner(prompt, latents)
    all_frames = sdxl_video(images)

    for i, frames in enumerate(all_frames):
        export_to_video(frames, f"generated-{i}.mp4")


def main():
    run_inference('A cute dog')

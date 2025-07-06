import time
from typing import Optional

import torch
from einops import rearrange
from PIL import Image
from torchvision import transforms

from flux.cli import SamplingOptions
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)


class FluxText2ImageInference:
    
    def __init__(
        self,
        model_name: str = "flux-dev",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        offload: bool = False
    ):
        self.model_name = model_name
        self.device = torch.device(device)
        self.offload = offload
        self.is_schnell = model_name == "flux-schnell"
        
        
        self.t5 = load_t5(self.device, max_length=256 if self.is_schnell else 512)
        self.clip = load_clip(self.device)
        self.model = load_flow_model(model_name, device="cpu" if offload else self.device)
        self.ae = load_ae(model_name, device="cpu" if offload else self.device)
        
        
        self.rng = torch.Generator(device="cpu")
    
    def generate(
        self,
        prompt: str,
        width: int = 1360,
        height: int = 768,
        num_steps: Optional[int] = None,
        guidance: float = 3.5,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate an image from text prompt
        
        Args:
            prompt: Text description of the image to generate
            width: Image width (will be rounded to nearest multiple of 16)
            height: Image height (will be rounded to nearest multiple of 16)
            num_steps: Number of denoising steps (defaults: 4 for schnell, 50 for dev)
            guidance: Guidance scale (ignored for schnell)
            seed: Random seed for reproducibility
            
        Returns:
            PIL Image
        """
        
        if num_steps is None:
            num_steps = 4 if self.is_schnell else 50
        
        
        width = int(16 * (width // 16))
        height = int(16 * (height // 16))
        
        
        if seed is None:
            seed = self.rng.seed()
        
        opts = SamplingOptions(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance if not self.is_schnell else 3.5,
            seed=seed,
        )
        
       
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        
        
        timesteps = get_schedule(
            opts.num_steps,
            (x.shape[-1] * x.shape[-2]) // 4,
            shift=(not self.is_schnell),
        )
        
        
        if self.offload:
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
        
        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=opts.prompt)
        
        
        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)
        
       
        x = denoise(self.model, **inp, timesteps=timesteps, guidance=opts.guidance)
        
       
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)
        
        
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)
        
        if self.offload:
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()
        
       
        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        
        return img


class FluxImage2ImageInference:
    
    def __init__(
        self,
        model_name: str = "flux-dev",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        offload: bool = False
    ):
        if model_name == "flux-schnell":
            raise ValueError("Image-to-image is not supported for flux-schnell model")
        
        self.model_name = model_name
        self.device = torch.device(device)
        self.offload = offload
        
        
        self.t5 = load_t5(self.device, max_length=512)
        self.clip = load_clip(self.device)
        self.model = load_flow_model(model_name, device="cpu" if offload else self.device)
        self.ae = load_ae(model_name, device="cpu" if offload else self.device)
        
        self.rng = torch.Generator(device="cpu")
        
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2.0 * x - 1.0),
        ])
    
    def generate(
        self,
        prompt: str,
        init_image: Image.Image,
        strength: float = 0.8,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_steps: int = 50,
        guidance: float = 3.5,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate an image from text prompt and initial image
        
        Args:
            prompt: Text description of the image to generate
            init_image: PIL Image to use as starting point
            strength: Strength of the modification (0.0 = no change, 1.0 = complete change)
            width: Target width (if None, uses init_image width)
            height: Target height (if None, uses init_image height)
            num_steps: Number of denoising steps
            guidance: Guidance scale
            seed: Random seed for reproducibility
            
        Returns:
            PIL Image
        """
        
        if init_image.mode != "RGB":
            init_image = init_image.convert("RGB")
        
        init_tensor = self.transform(init_image)[None, ...]  # Add batch dimension
        
        
        if width is None or height is None:
            _, _, h, w = init_tensor.shape
            if width is None:
                width = w
            if height is None:
                height = h
        
        
        width = int(16 * (width // 16))
        height = int(16 * (height // 16))
        
        
        init_tensor = torch.nn.functional.interpolate(init_tensor, (height, width))
        
       
        if seed is None:
            seed = self.rng.seed()
        
        opts = SamplingOptions(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )
        
       
        
        
        if self.offload:
            self.ae.encoder.to(self.device)
        
        init_latent = self.ae.encode(init_tensor.to(self.device))
        
        if self.offload:
            self.ae = self.ae.cpu()
            torch.cuda.empty_cache()
        
        
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        
        
        timesteps = get_schedule(
            opts.num_steps,
            (x.shape[-1] * x.shape[-2]) // 4,
            shift=True,
        )
        
        t_idx = int((1 - strength) * num_steps)
        t = timesteps[t_idx]
        timesteps = timesteps[t_idx:]
        
        
        x = t * x + (1.0 - t) * init_latent.to(x.dtype)
        
        
        if self.offload:
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
        
        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=opts.prompt)
        
        
        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)
        
        
        x = denoise(self.model, **inp, timesteps=timesteps, guidance=opts.guidance)
        
        
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)
        
        
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)
        
        if self.offload:
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()
        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        
        return img
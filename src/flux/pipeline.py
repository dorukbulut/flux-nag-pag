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


class FluxInference:
    
    
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
        
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2.0 * x - 1.0),
        ])
    
    def text_to_image(
        self,
        prompt: str,
        width: int = 1360,
        height: int = 768,
        num_steps: Optional[int] = None,
        guidance: float = 3.5,
        seed: Optional[int] = None
    ) -> Image.Image:
        """Generate image from text"""
        return self.generate(prompt, None, width, height, num_steps, guidance, seed)
    
    def image_to_image(
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
        """Generate image from text and initial image"""
        if self.is_schnell:
            raise ValueError("Image-to-image not supported for flux-schnell")
        
        return self.generate(prompt, (init_image, strength), width, height, num_steps, guidance, seed)
    
    def generate(self, prompt, img2img_data, width, height, num_steps, guidance, seed):
        """Shared generation logic"""
        
        
        init_latent = None
        if img2img_data:
            init_image, strength = img2img_data
            if init_image.mode != "RGB":
                init_image = init_image.convert("RGB")
            
            init_tensor = self.transform(init_image)[None, ...]
            
            if width is None or height is None:
                _, _, h, w = init_tensor.shape
                width = width or w
                height = height or h
            
            width = int(16 * (width // 16))
            height = int(16 * (height // 16))
            init_tensor = torch.nn.functional.interpolate(init_tensor, (height, width))
            
            
            if self.offload:
                self.ae.encoder.to(self.device)
            init_latent = self.ae.encode(init_tensor.to(self.device))
            if self.offload:
                self.ae.encoder.cpu()
        else:
            width = int(16 * (width // 16))
            height = int(16 * (height // 16))
        
        
        if num_steps is None:
            num_steps = 4 if self.is_schnell else 50
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
        
        
        x = get_noise(1, opts.height, opts.width, device=self.device, dtype=torch.bfloat16, seed=opts.seed)
        
        
        timesteps = get_schedule(opts.num_steps, (x.shape[-1] * x.shape[-2]) // 4, shift=(not self.is_schnell))
        
        
        if init_latent is not None:
            strength = img2img_data[1]
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
        return Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
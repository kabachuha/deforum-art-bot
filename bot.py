import os
import PIL, numpy

import torch

# Literally 1984
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from profanity_filter import ProfanityFilter

import concurrent.futures
import asyncio
import time

# Blocking common core
class Core:
    def __init__(self):
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker",
            )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "CompVis/stable-diffusion-safety-checker",
        )
        assert torch.cuda.is_available()
        self.device = torch.device('cuda')
        self.safety_dtype = torch.float16
        self.pf = ProfanityFilter()

    def run_safety_checker(self, image):
        image_numpy = numpy.asarray(image)
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(self.device)
            image_numpy, has_nsfw_concept = self.safety_checker(
                images=image_numpy, clip_input=safety_checker_input.pixel_values.to(self.safety_dtype)
            )
        else:
            has_nsfw_concept = None
        return has_nsfw_concept

    def verify_generated_video_safety(self, path):
        for f in os.listdir(path):
            if f.endswith('.png') and self.run_safety_checker(PIL.Image.open(f)):
                return False
        return True

    def is_prompt_safe(self, text):
        return self.pf.is_clean(text)

# Each worker is dispatched to its own running Auto1111 instance via the HTTP api
class Worker:
    def __init__(self, core, api_url):
        self.core = core
        self.api_url = api_url

    

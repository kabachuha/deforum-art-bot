# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

import os, traceback, shutil
import PIL, numpy

import torch
from tqdm import tqdm

# Literally 1984
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from profanity_filter import ProfanityFilter

import concurrent.futures
import asyncio
import time

import discord
from discord.ext import commands

import requests
import typing, functools
import json

with open('config.json', 'r') as cfg_file:
    config = json.loads(cfg_file.read())
assert config is not None

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

key = ""

with open('.key.txt', 'r') as id_file:
    key = id_file.readline()

bot = commands.Bot(command_prefix='/', intents=intents)

def to_thread(func: typing.Callable) -> typing.Coroutine:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)
    return wrapper

# Blocking common core
class Safety:
    def __init__(self):
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker",
            )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "CompVis/stable-diffusion-safety-checker",
        )
        assert torch.cuda.is_available()
        self.device = torch.device('cuda')#torch.device('cuda')
        self.safety_dtype = torch.float16#torch.float16
        self.pf = ProfanityFilter()
    
    def numpy_to_pil(self, images):
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [PIL.Image.fromarray(image) for image in images]
        return pil_images
    
    def is_unsafe(self, x_image):
        safety_checker_input = self.feature_extractor(self.numpy_to_pil(x_image), return_tensors="pt").to(self.device).to(self.safety_dtype)
        try:
            x_checked_image, has_nsfw_concept = self.safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
        except Exception as e:
            # HACK: the safety checker crashes on positives
            return True
        
        # assert x_checked_image.shape[0] == len(has_nsfw_concept)
        for i in range(len(has_nsfw_concept)):
            if has_nsfw_concept[i]:
                return True
        return False

    def run_safety_checker(self, image):
        image_numpy = numpy.asarray(image)
        return self.is_unsafe(image_numpy)

    def verify_generated_video_safety(self, path, cadence=1):
        self.safety_checker.to(self.device).to(self.safety_dtype)
        
        ld = os.listdir(path)
        ld = [l for l in ld if l.endswith('.png')]
        for i in tqdm(range(0, len(ld), cadence)):
            f = ld[i]
            if self.run_safety_checker(PIL.Image.open(os.path.join(path, f))):
                self.safety_checker.to('cpu')
                return False
        self.safety_checker.to('cpu')
        return True

    def is_prompt_safe(self, text):
        return self.pf.is_clean(text)

@to_thread
def make_animation(deforum_settings):

    global config

    settings_json = json.dumps(deforum_settings)
    allowed_params = ';'.join(deforum_settings.keys()).strip(';')

    request = {'settings_json':settings_json, 'allowed_params':allowed_params}

    try:
        response = requests.post(config['URL'], params=request)
        if response.status_code == 200:
            result = response.json()['outdir']
        else:
            print(f"Bad status code {response.status_code}")
            if response.status_code == 422:
                print(response.json()['detail'])
            return ""
        
    except Exception as e:
        traceback.print_exc()
        print(e)
        return ""

    return result#path string

safety = Safety()
semaphore = asyncio.Semaphore(1) # mutex for safety checker access

@to_thread
def check_animation(animation_path):
    global safety
    return safety.verify_generated_video_safety(animation_path)

@to_thread
def check_words(prompts):
    global safety
    return safety.is_prompt_safe(prompts.replace('"', "").replace("'", "").replace('(', "").replace(')', ""))

# Simpifiled version of parse_key_frames from Deforum
# this one converts format like `0:(lol), 20:(kek)` to prompts JSON
def parse_prompts(string, filename='unknown'):
    frames = dict()
    for match_object in string.split(","):
        frameParam = match_object.split(":")
        try:
            frame = frameParam[0].strip()
            frames[frame] = frameParam[1].strip()
        except SyntaxError as e:
            e.filename = filename
            raise e
    if frames == {} and len(string) != 0:
        traceback.print_exc()
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames

def find_animation(d):
    for f in os.listdir(d):
        if f.endswith('.mp4'):
            return os.path.join(d, f)
    return ''

def find_settings(d):
    for f in os.listdir(d):
        if f.endswith('.txt'):
            return os.path.join(d, f)
    return ''

def wrap_value(val:str):
    val = val.strip()
    if len(val) > 0 and not '(' in val and not ')' in val:
        val = f'0: ({val})'
    return val

# Starting the bot part

@bot.event
async def on_ready():
    await bot.tree.sync()
    print(f'We have logged in as {bot.user}')

@bot.hybrid_command(name="deforum", with_app_command=True)
async def deforum(ctx, prompts: str = "", cadence: int = 10, w:int = 512, h: int = 512, fps: int = 15, seed = -1, strength_schedule: str = "0: (0.65)", preview_mode: bool = False, speed_x: str = "0: (0)", speed_y: str = "0: (0)", speed_z: str = "0: (1.75)", rotate_x:str = "0: (0)", rotate_y: str = "0: (0)", rotate_z: str = "0: (0)"):
    await bot.tree.sync()

    print('Received a /deforum command!')
    print(prompts)

    global safety
    global semaphore

    prompts = wrap_value(prompts)
    strength_schedule = wrap_value(strength_schedule)
    speed_x = wrap_value(speed_x)
    speed_y = wrap_value(speed_y)
    speed_z = wrap_value(speed_z)
    rotate_x = wrap_value(rotate_x)
    rotate_y = wrap_value(rotate_y)
    rotate_z = wrap_value(rotate_z)

    chan = ctx.message.channel

    deforum_settings = {'diffusion_cadence':cadence, 'W':w, 'H':h, 'fps':fps, 'seed':seed, 'strength_schedule':strength_schedule, 'motion_preview_mode':preview_mode, 'translation_x':speed_x, 'translation_y':speed_y, 'translation_z':speed_z, 'rotation_3d_x':rotate_x, 'rotation_3d_y':rotate_y, 'rotation_3d_z':rotate_z}

    if len(prompts) > 0:

        print('Checking if the words are safe')

        words_safe = await check_words(prompts)

        if not words_safe:
            print(f'Possible bad words detected from {ctx.message.author.name} (id {ctx.message.author.id})')
            await ctx.reply("Your prompt seems to contain bad words which we can't process due to Discord's TOS, please edit it")
            return
        
        print('Parsing prompts')

        try:
            prompts = parse_prompts(prompts)
        except Exception as e:
            await ctx.reply('Error parsing prompts!')
            return
        
        deforum_settings['prompts'] = prompts
    
    await ctx.reply('Making a Deforum animation...')

    print('Making the animation')

    async with semaphore:
        path = await make_animation(deforum_settings)

        if len(path) > 0:
            print('Animation made. Checking safety')

            if preview_mode:
                is_safe = True
            else:
                is_safe = await check_animation(path)

            if not is_safe:
                print(f'Possible unsafe animation detected from {ctx.message.author.name} (id {ctx.message.author.id})')
                print(f'Used prompts: {prompts}')
                shutil.rmtree(path)
                await ctx.reply("Possible unsafe contents detected in the animation, cannot continue")
                return
            
            anim_file = find_animation(os.path.abspath(path))
            await ctx.send(file=discord.File(anim_file))
            settings_file = find_settings(os.path.abspath(path))
            
            result_seed = -2
            try:
                with open(settings_file, 'r', encoding='utf-8') as sttn:
                    result_settings = json.loads(sttn.read())
                result_seed = result_settings['seed']
            except:
                ...
            #await ctx.send(file=discord.File(settings_file)) # feature for selected users?
            await ctx.reply('Your animation is done!' + (f' Seed used: {result_seed}' if result_seed != -2 else ''))
        else:
            print('Failed to make an animation!')
            traceback.print_exc()
            await ctx.reply('Sorry, there was an error making the animation!')

bot.run(key)

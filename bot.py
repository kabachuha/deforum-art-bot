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
        self.device = torch.device('cpu')#torch.device('cuda')
        self.safety_dtype = torch.float32#torch.float16
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

@to_thread
def make_animation(deforum_settings):

    global config

    settings_json = json.dumps(deforum_settings)
    allowed_params = ';'.join(deforum_settings.keys()).strip(';')

    request = {'settings_json':settings_json, 'allowed_params':allowed_params}

    try:
        response = requests.post(config['URL'], json=request)
        if response.status_code == 200:
            result = response.json()['outdir']
        else:
            return ""
        
    except Exception as e:
        traceback.print_exc()
        print(e)

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
    return safety.is_prompt_safe(prompts)

# Simpifiled version of parse_key_frames from Deforum
# this one converts format like `0:(lol), 20:(kek)` to prompts JSON
def parse_prompts(string, filename='unknown'):
    frames = dict()
    for match_object in string.split(","):
        frameParam = match_object.split(":")
        try:
            frame = frameParam[0]
            frames[frame] = frameParam[1].strip()
        except SyntaxError as e:
            e.filename = filename
            raise e
    if frames == {} and len(string) != 0:
        traceback.print_exc()
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames

def find_animation(dir):
    for f in os.listdir(dir):
        if f.endswith('.mp4'):
            return f
    return ''

# Starting the bot part

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command()
async def deforum(ctx, prompts: str = "", cadence: int = 10):

    print('Received a /deforum command!')
    print(prompts)

    global safety
    global semaphore

    chan = ctx.message.channel

    deforum_settings = {'diffusion_cadence':cadence}

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

        async with semaphore:
            is_safe = await check_animation(path)

        if not is_safe:
            print(f'Possible unsafe animation detected from {ctx.message.author.name} (id {ctx.message.author.id})')
            print(f'Used prompts: {prompts}')
            shutil.rmtree(path)
            await ctx.reply("Possible unsafe contents detected in the animation, cannot continue")
            return
        
        anim_file = find_animation(os.path.abspath(path))
        await bot.send_file(chan, anim_file, filename="Deforum.mp4")
        await ctx.reply('Your animation is done!')
    else:
        print('Failed to make an animation!')
        await ctx.reply('Sorry, there was an error making the animation!')

bot.run(key)

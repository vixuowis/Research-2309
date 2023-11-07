from typing import *
from transformers import load_tool

def load_custom_tools():
    """Load the custom tools for image generation."""
    controlnet_transformer = load_tool("diffusers/controlnet-canny-tool")
    upscaler = load_tool("diffusers/latent-upscaler-tool")

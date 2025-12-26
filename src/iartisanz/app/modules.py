from importlib.resources import files

from iartisanz.modules.generation.generation_module import GenerationModule


TXT2IMG_ICON = files("iartisanz.theme.icons").joinpath("txtimg.png")

MODULES = {
    "Generation": (TXT2IMG_ICON, GenerationModule),
}

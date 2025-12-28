import attr


@attr.s(eq=True, slots=True)
class PreferencesObject:
    intermediate_images: bool = attr.ib(default=False)
    save_image_metadata: bool = attr.ib(default=False)
    hide_nsfw: bool = attr.ib(default=True)
    delete_lora_on_import: bool = attr.ib(default=False)
    delete_model_on_import: bool = attr.ib(default=False)
    delete_model_after_conversion: bool = attr.ib(default=False)

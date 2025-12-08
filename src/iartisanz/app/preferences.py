import attr


@attr.s(eq=True, slots=True)
class PreferencesObject:
    intermediate_images: bool = attr.ib(default=False)
    save_image_metadata: bool = attr.ib(default=False)

import attr


@attr.s(eq=True)
class DirectoriesObject:
    outputs_images = attr.ib(type=str)

import cv2


def fast_upscale_and_denoise(numpy_image, scale: float = 1.0, denoise_strength=3):
    if scale != 1.0:
        numpy_image = cv2.resize(numpy_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    numpy_image = cv2.fastNlMeansDenoisingColored(
        numpy_image, None, h=denoise_strength, hColor=10, templateWindowSize=7, searchWindowSize=21
    )

    return numpy_image

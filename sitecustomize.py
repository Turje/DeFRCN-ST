try:
    from PIL import Image
    # Provide aliases if missing
    if not hasattr(Image, "LINEAR") and hasattr(Image, "BILINEAR"):
        Image.LINEAR = Image.BILINEAR
    if not hasattr(Image, "CUBIC") and hasattr(Image, "BICUBIC"):
        Image.CUBIC = Image.BICUBIC
    if not hasattr(Image, "ANTIALIAS") and hasattr(Image, "LANCZOS"):
        Image.ANTIALIAS = Image.LANCZOS
except Exception:
    pass

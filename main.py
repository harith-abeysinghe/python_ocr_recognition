# opening and viewing images
from PIL import Image

img = Image.open("testimage.jpg")
print(img)
print(img.format, img.mode, img.size, img.width)
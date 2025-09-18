from defisheye import Defisheye

dtype = 'linear'
format = 'fullframe'
fov = 180
pfov = 120

img = "fish-eye_image.jpg"
img_out = f"defished_image_{pfov}_{fov}.jpg"

obj = Defisheye(img, dtype=dtype, format=format, fov=fov, pfov=pfov)

# To save image locally
obj.convert(outfile=img_out)

# To use the converted image in memory

new_image = obj.convert()

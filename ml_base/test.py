from common_lib_import_and_set import *

Y = 130
U = 120
V = 120
R = 1.164 * (Y - 16) + 1.596 * (V - 128)
G = 1.164 * (Y - 16) - 0.813 * (V - 128) - 0.391 * (U - 128)
B = 1.164 * (Y - 16) + 2.018 * (U - 128)

print("-------------------------------")
print(R, G, B)

# image_util.showImage('/E/dataset/COCO/COCO2017/val2017/000000397133.jpg')

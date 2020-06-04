
import imageio
import numpy, sys
img = imageio.imread(r'D:\d3-recon-ml\-Gfp64.png')
numpy.set_printoptions(threshold=sys.maxsize, precision=2)
img = img[:,:,0].flatten()
for i in range(128):
    for j in range(64):
        print("{:0.6f}".format(float(img[i*64 + j])/255.0), end = '')
        print(', ', end = '')
    print()
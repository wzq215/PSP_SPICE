import spiceypy as spice
import furnsh_kernels
from PIL import Image
import datetime
import matplotlib.pyplot as plt
import numpy as np
wispr_time_str = '20210111T080017'
wispr_png_name = 'psp_L3_wispr_'+wispr_time_str+'_V1_1221.png'
wispr_im_path = 'psp/wispr/images/'+wispr_png_name
wispr_im = Image.open(wispr_im_path)
wispr_imarray = np.array(wispr_im)
print(wispr_im.size)
print(wispr_imarray.size)
plt.imshow(wispr_im)
plt.show()
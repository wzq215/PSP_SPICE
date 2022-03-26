import astropy.units as u
import numpy as np
import sunpy.map
from matplotlib import pyplot as plt

# data, header = sunpy.io.fits.read('/Users/ephe/Desktop/PSP原位遥感观测/PSP_SPICE/data/orbit08/20210421/psp_L3_wispr_20210421T000022_V1_1221.fits')[0]
# data, header = sunpy.io.fits.read('data/WISPR_ENC07_L3_FITS/20210116/psp_L3_wispr_20210116T010018_V1_1211.fits')[0]
# data, header = sunpy.io.fits.read('data/WISPR_ENC07_L3_FITS/20210117/psp_L3_wispr_20210117T163027_V1_1211.fits')[0]
data, header = sunpy.io.fits.read('data/WISPR_ENC07_L3_FITS/20210117/psp_L3_wispr_20210117T130027_V1_1211.fits')[0]
print(header)
print(header['BUNIT'])
header['BUNIT'] = 'MSB'
a_map = sunpy.map.Map(data, header)

# my_fig = plt.figure(figsize=(9, 9))
# ax = plt.subplot(111)
# astropy_norm = simple_norm(true_data, stretch='log', log_a=500)
# norm_SL = colors.SymLogNorm(linthresh=0.001 * 1e-10, linscale=0.1 * 1e-10, vmin=-0.0038 * 1e-10, vmax=0.14 * 1e-10)
plt.imshow(data)
plt.xlabel('longitude (pixel)')
plt.ylabel('latitude (pixel)')
plt.gca().invert_yaxis()
plt.title('2021-01-17 13:00:27')
# my_mappabel = matplotlib.cm.ScalarMappable(cmap=my_colormap, norm=norm_SL)
plt.colorbar()
# plt.set_cmap('gist_ncar')
plt.clim([1e-14,1e-12])
plt.show()
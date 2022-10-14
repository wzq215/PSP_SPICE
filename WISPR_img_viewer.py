from datetime import datetime, timedelta

import sunpy.map
from matplotlib import pyplot as plt

orbit_path = 'orbit09'

rdstep = 60
dtnow = datetime(2021, 8, 6, 15, 7, 15)
dtpre = dtnow - timedelta(minutes=rdstep, seconds=0)

dtnow_str = dtnow.strftime('%Y%m%dT%H%M%S')
dtpre_str = dtpre.strftime('%Y%m%dT%H%M%S')
print('Current Time: ', dtnow_str)
print('Previous Time: ', dtpre_str)

data_pre, header_pre = sunpy.io.fits.read(
    'data/' + orbit_path + '/' + dtpre.strftime('%Y%m%d') + '/psp_L3_wispr_' + dtpre_str + '_V1_1221.fits')[0]
data_now, header_now = sunpy.io.fits.read(
    'data/' + orbit_path + '/' + dtnow.strftime('%Y%m%d') + '/psp_L3_wispr_' + dtnow_str + '_V1_1221.fits')[0]
data_rd = data_now - data_pre

plt.figure()
plt.imshow(data_now)
plt.xlabel('longitude (pixel)')
plt.ylabel('latitude (pixel)')
plt.gca().invert_yaxis()
plt.title(dtnow_str)
plt.colorbar()
plt.set_cmap('gist_gray')
plt.clim([1e-14, 3e-12])
plt.gca().set_aspect(1)
plt.show()

plt.subplot(121)
plt.imshow(data_rd)
plt.xlabel('longitude (pixel)')
plt.ylabel('latitude (pixel)')
plt.gca().invert_yaxis()
plt.title(dtnow_str + ' Running Difference (' + str(rdstep) + 'min)')
plt.colorbar()
plt.set_cmap('cool')
plt.clim([-1e-14, 1e-14])
plt.gca().set_aspect(1)

plt.subplot(122)
plt.imshow(data_now)
plt.xlabel('longitude (pixel)')
plt.ylabel('latitude (pixel)')
plt.gca().invert_yaxis()
plt.title(dtnow_str)
plt.colorbar()
plt.set_cmap('gist_gray')
plt.clim([1e-14, 2e-12])
plt.gca().set_aspect(1)
plt.show()

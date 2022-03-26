import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as markers
from mpl_toolkits.mplot3d import Axes3D
import spiceypy as spice
from datetime import datetime
import furnsh_kernels
import pandas as pd
from wispr_insitu import plot_frames, find_PSP_in_WISPR, find_WISPR_for_PSP, datetimestr2et
from PIL import Image
from load_psp_data import load_RTN_1min_data
from matplotlib.collections import LineCollection
import imageio

AU = 1.49e8  # km

# UTC2ET
start_time = '2021-01-15'
stop_time = '2021-01-21'
start_dt = datetime.strptime(start_time, '%Y-%m-%d')
stop_dt = datetime.strptime(stop_time, '%Y-%m-%d')
utc = [start_dt.strftime('%b %d, %Y'), stop_dt.strftime('%b %d, %Y')]
etOne = spice.str2et(utc[0])
etTwo = spice.str2et(utc[1])

# Epochs
step = 100
times = [x * (etTwo - etOne) / step + etOne for x in range(step)]

## Plot PSP&SUN orbit for specified time range
psp_positions, psp_LightTimes = spice.spkpos('SPP', times, 'SPP_HCI', 'NONE', 'SUN')
sun_positions, sun_LightTimes = spice.spkpos('SUN', times, 'SPP_HCI', 'NONE', 'SUN')
psp_positions = psp_positions.T  # psp_positions is shaped (4000, 3), let's transpose to (3, 4000) for easier indexing
psp_positions = psp_positions / AU
sun_positions = sun_positions.T
sun_positions = sun_positions / AU

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(psp_positions[0], psp_positions[1], psp_positions[2], c=times, cmap='jet')
ax.scatter(sun_positions[0], sun_positions[1], sun_positions[2], c='r')
# ax.scatter(1, 0, 0, c='red')
ax.set_xlabel('X (AU)')
ax.set_ylabel('Y (AU)')
ax.set_zlabel('Z (AU)')
plt.title('PSP' + '(' + start_time + '_' + stop_time + ')')
plt.show()

# plot_frames(times[0])
camera = 'inner'
size = 40
filelist = os.listdir('psp/wispr/images/' + camera + '/')
fnamelist = []
for filename in filelist:
    if re.match(r'^psp_L3_wispr_2021011', filename) != None:
        fnamelist.append(filename)
# filelist = re.findall(r'^psp_L3_wispr_20210115',filelist)
fnamelist.sort(key=lambda x: datetime.strptime(x[13:28], '%Y%m%dT%H%M%S'))
frames = []
for fname in fnamelist:
    wispr_time_str = fname[13:28]
    # wispr_time_str = '20210115T194421'
    # wispr_png_name = 'psp_L3_wispr_'+wispr_time_str+'_V1_1221.png'
    wispr_im_path = 'psp/wispr/images/' + camera + '/' + fname
    wispr_im = Image.open(wispr_im_path)
    wispr_et = spice.datetime2et(datetime.strptime(wispr_time_str, '%Y%m%dT%H%M%S'))
    inner_lst, outer_lst = find_PSP_in_WISPR(times, wispr_et)
    # print(type(inner_lst[:, 0]))
    # print(inner_lst[:, 0].astype(int))
    # print(times[inner_lst[:,0]]-times[0])
    # plt.figure()
    imsize = np.array(wispr_im.size)
    boresight = imsize / 2
    # if inner_lst.size > 0:
    #     # plt.figure()
    #     # plt.imshow(wispr_im)
    #     im = plt.scatter(inner_lst[:, 2], inner_lst[:, 3], c=inner_lst[:, 0])
    #     cbar = plt.colorbar(im)
    #     cbar.set_ticks(cbar.ax.get_yticks())
    #     cbar.set_ticklabels([spice.et2datetime(x).strftime('%Y-%m-%d %H:%m:%S') for x in inner_lst[:, 0]])
    #     plt.xlabel('longitude(deg)')
    #     plt.ylabel('latitude(deg)')
    #     plt.xlim([-20, 20])
    #     plt.ylim([-20, 20])
    #     plt.title('INNER FOV')
    #     plt.show()

    fig, axs = plt.subplots(2, 1, sharex=False, sharey=False, figsize=(12, 15), gridspec_kw={'height_ratios': [4, 1]})
    axs[0].imshow(wispr_im)
    if outer_lst.size > 0:
        imlon = outer_lst[:, 2] * 480 / size + boresight[0]
        imlat = outer_lst[:, 3] * 480 / size + boresight[1]
        im = axs[0].scatter(imlon, imlat, c=outer_lst[:, 0], cmap='jet', vmin=times[0], vmax=times[-1], s=50,
                            marker='.', alpha=0.5)

    axs[0].set_xlabel('longitude(deg)', fontsize=20)
    axs[0].set_ylabel('latitude(deg)', fontsize=20)
    cbar = plt.colorbar(im, ax=axs[0])
    cbar.set_ticks(cbar.ax.get_yticks())
    cbar.set_ticklabels([spice.et2datetime(x).strftime('%Y-%m-%d %H:%m:%S') for x in cbar.ax.get_yticks()])
    axs[0].set_title('INNER FOV' + wispr_time_str, fontsize=20)
    RTN = load_RTN_1min_data('20210115', '20210121')
    BR = np.array(RTN['psp_fld_l2_mag_RTN_1min'][:, 0])
    RTN_ets = spice.datetime2et(RTN['epoch_mag_RTN_1min'])
    points = np.array([RTN_ets, BR]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(RTN_ets[0], RTN_ets[-1])
    lc = LineCollection(segments, cmap='jet', norm=norm)
    # Set the values used for colormapping
    lc.set_array(RTN_ets)
    lc.set_linewidth(2)
    line = axs[1].add_collection(lc)
    # plt.colorbar(line, ax=axs[1])
    axs[1].set_xlim(RTN_ets[0], RTN_ets[-1])
    axs[1].set_ylim(np.nanmin(BR), np.nanmax(BR))
    # print(axs[0])
    # print(axs[1])
    axs[1].set_xticklabels([spice.et2datetime(x).strftime('%Y%m%d') for x in axs[1].get_xticks()])
    axs[1].set_ylabel('B_R (nT)', fontsize=20)
    # plt.plot(RTN['epoch_mag_RTN_1min'],RTN['psp_fld_l2_mag_RTN_1min'][:,0],linewidth=1)
    # plt.scatter(RTN['epoch_mag_RTN_1min'],RTN['psp_fld_l2_mag_RTN_1min'][:,0],c=RTN_ets,s=10,label=RTN['label_RTN'][0])
    fig.savefig('figures/' + camera + wispr_time_str + '.png')
    plt.close(fig)
    frames.append(imageio.imread('figures/' + camera + wispr_time_str + '.png'))

imageio.mimsave('movies/' + camera + start_time + '-' + stop_time + '.mp4', frames, fps=12)

# plt.show()
# if inner_lst.size > 0:
#     plt.figure()
#     # plt.imshow(wispr_im)
#     im = plt.scatter(inner_lst[:, 2], inner_lst[:, 3], c=inner_lst[:, 0])
#     cbar = plt.colorbar(im)
#     cbar.set_ticks(cbar.ax.get_yticks())
#     cbar.set_ticklabels([spice.et2datetime(x).strftime('%Y-%m-%d %H:%m:%S') for x in inner_lst[:, 0]])
#     plt.xlabel('longitude(deg)')
#     plt.ylabel('latitude(deg)')
#     plt.xlim([-20, 20])
#     plt.ylim([-20, 20])
#     plt.title('INNER FOV')
#     plt.show()

# spatial_resolution = 6.4 / 60  # 6.4 arcmin
# pixel_size_inner = 1.2 / 60  # 1.2 arcmin
# pixel_size_outer = 1.7 / 60  # 1.2 arcmin
# if inner_lst.size > 0:
#     plt.figure()
#     im = plt.scatter(inner_lst[:, 2] // pixel_size_inner, inner_lst[:, 3] // pixel_size_inner, c=inner_lst[:, 0])
#     cbar = plt.colorbar(im)
#     cbar.set_ticks(cbar.ax.get_yticks())
#     cbar.set_ticklabels([spice.et2datetime(x).strftime('%Y-%m-%d %H:%m:%S') for x in inner_lst[:, 0]])
#     plt.xlabel('n_pixels (from boresight)')
#     plt.ylabel('n_pixels (from boresight)')
#     plt.xlim([-20 // pixel_size_inner, 20 // pixel_size_inner])
#     plt.ylim([-20 // pixel_size_inner, 20 // pixel_size_inner])
#     plt.title('INNER FOV')
#     plt.show()
#
# if outer_lst.size > 0:
#     plt.figure()
#     im = plt.scatter(outer_lst[:, 2], outer_lst[:, 3], c=outer_lst[:, 0])
#     cbar = plt.colorbar(im)
#     cbar.set_ticks(cbar.ax.get_yticks())
#     cbar.set_ticklabels([spice.et2datetime(x).strftime('%Y-%m-%d %H:%m:%S') for x in outer_lst[:, 0]])
#     plt.xlabel('longitude(deg)')
#     plt.ylabel('latitude(deg)')
#     plt.xlim([-29, 29])
#     plt.ylim([-29, 29])
#     plt.title('OUTER FOV')
#     plt.show()
#     inner, outer = find_WISPR_for_PSP(times[-1], times[0:14])
#
#     '''TD Phan reconnection'''

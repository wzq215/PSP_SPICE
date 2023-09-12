from datetime import datetime, timedelta

import astropy.units as u
import astropy.constants as const
import matplotlib.colors as mcolor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pfsspy
import spiceypy as spice
import sunpy.io.fits
import sunpy.map
from astropy.coordinates import SkyCoord
from scipy import interpolate
from scipy.interpolate import interp2d

from load_psp_data import load_RTN_1min_data
from plot_body_positions import get_rlonlat_psp_carr
import furnsh_kernels

datetime_beg = datetime(2021, 4, 28, 15, 30, 0)
datetime_end = datetime(2021, 4, 30, 4, 30, 0)

timestep = timedelta(minutes=30)
steps = (datetime_end - datetime_beg) // timestep + 1
dttimes = np.array([x * timestep + datetime_beg for x in range(steps)])
print(dttimes[0:-1:24])
times = spice.datetime2et(dttimes)

r_psp_carr_br, psp_p, psp_t = get_rlonlat_psp_carr(dttimes)

df_trace = pd.read_csv('E8trace0429_ss27.csv')
field_lines = np.load('save_field_lines_0429_ss27.npy', allow_pickle=True)

ss_lon = df_trace['lon_footpoint_on_SourceSurface_deg']
photo_lon = df_trace['MFL_photosphere_lon_deg']
psp_lon = np.rad2deg(psp_p)
ss_lat = (90 + df_trace['lat_footpoint_on_SourceSurface_deg'])
photo_lat = (90 + df_trace['MFL_photosphere_lat_deg'])
psp_lat = np.rad2deg(psp_t)

ss_lon_ind = df_trace['lon_footpoint_on_SourceSurface_deg'] * 1440 / 360
photo_lon_ind = df_trace['MFL_photosphere_lon_deg'] * 1440 / 360
psp_lon_ind = np.rad2deg(psp_p) * 1440 / 360
ss_lat_ind = (90 + df_trace['lat_footpoint_on_SourceSurface_deg']) * 720 / 180
photo_lat_ind = (90 + df_trace['MFL_photosphere_lat_deg']) * 720 / 180
psp_lat_ind = np.rad2deg(psp_t) * 720 / 180
expansion_factor = df_trace['Expansion_Factor']

# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# import astropy.constants as const
# for field_line in field_lines:
#     coords = field_line
#     try:
#         coords.representation_type = 'cartesian'
#         ax.plot(coords.x / const.R_sun,
#             coords.y / const.R_sun,
#             coords.z / const.R_sun,
#             linewidth=1)
#     except:
#         print('Nan')
# plt.show()


beg_time = datetime(2021, 4, 28, 15, 30)
end_time = datetime(2021, 4, 30, 4, 30)
mag_RTN = load_RTN_1min_data(beg_time.strftime('%Y%m%d'), end_time.strftime('%Y%m%d'))
epochmag = mag_RTN['epoch_mag_RTN_1min']
timebinmag = (epochmag > beg_time) & (epochmag < end_time)
epochmag = epochmag[timebinmag]

Br = mag_RTN['psp_fld_l2_mag_RTN_1min'][timebinmag, 0]
Bt = mag_RTN['psp_fld_l2_mag_RTN_1min'][timebinmag, 1]
Bn = mag_RTN['psp_fld_l2_mag_RTN_1min'][timebinmag, 2]
Babs = np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2)

f = interpolate.interp1d(np.array((epochmag - epochmag[0]) / timedelta(days=1), dtype='float64'), Br, kind='previous',
                         fill_value='extrapolate')
Br_interp = f(np.array((dttimes - epochmag[0]) / timedelta(days=1),
                       dtype='float64'))  # use interpolation function returned by `interp1d`

cr_map_2243, cr_map_2243header = sunpy.io.fits.read('/Users/ephe/PFSS_Data/mrmqs210427t2029c2243_000.fits')[0]

Brqdata, Brqheader = sunpy.io.fits.read('/Users/ephe/Downloads/hmi.q_synop.2243.Brq.fits')
Brq_data = Brqheader.data

chmapdata, chmapheader = sunpy.io.fits.read('/Users/ephe/Downloads/hmi.q_synop.2243.chmap.fits')
chmap_data = chmapheader.data

Qdata, Qheader = sunpy.io.fits.read('/Users/ephe/Downloads/hmi.q_synop.2243.slogQ.fits')
Q_data = Qheader.data

from sunpy.net import Fido, attrs as a

wavelength = 195
query = (a.Time('2004/9/5', '2004/9/5')) & a.Instrument('SECCHI') & a.Source('STEREO_A') & a.Detector(
    'EUVI') & a.Wavelength(wavelength * u.Angstrom)
results = Fido.search(query)
print(results)
data_dir = '/Users/ephe/STEREO_Data/EUVI/' + str(wavelength) + '/'
dld_file = Fido.fetch(results, path=data_dir + '{file}', overwrite=False)

quit()
# data,header = sunpy.io.fits.read('/Users/ephe/STEREO_Data/EUVI20210429_081910_n4eua.fts')[0]

long0 = 105
Magmap, Magmapheader = sunpy.io.fits.read('/Users/ephe/PFSS_Data/mrzqs210429t0004c2243_105.fits')[0]
gong_map = sunpy.map.Map('/Users/ephe/PFSS_Data/mrzqs210429t0004c2243_105.fits.gz')
nrho = 35
rss = 2.7
pfss_in = pfsspy.Input(gong_map, nrho, rss)
pfss_out = pfsspy.pfss(pfss_in)

data, header = sunpy.io.fits.read('/Users/ephe/STEREO_Data/EUVI/195/20210428_235530_n4eua.fts')[0]
print(header)
sta_map = sunpy.map.Map(data, header)
shape_out = (720, 1440)
frame_out = SkyCoord(0, 0, unit=u.deg,
                     frame="heliographic_carrington",
                     obstime=sta_map.date,
                     rsun=sta_map.coordinate_frame.rsun,
                     observer='earth')
header = sunpy.map.make_fitswcs_header(shape_out,
                                       frame_out,
                                       scale=(360 / shape_out[1],
                                              180 / shape_out[0]) * u.deg / u.pix,
                                       projection_code="CAR")
outmap = sta_map.reproject_to(header)
top_right = SkyCoord(180 * u.deg, 90 * u.deg, frame=outmap.coordinate_frame)
bottom_left = SkyCoord(0 * u.deg, -90 * u.deg, frame=outmap.coordinate_frame)
outmap_sub = outmap.submap(bottom_left, top_right=top_right)

print(np.shape(Q_data))

# plt.pcolormesh(Q_data[:,:,320],shading='auto',edgecolors='face')
# plt.set_cmap('RdBu_r')
# plt.colorbar()
# plt.clim([-3,3])
# plt.show()


R = np.linspace(0, 2.5, 10)
Lat = np.linspace(-90, 90, 721)
f = interp2d(Lat, R, Q_data[:, :, 320], kind='linear')
ynew = np.linspace(0, 2.5, 40)
xnew = np.linspace(-90, 90, 721)
data1 = f(xnew, ynew)
Xn, Yn = np.meshgrid(xnew, ynew)
# plt.subplot(3, 2, 5)
# plt.pcolormesh(Xn, Yn, data1, cmap='RdBu')
# plt.clim([-3,3])
# plt.show()

fig = plt.figure()
# plt.subplot(2,1,1)
# plt.imshow(Magmap[:,np.arange(360)])
# plt.set_cmap('RdBu_r')
# # plt.colorbar()
# plt.clim([-8,8])
# plt.gca().invert_yaxis()
# plt.xticks(np.arange(0,360,step=30),np.arange(0,360,step=30))
# plt.yticks(np.arange(0,180,step=30),np.arange(-90,90,step=30))
# plt.gca().set_aspect(1)
fig = plt.figure()
# m = pfss_in.map
# ax = fig.add_subplot(2, 1, 1, projection=m)
# m.plot()
# ax.set_title('Input GONG magnetogram')
from astropy import constants as const
from pfsspy import tracing

ss_br = pfss_out.source_surface_br
ax = fig.add_subplot()
# Number of steps in cos(latitude)
nsteps = 90
lon_1d = np.linspace(0, 2 * np.pi, nsteps * 2 + 1)
lat_1d = np.arcsin(np.linspace(-1, 1, nsteps + 1))
lon, lat = np.meshgrid(lon_1d, lat_1d, indexing='ij')
lon, lat = lon * u.rad, lat * u.rad
seeds = SkyCoord(lon.ravel(), lat.ravel(), const.R_sun, frame=pfss_out.coordinate_frame)
tracer = tracing.FortranTracer(max_steps=2000)
field_lines_openclose = tracer.trace(seeds, pfss_out)
cmap = mcolor.ListedColormap(['tab:blue', 'white', 'tab:red'])
norm = mcolor.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], ncolors=3)
pols = field_lines_openclose.polarities.reshape(2 * nsteps + 1, nsteps + 1).T
ax.contourf(np.rad2deg(lon_1d), np.rad2deg(lat_1d), pols, norm=norm, cmap=cmap, alpha=0.9)
for field_line in field_lines:
    try:
        field_line.representation_type = 'spherical'
        ax.plot((field_line.lon), (field_line.lat), 'gray', linewidth=0.5)
        print(field_line.lon[0] * 360 / 360)

    except:
        print('Nan')
print(photo_lon_ind)
lon_1d = np.linspace(0, 2 * np.pi, 361)
lat_1d = np.arcsin(np.linspace(-1, 1, 181))
plt.scatter(ss_lon_ind / 4, ss_lat_ind / 4 - 90, c=np.sign(Br_interp), s=8, cmap='RdBu_r', label='FP @Source Surface')
plt.scatter(photo_lon_ind / 4, photo_lat_ind / 4 - 90, s=5, c='green', marker='o',
            label='FP @Photosphere')  # c=np.arange(len(df_trace['Epoch']))
plt.scatter(ss_lon_ind[[0, 44, 74]] / 4, ss_lat_ind[[0, 44, 74]] / 4 - 90, c='white', s=10)
# ss_br.plot()
print(pfss_out.source_surface_pils)
ax.scatter(pfss_out.source_surface_pils[0].lon.to_value(u.deg), pfss_out.source_surface_pils[0].lat.to_value(u.deg),
           s=1)
# ax.contour(np.sqrt(pfss_out.bg[:,:,-1,0]**2+pfss_out.bg[:,:,-1,1]**2+pfss_out.bg[:,:,-1,2]**2))
ax.set_ylabel('latitude')
ax.set_ylabel('longitude')
ax.set_title('Open (blue/red) and closed (black) field')
# ax.set_aspect(0.5 * 360 / 2)
# quit()

# plt.show()
# # fig = plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(Magmap[:, np.arange(360) - 107])
# # B_pfss = pfss_out.bc
# # plt.contour(B_pfss[:,:,-1,0])
# plt.set_cmap('RdBu_r')
# plt.colorbar()
# plt.clim([-8, 8])
# plt.gca().invert_yaxis()
# plt.xticks(np.arange(0, 360, step=30), np.arange(0, 360, step=30))
# plt.yticks(np.arange(0, 180, step=30), np.arange(-90, 90, step=30))
# plt.gca().set_aspect(1)
# # plt.show()
# # quit()
#
# for field_line in field_lines:
#     try:
#         field_line.representation_type = 'spherical'
#         plt.plot(field_line.lon * 360 / 360, (field_line.lat + 90 * u.deg) * 180 / 180, 'gray', linewidth=0.5)
#         print(field_line.lon[0] * 360 / 360)
#
#     except:
#         print('Nan')
# print(photo_lon_ind)
# plt.scatter(ss_lon_ind / 4, ss_lat_ind / 4, c=np.sign(Br_interp), s=8, cmap='RdBu_r', label='FP @Source Surface')
# plt.scatter(photo_lon_ind / 4, photo_lat_ind / 4, s=5, c='green', marker='o',
#             label='FP @Photosphere')  # c=np.arange(len(df_trace['Epoch']))
# plt.scatter(ss_lon_ind[[0, 44, 74]] / 4, ss_lat_ind[[0, 44, 74]] / 4, c='white', s=10)
# plt.xlim([0, 180])
# plt.title('$Br [nT]$')
# plt.ylabel('Latitude')
#
# ax2 = fig.add_subplot(1, 2, 2, projection=outmap_sub)
#
# # fig = plt.figure()
# outmap_sub.plot(axes=ax2,clip_interval=(10., 99.9) * u.percent)
# plt.colorbar()
# print(outmap_sub)
# for field_line in field_lines:
#     try:
#         field_line.representation_type = 'spherical'
#         plt.plot(field_line.lon * 1440 / 360, (field_line.lat + 90 * u.deg) * 720 / 180, 'gray', linewidth=0.5)
#         print(field_line.lon[0] * 1440 / 360)
#
#     except:
#         print('Nan')
# print(photo_lon_ind)
# plt.scatter(ss_lon_ind, ss_lat_ind, c=np.sign(Br_interp), s=8, cmap='RdBu_r', label='FP @Source Surface')
# plt.scatter(photo_lon_ind, photo_lat_ind, s=5, c='white', marker='o',
#             label='FP @Photosphere')  # c=np.arange(len(df_trace['Epoch']))
# # ax2.set_xticks(np.arange(0,720,step=120),np.arange(0,180,step=30))
# # ax2.set_yticks(np.arange(0,720,step=120),np.arange(-90,90,step=30))
# ax2.set_xlabel(' ')
# ax2.set_ylabel(' ')
# ax2.set_title('193 A')
# # plt.colorbar()
# plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
plt.subplot(2, 2, 1)
plt.imshow(Magmap)
plt.set_cmap('RdBu_r')
# plt.colorbar()
plt.clim([-5, 5])
plt.gca().invert_yaxis()
plt.xticks(np.arange(0, 360, step=30), np.arange(0, 360, step=30))
plt.yticks(np.arange(0, 180, step=30), np.arange(-90, 90, step=30))
plt.gca().set_aspect(1)

for field_line in field_lines:
    try:
        field_line.representation_type = 'spherical'
        plt.plot(field_line.lon * 360 / 360, (field_line.lat + 90 * u.deg) * 180 / 180, 'gray', linewidth=0.5)
        print(field_line.lon[0] * 360 / 360)

    except:
        print('Nan')
print(photo_lon_ind)
plt.scatter(ss_lon_ind / 4, ss_lat_ind / 4, c=np.sign(Br_interp), s=8, cmap='RdBu_r', label='FP @Source Surface')
plt.scatter(photo_lon_ind / 4, photo_lat_ind / 4, s=5, c='green', marker='o',
            label='FP @Photosphere')  # c=np.arange(len(df_trace['Epoch']))
plt.scatter(ss_lon_ind[[0, 44, 74]] / 4, ss_lat_ind[[0, 44, 74]] / 4, c='white', s=10)
plt.xlim([0, 180])
plt.title('$Br [nT]$')
plt.ylabel('Latitude')
plt.colorbar()

ax2 = fig.add_subplot(2, 2, 2, projection=outmap_sub)
# fig = plt.figure()
outmap_sub.plot(clip_interval=(15., 99.9) * u.percent)
print(outmap_sub)
for field_line in field_lines:
    try:
        field_line.representation_type = 'spherical'
        plt.plot(field_line.lon * 1440 / 360, (field_line.lat + 90 * u.deg) * 720 / 180, 'gray', linewidth=0.5)
        print(field_line.lon[0] * 1440 / 360)

    except:
        print('Nan')
print(photo_lon_ind)
plt.scatter(ss_lon_ind, ss_lat_ind, c=np.sign(Br_interp), s=8, cmap='RdBu_r', label='FP @Source Surface')
plt.scatter(photo_lon_ind, photo_lat_ind, s=5, c='white', marker='o',
            label='FP @Photosphere')  # c=np.arange(len(df_trace['Epoch']))
# ax2.set_xticks(np.arange(0,720,step=120),np.arange(0,180,step=30))
# ax2.set_yticks(np.arange(0,720,step=120),np.arange(-90,90,step=30))
ax2.set_xlabel(' ')
ax2.set_ylabel(' ')
ax2.set_title('193 A')
plt.colorbar()
# plt.colorbar()
# plt.clim(0,4000)
# plt.gca().set_aspect(1)

# plt.imshow(chmap_data)
# plt.set_cmap('RdBu_r')
# # plt.colorbar()
# plt.clim([-1,1])
# plt.gca().invert_yaxis()
# plt.xticks(np.arange(0,1440,step=120),np.arange(0,360,step=30))
# plt.yticks(np.arange(0,720,step=120),np.arange(-90,90,step=30))
# # for i in range(len(times)):
# #     plt.plot([ss_lon_ind[i], photo_lon_ind[i]], [ss_lat_ind[i], photo_lat_ind[i]], 'gray', linewidth=0.5)
# import astropy.units as u
# for field_line in field_lines:
#     field_line.representation_type = 'spherical'
#     plt.plot(field_line.lon* 1440/360,(field_line.lat+90*u.deg)* 720/180,'gray',linewidth=0.5)
# plt.scatter(ss_lon_ind, ss_lat_ind, c=np.sign(Br_interp), s=8,cmap='RdBu_r',label='Sub-PSP Points')
# plt.scatter(photo_lon_ind,photo_lat_ind,s=5,c='green',marker='o',label='Foot Points')#c=np.arange(len(df_trace['Epoch']))
# plt.gca().set_aspect(1)
# plt.xlim([0,720])
# plt.title('CH map')
# plt.legend()

# plt.subplot(2,2,3)
ax3 = fig.add_subplot(2, 2, 3)
# plt.figure()
plt.imshow(Q_data[1, :, :])
plt.set_cmap('RdBu_r')
# plt.colorbar()
plt.clim([-3, 3])
plt.gca().invert_yaxis()
plt.xticks(np.arange(0, 1440, step=120), np.arange(0, 360, step=30))
plt.yticks(np.arange(0, 720, step=120), np.arange(-90, 90, step=30))
for field_line in field_lines:
    try:
        field_line.representation_type = 'spherical'
        plt.plot(field_line.lon * 1440 / 360, (field_line.lat + 90 * u.deg) * 720 / 180, 'gray', linewidth=0.5)
        print(field_line.lon[0] * 1440 / 360)

    except:
        print('Nan')
print(photo_lon_ind)
plt.scatter(ss_lon_ind, ss_lat_ind, c=np.sign(Br_interp), s=8, cmap='RdBu_r', label='FP @Source Surface')
plt.scatter(photo_lon_ind, photo_lat_ind, s=5, c='green', marker='o',
            label='FP @Photosphere')  # c=np.arange(len(df_trace['Epoch']))
plt.gca().set_aspect(1)
plt.xlim([0, 720])
plt.title('slogQ at photosphere')
plt.ylabel('Latitude')
plt.xlabel('Lontitude')
plt.legend()
plt.colorbar()
# plt.show()
# plt.subplot(2,2,4)
ax4 = fig.add_subplot(2, 2, 4)
plt.imshow(Q_data[9, :, :])
plt.set_cmap('RdBu_r')
# plt.colorbar()
plt.clim([-3, 3])
plt.gca().invert_yaxis()
plt.xticks(np.arange(0, 1440, step=120), np.arange(0, 360, step=30))
plt.yticks(np.arange(0, 720, step=120), np.arange(-90, 90, step=30))
# for i in range(len(times)):
#     plt.plot([psp_lon_ind[i], ss_lon_ind[i]], [psp_lat_ind[i], ss_lat_ind[i]], 'gray', linewidth=0.5)
plt.scatter(ss_lon_ind, ss_lat_ind, c=np.sign(Br_interp), s=8, cmap='RdBu_r', label='FP @Source Surface')
# plt.scatter(ss_lon_ind,ss_lat_ind,s=5,c='green',marker='o',label='Foot Points')#c=np.arange(len(df_trace['Epoch']))
plt.gca().set_aspect(1)
plt.xlim([0, 720])
plt.title('slogQ at source surface')
plt.legend()
plt.xlabel('Lontitude')
plt.colorbar()

plt.show()
Qseries = []
EFseries = []
print(photo_lon_ind)
print(photo_lat_ind)

for i in range(len(times)):
    # print(i)
    if np.isnan(photo_lat_ind[i]):
        Qseries.append(np.nan)
        EFseries.append(np.nan)
    else:
        Q_tmp = Q_data[0, int(photo_lon_ind[i]), int(photo_lat_ind[i])]
        Qseries.append(Q_tmp)
        Br_photo = Brq_data[0, int(photo_lon_ind[i]), int(photo_lat_ind[i])]
        Br_ss = Brq_data[9, int(ss_lon_ind[i]), int(ss_lat_ind[i])]
        EF_tmp = (1 / 2.5) ** 2 * (Br_photo / Br_ss)
        EFseries.append(EF_tmp)
plt.figure()
plt.subplot(3, 1, 1)
plt.title('logQ')
plt.plot(df_trace['Epoch'], np.abs(Qseries))
plt.xticks(df_trace['Epoch'][0:96:16])
plt.subplot(3, 1, 2)
plt.title('Expansion Factor')
plt.plot(df_trace['Epoch'], expansion_factor)
# plt.plot([datetime(2021,4,28,15,30),datetime(2021,4,28,15,30)],[0,70])
# plt.plot([datetime(2021,4,29,13,40),datetime(2021,4,29,13,40)],[0,70])
# plt.plot([datetime(2021,4,30,4,0),datetime(2021,4,30,4,0)],[0,70])
plt.xticks(df_trace['Epoch'][0:96:16])
plt.subplot(3, 1, 3)
plt.plot(epochmag, Br)
plt.title('Br')
plt.show()

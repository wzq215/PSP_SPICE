import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.colors as mcolor

from scipy.interpolate import interp1d

from sunpy.net import Fido, attrs as a
import sunpy.io.fits
import sunpy.map

import astropy.constants as const
import astropy.units as u
from astropy.coordinates import SkyCoord

import pfsspy

# 这两个函数是我写的，为了获取PSP的RTN数据和卡林顿坐标系内坐标。应该可以改成其他的
from load_psp_data import load_RTN_1min_data
from plot_body_positions import get_rlonlat_psp_carr

# DOWNLOAD STEREO DATA
# wavelength = 195
# time_beg_str = '2021/4/28'
# time_end_str = '2021/4/30'
# query = (a.Time(time_beg_str,time_end_str)) & a.Instrument('SECCHI') & a.Source('STEREO_A') & a.Detector('EUVI') & a.Wavelength(wavelength*u.Angstrom)
# results = Fido.search(query)
# print(results)
# data_dir = '/Users/ephe/STEREO_Data/EUVI/'+str(wavelength)+'/'
# dld_file = Fido.fetch(results,path=data_dir+'{file}',overwrite=False)

# READ TRACING BACK DATA
datetime_beg = datetime(2021, 4, 28, 15, 30, 0)
datetime_end = datetime(2021, 4, 30, 4, 30, 0)
df_trace = pd.read_csv('E8trace0429_ss27.csv')
field_lines_psp = np.load('save_field_lines_0429_ss27.npy', allow_pickle=True)
# 这里读的文件是从two_step_ballistic_backmapping_method.two_step_backmapping里面输出的数据存起来的，可能需要再到two_step_backmapping里面改一下输出
# ----------------给two_step_backmapping套的皮----------------------
# def psp_backmap(epoch, r_source_surface_rs=2.5):
#     pmom_SPI = load_spi_data(epoch[0].strftime('%Y%m%d'), epoch[-1].strftime('%Y%m%d'))
#     epochpmom = pmom_SPI['Epoch']
#     timebinpmom = (epochpmom > epoch[0]) & (epochpmom < epoch[-1])
#     epochpmom = epochpmom[timebinpmom]
#     densp = pmom_SPI['DENS'][timebinpmom]
#     vp_r = pmom_SPI['VEL_RTN_SUN'][timebinpmom, 0]
#     vp_t = pmom_SPI['VEL_RTN_SUN'][timebinpmom, 1]
#     vp_n = pmom_SPI['VEL_RTN_SUN'][timebinpmom, 2]
#     # r_source_surface_rs = 2.
#     dir_data_PFSS = '/Users/ephe/PFSS_data/'
#
#     # Vsw_r_interp = np.interp(np.array((epoch-epochpmom[0])/timedelta(days=1),dtype='float64'),np.array((epochpmom-epochpmom[0])/timedelta(days=1),dtype='float64'),vp_r)
#
#     f = interpolate.interp1d(np.array((epochpmom - epochpmom[0]) / timedelta(days=1), dtype='float64'), vp_r,
#                              kind='previous', fill_value='extrapolate')
#
#     Vsw_r_interp = f(np.array((epoch - epochpmom[0]) / timedelta(days=1),
#                               dtype='float64'))  # use interpolation function returned by `interp1d`
#     plt.plot(np.array((epochpmom - epochpmom[0]) / timedelta(days=1), dtype='float64'))
#     plt.plot(np.array((epoch - epochpmom[0]) / timedelta(days=1), dtype='float64'))
#     plt.show()
#     plt.plot(epochpmom, vp_r)
#     plt.plot(epoch, Vsw_r_interp)
#     plt.show()
#
#     ets = spice.datetime2et(epoch)
#     psp_pos_carr, _ = spice.spkpos('SPP', ets, 'SPP_HG', 'NONE', 'SUN')
#     psp_pos_carr = psp_pos_carr.T / Rs
#     psp_pos_carr_rtp = np.array([xyz2rtp_in_Carrington(psp_pos_carr[:, i]) for i in range(len(psp_pos_carr.T))])
#     psp_pos_carr_rtp = psp_pos_carr_rtp.T
#
#     r_footpoint_on_SourceSurface_rs = np.arange(len(epoch)) * np.nan
#     lon_footpoint_on_SourceSurface_deg = np.arange(len(epoch)) * np.nan
#     lat_footpoint_on_SourceSurface_deg = np.arange(len(epoch)) * np.nan
#     MFL_photosphere_lon_deg = np.arange(len(epoch)) * np.nan
#     MFL_photosphere_lat_deg = np.arange(len(epoch)) * np.nan
#     field_lines = []
#
#     from two_step_ballistic_backmapping_method import two_step_backmapping
#     for i, datetime_trace in enumerate(epoch):
#         r_beg_au = psp_pos_carr_rtp[0, i] * Rs / AU
#         lat_beg_deg = np.rad2deg(psp_pos_carr_rtp[2, i])
#         lon_beg_deg = np.rad2deg(psp_pos_carr_rtp[1, i])
#         Vsw_r = Vsw_r_interp[i]
#         print('Vsw_r', Vsw_r)
#         r_footpoint_on_SourceSurface_rs[i], lon_footpoint_on_SourceSurface_deg[i], lat_footpoint_on_SourceSurface_deg[
#             i], \
#         MFL_photosphere_lon_deg[i], MFL_photosphere_lat_deg[i], field_line \
#             = two_step_backmapping(datetime_trace, r_beg_au, lat_beg_deg, lon_beg_deg, Vsw_r, r_source_surface_rs,
#                                    dir_data_PFSS)
#         field_lines.append(field_line)
#
#     plt.figure()
#     plt.scatter(MFL_photosphere_lon_deg, MFL_photosphere_lat_deg, c=np.arange(len(epoch)))
#     plt.colorbar()
#     plt.show()
#     return r_footpoint_on_SourceSurface_rs, lon_footpoint_on_SourceSurface_deg, lat_footpoint_on_SourceSurface_deg, \
#            MFL_photosphere_lon_deg, MFL_photosphere_lat_deg, field_lines

# ------------------调用上面这个皮并且存数据------------------
#     datetime_beg = datetime(2022,3,11,6,15,0)
#     datetime_end = datetime(2022,3,11,12,0,0)
#     timestep = timedelta(minutes=15)
#
#     timestr_beg = datetime_beg.strftime('%Y%m%dT%H%M%S')
#     timestr_end = datetime_end.strftime('%Y%m%dT%H%M%S')
#
#     steps = (datetime_end - datetime_beg) // timestep + 1
#     epoch = np.array([x * timestep + datetime_beg for x in range(steps)])
#     r_footpoint_on_SourceSurface_rs, lon_footpoint_on_SourceSurface_deg, lat_footpoint_on_SourceSurface_deg, MFL_photosphere_lon_deg, MFL_photosphere_lat_deg, field_lines = psp_backmap(
#         epoch, r_source_surface_rs=2.5)
#
#
#     df = pd.DataFrame()
#     df['Epoch'] = epoch
#     df['r_footpoint_on_SourceSurface_rs'] = r_footpoint_on_SourceSurface_rs
#     df['lon_footpoint_on_SourceSurface_deg'] = lon_footpoint_on_SourceSurface_deg
#     df['lat_footpoint_on_SourceSurface_deg'] = lat_footpoint_on_SourceSurface_deg
#     df['MFL_photosphere_lon_deg'] = MFL_photosphere_lon_deg
#     df['MFL_photosphere_lat_deg'] = MFL_photosphere_lat_deg
#
#     fl_coords = []
#     fl_expansions = []
#     for fl in field_lines:
#         try:
#             fl_coords.append(fl.coords)
#             fl_expansions.append(fl.expansion_factor)
#         except:
#             fl_coords.append(np.nan)
#             fl_expansions.append(np.nan)
#     df['Expansion_Factor'] = fl_expansions
#     df.to_csv('export/plot_body_positions/pfss_trace/trace_fps_('+timestr_beg+'-'+timestr_end+'-'+str(timestep//timedelta(minutes=1))+'min).csv')
#     np.save('export/plot_body_positions/pfss_trace/trace_fls_('+timestr_beg+'-'+timestr_end+'-'+str(timestep//timedelta(minutes=1))+'min).npy', fl_coords)

# Define Epoch (SAME IN BACKMAP)
timestep = timedelta(minutes=30)
steps = (datetime_end - datetime_beg) // timestep + 1
dttimes = np.array([x * timestep + datetime_beg for x in range(steps)])

r_psp_carr_br, psp_p, psp_t = get_rlonlat_psp_carr(dttimes)  # !!!!!这个get_rlonlat是用的SPICE数据，可能要改成其他方式
project_size = [1440, 720]  # !!!!这里把PSP经纬度放到STEREO图上的方式很原始，是按照STEREO图片的像素数手动缩放的，可能要改成用sunpy的method之类
ss_lon_ind = df_trace['lon_footpoint_on_SourceSurface_deg'] * project_size[0] / 360
photo_lon_ind = df_trace['MFL_photosphere_lon_deg'] * project_size[0] / 360
psp_lon_ind = np.rad2deg(psp_p) * project_size[0] / 360
ss_lat_ind = (90 + df_trace['lat_footpoint_on_SourceSurface_deg']) * project_size[1] / 180
photo_lat_ind = (90 + df_trace['MFL_photosphere_lat_deg']) * project_size[1] / 180
psp_lat_ind = np.rad2deg(psp_t) * project_size[1] / 180
expansion_factor = df_trace['Expansion_Factor']

# READ PSP INSITU DATA
# 读PSP的磁场数据，插值到回溯的epoch上，用来给磁力线上色
mag_RTN = load_RTN_1min_data(datetime_beg.strftime('%Y%m%d'), datetime_end.strftime('%Y%m%d'))
epochmag = mag_RTN['epoch_mag_RTN_1min']
timebinmag = (epochmag > datetime_beg) & (epochmag < datetime_end)
epochmag = epochmag[timebinmag]

Br = mag_RTN['psp_fld_l2_mag_RTN_1min'][timebinmag, 0]
Bt = mag_RTN['psp_fld_l2_mag_RTN_1min'][timebinmag, 1]
Bn = mag_RTN['psp_fld_l2_mag_RTN_1min'][timebinmag, 2]
Babs = np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2)

f = interp1d(np.array((epochmag - epochmag[0]) / timedelta(days=1), dtype='float64'), Br, kind='previous',
             fill_value='extrapolate')
Br_interp = f(np.array((dttimes - epochmag[0]) / timedelta(days=1),
                       dtype='float64'))  # use interpolation function returned by `interp1d`

# READ GONG MAP & STEREO DATA
# 这个也很原始，是随机选了某个时刻的GONG MAP把所有时刻的回溯结果都画在上面
gong_long0 = 111
gong_timestr = '210428t1204'
gong_fname = '/Users/ephe/PFSS_Data/mrzqs' + gong_timestr + 'c2243_' + str(gong_long0) + '.fits.gz'
gong_map = sunpy.map.Map(gong_fname)

data, header = sunpy.io.fits.read('/Users/ephe/STEREO_Data/EUVI/195/20210428_134030_n4eua.fts')[0]
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

# 用GONG做一次PFSS外推，为了得到光球开放场区域以及中性线
nrho = 35
rss = 2.5
pfss_in = pfsspy.Input(gong_map, nrho, rss)
pfss_out = pfsspy.pfss(pfss_in)

fig = plt.figure()
ss_br = pfss_out.source_surface_br
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=1)
# Number of steps in cos(latitude)
nsteps = 90
lon_1d = np.linspace(0, 2 * np.pi, nsteps * 2 + 1)
lat_1d = np.arcsin(np.linspace(-1, 1, nsteps + 1))
lon, lat = np.meshgrid(lon_1d, lat_1d, indexing='ij')
lon, lat = lon * u.rad, lat * u.rad

seeds = SkyCoord(lon.ravel(), lat.ravel(), const.R_sun, frame=pfss_out.coordinate_frame)
tracer = pfsspy.tracing.FortranTracer(max_steps=2000)
field_lines_openclose = tracer.trace(seeds, pfss_out)
cmap = mcolor.ListedColormap(['tab:blue', 'white', 'tab:red'])
norm = mcolor.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], ncolors=3)
pols = field_lines_openclose.polarities.reshape(2 * nsteps + 1, nsteps + 1).T
ax1.contourf(np.rad2deg(lon_1d), np.rad2deg(lat_1d), pols, norm=norm, cmap=cmap, alpha=0.9)

# 把trace出来的磁力线投影上去
for field_line in field_lines_psp:
    try:
        field_line.representation_type = 'spherical'
        ax1.plot(field_line.lon, field_line.lat, 'gray', linewidth=0.5)
    except:
        print('Nan')

plt.plot(ss_lon_ind / 4, ss_lat_ind / 4 - 90, c='green', label='Footpoints @Source Surface')
plt.scatter(photo_lon_ind / 4, photo_lat_ind / 4 - 90, s=5, c='green', marker='o',
            label='FootPoints @Photosphere')  # c=np.arange(len(df_trace['Epoch']))
plt.scatter(ss_lon_ind[[0, 44, 74]] / 4, ss_lat_ind[[0, 44, 74]] / 4 - 90, c='white', s=10)

ax1.scatter(pfss_out.source_surface_pils[0].lon.to_value(u.deg), pfss_out.source_surface_pils[0].lat.to_value(u.deg),
            s=0.5, c='black', label='PIL @Source Surface')

ax1.set_ylabel('latitude')
ax1.set_xlabel('longitude')
ax1.set_aspect(1)
plt.legend()
ax1.set_title('PSP Footpoints on open field map')

# 这里是为了把EUV图像切割到我们想要的区域
top_right_bt = SkyCoord(180 * u.deg, 90 * u.deg, frame=outmap.coordinate_frame)
bottom_left_bt = SkyCoord(0 * u.deg, -90 * u.deg, frame=outmap.coordinate_frame)
outmap_sub_bt = outmap.submap(bottom_left_bt, top_right=top_right_bt)

ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1, rowspan=1, projection=outmap_sub_bt)
outmap_sub_bt.plot_settings['norm'] = colors.LogNorm(600, 2400)  # 挑EUV图亮度
outmap_sub_bt.plot()
plt.colorbar()
plt.scatter(photo_lon_ind, photo_lat_ind, s=5, c='white', marker='x',
            label='FP @Photosphere')  #
# 修改EUV图的范围
xlims_world = [60, 90] * u.deg
ylims_world = [-50, -20] * u.deg
world_coords = SkyCoord(lon=xlims_world, lat=ylims_world, frame=outmap_sub_bt.coordinate_frame)
pixel_coords = outmap_sub_bt.world_to_pixel(world_coords)
xlims_pixel = pixel_coords.x.value
ylims_pixel = pixel_coords.y.value
ax2.set_xlim(xlims_pixel)
ax2.set_ylim(ylims_pixel)
plt.legend()

ax2 = plt.subplot2grid((2, 2), (1, 1), colspan=1, rowspan=1, projection=outmap_sub_bt)
outmap_sub_bt.plot_settings['norm'] = colors.LogNorm(600, 2400)

outmap_sub_bt.plot()
plt.colorbar()
plt.scatter(photo_lon_ind, photo_lat_ind, s=5, c='white', marker='x',
            label='FP @Photosphere')  # c=np.arange(len(df_trace['Epoch']))
xlims_world = [110, 140] * u.deg
ylims_world = [45, 75] * u.deg
world_coords = SkyCoord(lon=xlims_world, lat=ylims_world, frame=outmap_sub_bt.coordinate_frame)
pixel_coords = outmap_sub_bt.world_to_pixel(world_coords)
xlims_pixel = pixel_coords.x.value
ylims_pixel = pixel_coords.y.value
ax2.set_xlim(xlims_pixel)
ax2.set_ylim(ylims_pixel)
plt.legend()
plt.show()

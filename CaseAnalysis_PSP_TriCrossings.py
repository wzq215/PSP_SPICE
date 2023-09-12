from datetime import datetime, timedelta
import astropy.constants as const
import numpy as np
import pandas as pd
import pfsspy
import plotly.graph_objects as go
import plotly.offline as py
import pyvista
import astropy.units as u
import spiceypy as spice
import sunpy
from matplotlib import pyplot as plt
from astropy.coordinates import SkyCoord
from load_psp_data import load_spi_data
from plot_body_positions import xyz2rtp_in_Carrington, rtp2xyz_in_Carrington
from pfsspy import tracing
import matplotlib.colors as mcolor
import furnsh_kernels
# from Plot_HCS import dphidr, parker_spiral
from PIL import Image, ImageOps

from scipy import interpolate


def dphidr(r, phi_at_r, Vsw_at_r):
    period_sunrot = 27. * (24. * 60. * 60)  # unit: s
    omega_sunrot = 2 * np.pi / period_sunrot
    result = omega_sunrot / Vsw_at_r  # unit: rad/km
    return result


def parker_spiral(r_vect_au, lat_beg_deg, lon_beg_deg, Vsw_r_vect_kmps):
    from_au_to_km = 1.49597871e8  # unit: km
    from_deg_to_rad = np.pi / 180.
    from_rs_to_km = 6.96e5
    from_au_to_rs = from_au_to_km / from_rs_to_km
    r_vect_km = r_vect_au * from_au_to_km
    num_steps = len(r_vect_km) - 1
    phi_r_vect = np.zeros(num_steps + 1)
    for i_step in range(0, num_steps):
        if i_step == 0:
            phi_at_r_current = lon_beg_deg * from_deg_to_rad  # unit: rad
            phi_r_vect[0] = phi_at_r_current
        else:
            phi_at_r_current = phi_at_r_next
        r_current = r_vect_km[i_step]
        r_next = r_vect_km[i_step + 1]
        r_mid = (r_current + r_next) / 2
        dr = r_current - r_next
        Vsw_at_r_current = Vsw_r_vect_kmps[i_step - 1]
        Vsw_at_r_next = Vsw_r_vect_kmps[i_step]
        Vsw_at_r_mid = (Vsw_at_r_current + Vsw_at_r_next) / 2
        k1 = dr * dphidr(r_current, phi_at_r_current, Vsw_at_r_current)
        k2 = dr * dphidr(r_current + 0.5 * dr, phi_at_r_current + 0.5 * k1, Vsw_at_r_mid)
        k3 = dr * dphidr(r_current + 0.5 * dr, phi_at_r_current + 0.5 * k2, Vsw_at_r_mid)
        k4 = dr * dphidr(r_current + 1.0 * dr, phi_at_r_current + 1.0 * k3, Vsw_at_r_next)
        phi_at_r_next = phi_at_r_current + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        phi_r_vect[i_step + 1] = phi_at_r_next
    lon_r_vect_deg = phi_r_vect / from_deg_to_rad  # from [rad] to [degree]
    lat_r_vect_deg = np.zeros(num_steps + 1) + lat_beg_deg  # unit: [degree]
    # r_footpoint_on_SourceSurface_rs = r_vect_au[-1] * from_au_to_rs
    # lon_footpoint_on_SourceSurface_deg = lon_r_vect_deg[-1]
    # lat_footpoint_on_SourceSurface_deg = lat_r_vect_deg[-1]
    return lon_r_vect_deg, lat_r_vect_deg


Rs = 696300  # km
AU = 1.49597871e8
r_1au = AU / Rs
r_ss = 2.5
# ========Data Preparation=======

# set time range for PSP orbit
start_time = '2022-02-16'
stop_time = '2022-03-12'
# start_time = '2022-02-12'
# stop_time = '2022-03-12'
start_dt = datetime.strptime(start_time, '%Y-%m-%d')
stop_dt = datetime.strptime(stop_time, '%Y-%m-%d')
utc = [start_dt.strftime('%b %d, %Y'), stop_dt.strftime('%b %d, %Y')]
etOne = spice.str2et(utc[0])
etTwo = spice.str2et(utc[1])

step = 100
times = [x * (etTwo - etOne) / step + etOne for x in range(step)]

psp_pos, _ = spice.spkpos('SPP', times, 'IAU_SUN', 'NONE', 'SUN')  # km
psp_pos = psp_pos.T / Rs
psp_pos = {'x': psp_pos[0], 'y': psp_pos[1], 'z': psp_pos[2]}
psp_pos = pd.DataFrame(data=psp_pos)
datetimes = spice.et2datetime(times)
datetimes = [dt.strftime('%Y%m%d%H%M%S') for dt in datetimes]
rs = np.sqrt(psp_pos['x'] ** 2 + psp_pos['y'] ** 2 + psp_pos['z'] ** 2)

marker_dts = [datetime(2022, 2, 17, 13, 48, 0), datetime(2022, 2, 17, 17, 7, 0), datetime(2022, 2, 18, 1, 40, 0),
              datetime(2022, 2, 25, 12, 26, 0), datetime(2022, 2, 25, 12, 33, 0), datetime(2022, 2, 25, 12, 37, 0),
              datetime(2022, 3, 10, 11, 50, 0), datetime(2022, 3, 10, 22, 30, 0), datetime(2022, 3, 11, 23, 59, 0)]
marker_ets = [spice.datetime2et(dt) for dt in marker_dts]
marker_txts = [dt.strftime('%m%d-%H:%M') for dt in marker_dts]
epoch = np.array(marker_dts)
pmom_SPI = load_spi_data(epoch[0].strftime('%Y%m%d'), epoch[-1].strftime('%Y%m%d'))
epochpmom = pmom_SPI['Epoch']
timebinpmom = (epochpmom > epoch[0]) & (epochpmom < epoch[-1])
epochpmom = epochpmom[timebinpmom]
densp = pmom_SPI['DENS'][timebinpmom]
vp_r = pmom_SPI['VEL_RTN_SUN'][timebinpmom, 0]
vp_t = pmom_SPI['VEL_RTN_SUN'][timebinpmom, 1]
vp_n = pmom_SPI['VEL_RTN_SUN'][timebinpmom, 2]

f = interpolate.interp1d(np.array((epochpmom - epochpmom[0]) / timedelta(days=1), dtype='float64'), vp_r,
                         kind='nearest', fill_value='extrapolate')

Vsw_r_interp = f(np.array((epoch - epochpmom[0]) / timedelta(days=1),
                          dtype='float64'))
psp_obs_cross, _ = spice.spkpos('SPP', marker_ets, 'IAU_SUN', 'NONE', 'SUN')  # km
psp_obs_cross = np.array(psp_obs_cross.T / Rs)

ftps_ss_lon = []
ftps_ss_lat = []

for i in range(len(marker_ets)):
    print('Current Time: ', marker_txts[i])
    vr_tmp = Vsw_r_interp[i]
    print('Vr [km/s]: ', vr_tmp)
    print('PSP Position xyz [Rs]: ', psp_obs_cross[:, i])
    r_psp_obs_cross, lon_psp_obs_cross, lat_psp_obs_cross = xyz2rtp_in_Carrington(psp_obs_cross[:, i], for_psi=False)
    print('PSP Position rlonlat [Rs,rad,rad]: ', r_psp_obs_cross, lon_psp_obs_cross, lat_psp_obs_cross)
    r_vect = np.linspace(r_psp_obs_cross, r_ss, num=100)

    vr_vect = r_vect * 0 + vr_tmp
    lon_vect, lat_vect = parker_spiral(r_vect * Rs / AU, np.rad2deg(lat_psp_obs_cross),
                                       np.rad2deg(lon_psp_obs_cross), vr_vect)
    print('Footpoints at Source Surface lonlat [deg]: ', lon_vect[-1], lat_vect[-1])
    ftps_ss_lon.append(lon_vect[-1])
    ftps_ss_lat.append(lat_vect[-1])
    x_vect, y_vect, z_vect = rtp2xyz_in_Carrington([r_vect, lon_vect, lat_vect], for_psi=False)
    print('----------')

ftps_ss_lon = np.array(ftps_ss_lon)
ftps_ss_lat = np.array(ftps_ss_lat)
plt.figure()
plt.scatter(ftps_ss_lon, ftps_ss_lat, c=['g', 'g', 'g', 'r', 'r', 'r', 'b', 'b', 'b'],
            s=[10, 20, 10, 10, 20, 10, 10, 20, 10])
plt.show()

gong_path = '/Users/ephe/PFSS_Data/'
gong_fname = 'mrzqs220225t0014c2254_078.fits'  # 文件名改成自己的
gong_map = sunpy.map.Map(gong_path + gong_fname)
# Remove the mean，这里为了使curl B = 0
gong_map = sunpy.map.Map(gong_map.data - np.mean(gong_map.data), gong_map.meta)
# 5. 设置网格数量和source surface高度并计算
nrho = 30
rss = 2.5  # 单位：太阳半径
input = pfsspy.Input(gong_map, nrho, rss)
output = pfsspy.pfss(input)
# 6. 画输入的GONG磁图
m = input.map
fig = plt.figure()
ax = plt.subplot(projection=m)
m.plot()
plt.colorbar()
ax.set_title('Input field')
ax.set_xlim(0, 360)
ax.set_ylim(0, 180)

# 7. 画输出的source surface磁场分布
ftpts_ss = SkyCoord(np.deg2rad(ftps_ss_lon.ravel()) * u.rad, np.deg2rad(ftps_ss_lat.ravel()) * u.rad,
                    frame=m.coordinate_frame)
ss_br = output.source_surface_br
# Create the figure and axes
fig = plt.figure()
ax = plt.subplot(projection=ss_br)

# Plot the source surface map
ss_br.plot()
# Plot the polarity inversion line
ax.plot_coord(output.source_surface_pils[0])
ax.plot_coord(ftpts_ss[:3], color='blue', marker='.', linewidth=1)
ax.plot_coord(ftpts_ss[3:6], color='red', marker='.', linewidth=1)
ax.plot_coord(ftpts_ss[6:], color='green', marker='.', linewidth=1)

print(output.source_surface_pils)
# Plot formatting
plt.colorbar()
ax.set_title('Source surface magnetic field')
# ax.set_xlim(10, 45)
# ax.set_ylim(75, 105)
ax.set_xlim(0, 360)
ax.set_ylim(0, 180)
plt.show()

r = const.R_sun
# Number of steps in cos(latitude)
nsteps = 45
lon_1d = np.linspace(0, 2 * np.pi, nsteps * 2 + 1)
lat_1d = np.arcsin(np.linspace(-1, 1, nsteps + 1))
lon, lat = np.meshgrid(lon_1d, lat_1d, indexing='ij')
lon, lat = lon * u.rad, lat * u.rad
seeds = SkyCoord(lon.ravel(), lat.ravel(), r, frame=output.coordinate_frame)

print('Tracing field lines...')
tracer = tracing.FortranTracer(max_steps=2000)
field_lines = tracer.trace(seeds, output)
print('Finished tracing field lines')

fig = plt.figure()
m = input.map
ax = fig.add_subplot(2, 1, 1, projection=m)
m.plot()
ax.set_title('Input GONG magnetogram')

ax = fig.add_subplot(2, 1, 2)
cmap = mcolor.ListedColormap(['tab:red', 'black', 'tab:blue'])
norm = mcolor.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], ncolors=3)
pols = field_lines.polarities.reshape(2 * nsteps + 1, nsteps + 1).T
ax.contourf(np.rad2deg(lon_1d), np.sin(lat_1d), pols, norm=norm, cmap=cmap)
ax.set_ylabel('sin(latitude)')

ax.set_title('Open (blue/red) and closed (black) field')
ax.set_aspect(0.5 * 360 / 2)

plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for field_line in field_lines:
    color = {0: 'black', -1: 'tab:blue', 1: 'tab:red'}.get(field_line.polarity)
    coords = field_line.coords
    coords.representation_type = 'cartesian'
    ax.plot(coords.x / const.R_sun,
            coords.y / const.R_sun,
            coords.z / const.R_sun,
            color=color, linewidth=1)

ax.set_title('PFSS solution')
plt.show()

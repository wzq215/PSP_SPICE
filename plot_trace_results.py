from datetime import datetime, timedelta

import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.colors as mcolor
import numpy as np
import pandas as pd
import pfsspy
import pyvista as pv
from sunkit_pyvista import SunpyPlotter
import spiceypy as spice
import sunpy.io.fits
import sunpy.map
from astropy.coordinates import SkyCoord
from pfsspy import tracing
from scipy.interpolate import interp1d

from load_psp_data import load_RTN_1min_data
from plot_body_positions import get_rlonlat_psp_carr
import furnsh_kernels

# gong_fname = '/Users/ephe/PFSS_data/mrzqs210429t0004c2243_105.fits.gz'
# gong_map = sunpy.map.Map(gong_fname)
# quit()
# READ TRACING BACK DATA
datetime_beg = datetime(2021, 4, 28, 15, 30, 0)
datetime_end = datetime(2021, 4, 30, 4, 30, 0)
df_trace = pd.read_csv('E8trace0429_ss27.csv')
field_lines_psp = np.load('save_field_lines_0429_ss27.npy', allow_pickle=True)

timestep = timedelta(minutes=30)
steps = (datetime_end - datetime_beg) // timestep + 1
dttimes = np.array([x * timestep + datetime_beg for x in range(steps)])
times = spice.datetime2et(dttimes)

r_psp_carr_br, psp_p, psp_t = get_rlonlat_psp_carr(dttimes)
project_size = [1440, 720]
ss_lon_ind = df_trace['lon_footpoint_on_SourceSurface_deg'] * project_size[0] / 360
photo_lon_ind = df_trace['MFL_photosphere_lon_deg'] * project_size[0] / 360
psp_lon_ind = np.rad2deg(psp_p) * project_size[0] / 360
ss_lat_ind = (90 + df_trace['lat_footpoint_on_SourceSurface_deg']) * project_size[1] / 180
photo_lat_ind = (90 + df_trace['MFL_photosphere_lat_deg']) * project_size[1] / 180
psp_lat_ind = np.rad2deg(psp_t) * project_size[1] / 180
expansion_factor = df_trace['Expansion_Factor']

# READ PSP INSITU DATA
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
gong_long0 = 111
gong_timestr = '210428t1204'
Magmap, Magmapheader = \
    sunpy.io.fits.read('/Users/ephe/PFSS_Data/mrzqs' + gong_timestr + 'c2243_' + str(gong_long0) + '.fits')[0]

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

gong_fname = '/Users/ephe/PFSS_Data/mrzqs' + gong_timestr + 'c2243_' + str(gong_long0) + '.fits.gz'
gong_map = sunpy.map.Map(gong_fname)

nrho = 35
rss = 2.5
pfss_in = pfsspy.Input(gong_map, nrho, rss)
pfss_out = pfsspy.pfss(pfss_in)

if False:
    sg_vect = pfss_out.grid.sg
    pg_vect = pfss_out.grid.pg
    rg_vect = pfss_out.grid.rg
    sc_vect = pfss_out.grid.sc
    pc_vect = pfss_out.grid.pc
    rc_vect = pfss_out.grid.rc
    b_arr = pfss_out.bg
    bc_r = pfss_out.bc[0]
    # print(pc_vect)
    # print(np.arccos(sc))
    [s_arr, p_arr, r_arr] = np.meshgrid(sc_vect, pc_vect, rg_vect)
    x_arr = np.exp(r_arr) * np.cos(p_arr) * np.sqrt(1 - s_arr ** 2)
    y_arr = np.exp(r_arr) * np.sin(p_arr) * np.sqrt(1 - s_arr ** 2)
    z_arr = np.exp(r_arr) * s_arr
    # lon_vect = pc_vect * u.rad
    # lat_vect = np.arccos(sc_vect) * u.rad
    # r_vect = np.exp(rg_vect)
    # [lon_grid,lat_grid] = np.meshgrid(lon_vect,lat_vect)
    # coord_grid = SkyCoord(x_arr.ravel()*const.R_sun, y_arr.ravel()*const.R_sun, z_arr.ravel()*const.R_sun,frame=pfss_out.coordinate_frame,representation_type='cartesian')
    # x_arr = coord_grid.x.to_value(const.R_sun)
    # y_arr = coord_grid.y.to_value(const.R_sun)
    # z_arr = coord_grid.z.to_value(const.R_sun)
    # x_arr = np.array(x_arr).reshape(360,180,36)
    # y_arr = np.array(y_arr).reshape(360,180,36)
    # z_arr = np.array(z_arr).reshape(360,180,36)
    # bc = pfss_out.get_bvec(coord_grid)
    # bc_r_interp = np.array(bc[:,0]).reshape(360,180,36)
    # [x_arr,y_arr,z_arr] = np.meshgrid(x_vect,y_vect,z_vect)
    mesh_g = pv.StructuredGrid(x_arr, y_arr, z_arr)
    mesh_g.point_data['values'] = bc_r.ravel(order='F')
    isos_b0 = mesh_g.contour(isosurfaces=1, rng=[0, 0])

    tracer = tracing.FortranTracer()
    # r = 1.2 * const.R_sun
    # lat_2 = np.deg2rad(np.linspace(55, 60, 5, endpoint=False))
    # lon_2 = np.deg2rad(np.linspace(115, 145, 30, endpoint=False))
    # lat_2, lon_2 = np.meshgrid(lat_2, lon_2, indexing='ij')
    # lat_2, lon_2 = lat_2.ravel() * u.rad, lon_2.ravel() * u.rad
    # lat_1 = np.deg2rad(np.linspace(-40, -30, 10, endpoint=False))
    # lon_1 = np.deg2rad(np.linspace(60, 90, 30, endpoint=False))
    # lat_1, lon_1 = np.meshgrid(lat_1, lon_1, indexing='ij')
    # lat_1, lon_1 = lat_1.ravel() * u.rad, lon_1.ravel() * u.rad
    # lat = np.append(lat_1,lat_2)
    # lon = np.append(lon_1,lon_2)

    lat = np.deg2rad(np.linspace(-90, 90, 9, endpoint=False))
    lon = np.deg2rad(np.linspace(0, 360, 18, endpoint=False))
    lat, lon = np.meshgrid(lat, lon, indexing='ij')
    lat, lon = lat.ravel() * u.rad, lon.ravel() * u.rad
    # df_trace['lon_footpoint_on_SourceSurface_deg']
    # seeds = SkyCoord(df_trace['lon_footpoint_on_SourceSurface_deg'] * u.deg,
    #                  df_trace['lat_footpoint_on_SourceSurface_deg'] * u.deg, rss * const.R_sun,
    #                  frame=pfss_out.coordinate_frame)
    seeds = SkyCoord(lon, lat, rss * const.R_sun,
                     frame=pfss_out.coordinate_frame)
    # print(seeds)
    field_lines = tracer.trace(seeds, pfss_out)

    plotter = SunpyPlotter()

    # plotter.plot_map(gong_map,assume_spherical_screen=False,cmap='Accent')
    # plotter.plot_map(sta_map,clip_interval=(0.2, 99.) * u.percent,assume_spherical_screen=False)
    plotter.plot_solar_axis()


    def my_fline_color_func(field_line):
        norm = colors.LogNorm(vmin=1, vmax=1000)
        cmap = plt.get_cmap("viridis")
        return cmap(norm(np.abs(field_line.expansion_factor)))


    # bottom_left = SkyCoord(110 * u.deg, 50 * u.deg, frame=pfss_out.coordinate_frame, )
    # plotter.plot_quadrangle(bottom_left=bottom_left, width=30 * u.deg, height=30 * u.deg, color="blue")
    # bottom_left = SkyCoord(60 * u.deg, -60 * u.deg, frame=pfss_out.coordinate_frame, )
    # plotter.plot_quadrangle(bottom_left=bottom_left, width=30 * u.deg, height=30 * u.deg, color="blue")
    plotter.plot_field_lines(field_lines, color_func=my_fline_color_func)
    plotter.plotter.add_mesh(pv.Sphere(radius=1))
    plotter.plotter.add_mesh(isos_b0, opacity=0.6)
    # plotter.plot_coordinates()
    plotter.show()

    quit()

# vertices2 = isos_b0.points
# triangles2 = isos_b0.faces.reshape(-1, 4)
# plot = go.Figure()
# plot.add_trace(go.Mesh3d(x=vertices2[:, 0], y=vertices2[:, 1], z=vertices2[:, 2],
#                          opacity=0.9, color='purple',
#                          # colorscale='jet',
#                          # colorscale='Viridis',
#                          # cmax=450, cmin=150,
#                          i=triangles2[:, 1], j=triangles2[:, 2], k=triangles2[:, 3],
#                          # intensity=intensity,
#                          showscale=False,
#                          ))
# sunim = Image.open('data/euvi_aia304_2012_carrington_print.jpeg')
# sunimgray = ImageOps.grayscale(sunim)
# sundata = np.array(Magmap[:,np.arange(360)-gong_long0])
# theta = np.linspace(0, 2 * np.pi, sundata.shape[1])
# phi = np.linspace(0, np.pi, sundata.shape[0])
#
# tt, pp = np.meshgrid(theta, phi)
# r = 1.2
# x0 = r * np.cos(tt) * np.sin(pp)
# y0 = r * np.sin(tt) * np.sin(pp)
# z0 = -r * np.cos(pp)
#
# plot.add_trace(go.Surface(x=x0, y=y0, z=z0, surfacecolor=sundata, colorscale='RdBu', opacity=1, showscale=False,
#                           cmax=8, cmin=-8,))
#
# for field_line in field_lines:
#     color = {0: 'black', -1: 'tab:blue', 1: 'tab:red'}.get(field_line.polarity)
#     coords = field_line.coords
#     coords.representation_type = 'cartesian'
#     # print(coords)
#     # ax.plot(coords.lon,
#     #         coords.lat,
#     #         coords.radius / const.R_sun,
#     #         color=color, linewidth=1)
#     if field_line.polarity ==0:
#         plot.add_trace(go.Scatter3d(x=coords.x/const.R_sun, y=coords.y/const.R_sun, z=coords.z/const.R_sun,
#                                 mode='lines', line=dict(color='black', width=3),))
#     elif field_line.polarity == -1:
#         plot.add_trace(go.Scatter3d(x=coords.x/const.R_sun, y=coords.y/const.R_sun, z=coords.z/const.R_sun,
#                                     mode='lines', line=dict(color='blue', width=3),))
#     elif field_line.polarity == 1:
#         plot.add_trace(go.Scatter3d(x=coords.x/const.R_sun, y=coords.y/const.R_sun, z=coords.z/const.R_sun,
#                                     mode='lines', line=dict(color='red', width=3),))
#
# plot.update_layout(showlegend=False,)
# py.plot(plot, filename='HCS_pfss.html')
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# tracer = tracing.FortranTracer()
# r = 1.2 * const.R_sun
# lat = np.deg2rad(np.linspace(50, 65, 3, endpoint=False))
# lon = np.deg2rad(np.linspace(100, 150, 10, endpoint=False))
# lat = np.deg2rad(np.linspace(-60, -30, 6, endpoint=False))
# lon = np.deg2rad(np.linspace(60, 90, 6, endpoint=False))
# lat, lon = np.meshgrid(lat, lon, indexing='ij')
# lat, lon = lat.ravel() * u.rad, lon.ravel() * u.rad
#
# seeds = SkyCoord(lon, lat, r, frame=pfss_out.coordinate_frame)
#
# field_lines = tracer.trace(seeds, pfss_out)
#
# for field_line in field_lines:
#     color = {0: 'black', -1: 'tab:blue', 1: 'tab:red'}.get(field_line.polarity)
#     coords = field_line.coords
#     coords.representation_type = 'spherical'
#     # print(coords)
#     ax.plot(coords.lon,
#             coords.lat,
#             coords.radius / const.R_sun,
#             color=color, linewidth=1)
#
# ax.set_title('PFSS solution')
# plt.show()

#
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
tracer = tracing.FortranTracer(max_steps=2000)
field_lines_openclose = tracer.trace(seeds, pfss_out)
cmap = mcolor.ListedColormap(['tab:blue', 'white', 'tab:red'])
norm = mcolor.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], ncolors=3)
pols = field_lines_openclose.polarities.reshape(2 * nsteps + 1, nsteps + 1).T
ax1.contourf(np.rad2deg(lon_1d), np.rad2deg(lat_1d), pols, norm=norm, cmap=cmap, alpha=0.9)
# ax1.contour(np.rad2deg(lon_1d), np.rad2deg(lat_1d), pfs, norm=norm, cmap=cmap, alpha=0.9)
for field_line in field_lines_psp:
    try:
        field_line.representation_type = 'spherical'
        ax1.plot(field_line.lon, field_line.lat, 'gray', linewidth=0.5)
    except:
        print('Nan')
lon_1d = np.linspace(0, 2 * np.pi, 361)
lat_1d = np.arcsin(np.linspace(-1, 1, 181))
# plt.scatter(ss_lon_ind / 4, ss_lat_ind / 4 - 90, c=np.sign(Br_interp), s=8, cmap='RdBu_r', label='FP @Source Surface')
plt.plot(ss_lon_ind / 4, ss_lat_ind / 4 - 90, c='green', label='Footpoints @Source Surface')
plt.scatter(photo_lon_ind / 4, photo_lat_ind / 4 - 90, s=5, c='green', marker='o',
            label='FootPoints @Photosphere')  # c=np.arange(len(df_trace['Epoch']))
plt.scatter(ss_lon_ind[[0, 44, 74]] / 4, ss_lat_ind[[0, 44, 74]] / 4 - 90, c='white', s=10)
# ss_br.plot()
# print(pfss_out.source_surface_pils)
ax1.scatter(pfss_out.source_surface_pils[0].lon.to_value(u.deg), pfss_out.source_surface_pils[0].lat.to_value(u.deg),
            s=0.5, c='black', label='PIL @Source Surface')
# ax.contour(np.sqrt(pfss_out.bg[:,:,-1,0]**2+pfss_out.bg[:,:,-1,1]**2+pfss_out.bg[:,:,-1,2]**2))
ax1.set_ylabel('latitude')
ax1.set_xlabel('longitude')
ax1.set_aspect(1)
plt.legend()
ax1.set_title('PSP Footpoints on open field map')

top_right_bt = SkyCoord(180 * u.deg, 90 * u.deg, frame=outmap.coordinate_frame)
bottom_left_bt = SkyCoord(0 * u.deg, -90 * u.deg, frame=outmap.coordinate_frame)
outmap_sub_bt = outmap.submap(bottom_left_bt, top_right=top_right_bt)

ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1, rowspan=1, projection=outmap_sub_bt)
outmap_sub_bt.plot_settings['norm'] = colors.LogNorm(600, 2400)
outmap_sub_bt.plot()
plt.colorbar()
plt.scatter(photo_lon_ind, photo_lat_ind, s=5, c='white', marker='x',
            label='FP @Photosphere')  # c=np.arange(len(df_trace['Epoch']))
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
# outmap_sub_bt.plot(clip_interval=(15, 99.9) * u.percent)
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

quit()

fig = plt.figure()
plt.subplot(3, 2, 1)
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
        # print(field_line.lon[0]*360/360)

    except:
        print('Nan')
# print(photo_lon_ind)
plt.scatter(ss_lon_ind / 4, ss_lat_ind / 4, c=np.sign(Br_interp), s=8, cmap='RdBu_r', label='FP @Source Surface')
plt.scatter(photo_lon_ind / 4, photo_lat_ind / 4, s=5, c='white', marker='x',
            label='FP @Photosphere')  # c=np.arange(len(df_trace['Epoch']))
plt.scatter(ss_lon_ind[[0, 44, 74]] / 4, ss_lat_ind[[0, 44, 74]] / 4, c='white', s=10)
plt.xlim([0, 180])
plt.title('$Br [nT]$')
plt.ylabel('Latitude')
plt.colorbar()

# plt.subplot(3,2,2)

ax2 = fig.add_subplot(3, 2, 2, projection=outmap_sub_bt)
outmap_sub_bt.plot()
plt.colorbar()
# for field_line in field_lines:
#     try:
#         field_line.representation_type = 'spherical'
#         plt.plot(field_line.lon* 1440/360,(field_line.lat+90*u.deg) * 720/180,'gray',linewidth=0.5)
#         print(field_line.lon[0]*1440/360)
#
#     except:
#         print('Nan')
# print(photo_lon_ind)
# plt.scatter(ss_lon_ind, ss_lat_ind, c=np.sign(Br_interp), s=8,cmap='RdBu_r',label='FP @Source Surface')
plt.scatter(photo_lon_ind, photo_lat_ind, s=5, c='white', marker='x',
            label='FP @Photosphere')  # c=np.arange(len(df_trace['Epoch']))
ax2.set_xlabel(' ')
ax2.set_ylabel(' ')
ax2.set_title('193 A')

ax = fig.add_subplot(3, 2, 3, projection='3d', proj_type='ortho')
[xx, yy] = np.meshgrid(np.linspace(70, 85, 15), np.linspace(-45, -30, 15))
ax.contourf(xx, yy, Magmap[70 - gong_long0:85 - gong_long0, -45 + 90:-30 + 90], offset=1, zdir='z')
tracer = tracing.FortranTracer()
r = 1.1 * const.R_sun
# lat = np.deg2rad(np.linspace(50, 65, 3, endpoint=False))
# lon = np.deg2rad(np.linspace(100, 150, 10, endpoint=False))
lat = np.deg2rad(np.linspace(-45, -30, 5, endpoint=False))
lon = np.deg2rad(np.linspace(70, 85, 5, endpoint=False))
lat, lon = np.meshgrid(lat, lon, indexing='ij')
lat, lon = lat.ravel() * u.rad, lon.ravel() * u.rad

seeds = SkyCoord(lon, lat, r, frame=pfss_out.coordinate_frame)

field_lines = tracer.trace(seeds, pfss_out)

for field_line in field_lines:
    color = {0: 'black', -1: 'tab:blue', 1: 'tab:red'}.get(field_line.polarity)
    coords = field_line.coords
    coords.representation_type = 'spherical'
    # print(coords)
    ax.plot(coords.lon,
            coords.lat,
            coords.radius / const.R_sun,
            color=color, linewidth=0.8)

ax.set_xlim([70, 85])
ax.set_ylim([-45, -30])
ax.set_zlim([1, 2.5])

ax.set_title('PFSS solution')

top_right_bt = SkyCoord(180 * u.deg, 90 * u.deg, frame=outmap.coordinate_frame)
bottom_left_bt = SkyCoord(0 * u.deg, -90 * u.deg, frame=outmap.coordinate_frame)
outmap_sub_bt = outmap.submap(bottom_left_bt, top_right=top_right_bt)

ax2 = fig.add_subplot(3, 2, 5, projection=outmap_sub_bt)
outmap_sub_bt.plot(clip_interval=(15, 99.9) * u.percent)
plt.colorbar()
plt.scatter(photo_lon_ind, photo_lat_ind, s=5, c='white', marker='x',
            label='FP @Photosphere')  # c=np.arange(len(df_trace['Epoch']))
xlims_world = [60, 90] * u.deg
ylims_world = [-50, -20] * u.deg
world_coords = SkyCoord(lon=xlims_world, lat=ylims_world, frame=outmap_sub_bt.coordinate_frame)
pixel_coords = outmap_sub_bt.world_to_pixel(world_coords)
xlims_pixel = pixel_coords.x.value
ylims_pixel = pixel_coords.y.value
ax2.set_xlim(xlims_pixel)
ax2.set_ylim(ylims_pixel)

ax2 = fig.add_subplot(3, 2, 6, projection=outmap_sub_bt)
outmap_sub_bt.plot(clip_interval=(15, 99.9) * u.percent)
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
plt.show()

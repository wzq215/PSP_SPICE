from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as py
import pyvista
import spiceypy as spice
from ps_read_hdf_3d import ps_read_hdf_3d

from plot_body_positions import xyz2rtp_in_Carrington

Rs = 696300  # km


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


# ========Data Preparation=======

# set time range for PSP orbit
start_time = '2021-04-28'
stop_time = '2021-05-01'
start_dt = datetime.strptime(start_time, '%Y-%m-%d')
stop_dt = datetime.strptime(stop_time, '%Y-%m-%d')
utc = [start_dt.strftime('%b %d, %Y'), stop_dt.strftime('%b %d, %Y')]
etOne = spice.str2et(utc[0])
etTwo = spice.str2et(utc[1])

step = 100
times = [x * (etTwo - etOne) / step + etOne for x in range(step)]

psp_utc_str = '20181104T000000'

# convert psp_pos to carrington coordination system
et = spice.datetime2et(datetime.strptime(psp_utc_str, '%Y%m%dT%H%M%S'))
subpnt_psp = spice.subpnt('INTERCEPT/ELLIPSOID', 'SUN', et, 'IAU_SUN', 'None', 'SPP')
subpnt_lat = np.rad2deg(spice.reclat(subpnt_psp[0])[1:])
if subpnt_lat[0] < 0:
    subpnt_lat[0] = subpnt_lat[0] + 360
subpnt_lat[1] = 90 - subpnt_lat[1]
psp_r = np.linalg.norm(subpnt_psp[0] - subpnt_psp[2], 2) / Rs
print('-------PSP-------')
print(psp_r)
print(subpnt_lat[0])
print(subpnt_lat[1])

# Load Psi Data
data_br = ps_read_hdf_3d(2243, 'helio', 'br002', periodicDim=3)
# data_br = h5py.File('simulation/20210117T131000/corona_h5/br002.h5')
r_br = np.array(data_br['scales1'])  # 201 in Rs, distance from sun
t_br = np.array(data_br['scales2'])  # 150 in rad, latitude
p_br = np.array(data_br['scales3'])  # 256 in rad, Carrington longitude
br = np.array(data_br['datas'])  # 1CU = 2.205G = 2.205e-4T = 2.205e5nT
br = br * 2.205e5  # nT
# print(t_br)
data_rho = ps_read_hdf_3d(2243, 'helio', 'rho002',
                          periodicDim=3)  # h5py.File('simulation/20210117T131000/corona_h5/rho002.h5')
r_rho = np.array(data_rho['scales1'])  # 201 in Rs, distance from sun
t_rho = np.array(data_rho['scales2'])  # 150 in rad, latitude
p_rho = np.array(data_rho['scales3'])  # 256 in rad, Carrington longitude
rho = np.array(data_rho['datas'])
rho = rho * 1e8  # cm^-3
# print(t_rho)
data_vr = ps_read_hdf_3d(2243, 'helio', 'vr002', periodicDim=3)
r_vr = np.array(data_vr['scales1'])  # 201 in Rs, distance from sun
t_vr = np.array(data_vr['scales2'])  # 150 in rad, latitude
p_vr = np.array(data_vr['scales3'])  # 256 in rad, Carrington longitude
vr = np.array(data_vr['datas'])
vr = vr * 481.3711  # km/s

# Find indexs for PSP in PSI dataset
p_index = abs(np.rad2deg(p_br) - subpnt_lat[0]).argmin()
t_index = abs(np.rad2deg(t_br) - subpnt_lat[1]).argmin()
r_index = abs(r_vr - psp_r).argmin()

# get HCS (isosurface of Br=0)
tv, pv, rv = np.meshgrid(t_br, p_br, r_br, indexing='xy')

xv = rv * np.cos(pv) * np.sin(tv)
yv = rv * np.sin(pv) * np.sin(tv)
zv = rv * np.cos(tv)

mesh = pyvista.StructuredGrid(xv, yv, zv)
mesh.point_data['values'] = br.ravel(order='F')  # also the active scalars
isos_br = mesh.contour(isosurfaces=1, rng=[0, 0])

mesh_z0 = pyvista.StructuredGrid(xv, yv, zv)
mesh_z0.point_data['values'] = zv.ravel(order='F')  # also the active scalars
isos_z0 = mesh_z0.contour(isosurfaces=1, rng=[-8, -8])
# isos.plot(opacity=0.7)

# get Isosurface of rho*r^2
tv2, pv2, rv2 = np.meshgrid(t_rho, p_rho, r_rho, indexing='xy')

xv2 = rv2 * np.cos(pv2) * np.sin(tv2)
yv2 = rv2 * np.sin(pv2) * np.sin(tv2)
zv2 = rv2 * np.cos(tv2)
mesh2 = pyvista.StructuredGrid(xv2, yv2, zv2)
rholog = rho * rv2 ** 2
print('min', np.nanmin(rholog))
print('max', np.nanmax(rholog))
mesh2.point_data['values'] = rholog.ravel(order='F')  # also the active scalars
isos_rho = mesh2.contour(isosurfaces=1, rng=[8e5, 8e5])
# isos_rho.plot(opacity=0.9)


# Color HCS by Vr
vertices = isos_z0.points
triangles = isos_z0.faces.reshape(-1, 4)

vr_points = vertices[:, 0] * 0
for i in range(len(vertices)):
    point = np.array(vertices[i])
    r_p, p_p, t_p = xyz2rtp_in_Carrington(point, for_psi=True)

    r_ind = np.argmin(abs(r_p - r_vr))
    p_ind = np.argmin(abs(p_p - p_vr))
    t_ind = np.argmin(abs(t_p - t_vr))

    vr_points[i] = vr[p_ind, t_ind, r_ind]  # * (r_p) ** 2

# intensity = np.log10(np.array(rho_points)).reshape(-1, 1)
intensity = np.array(vr_points).reshape(-1, 1)

# ===========Plot============
plot = go.Figure()
plot.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                         opacity=1,  # colorscale='Magma',
                         colorscale='picnic',
                         cmax=500, cmin=300,
                         #  color='purple',
                         i=triangles[:, 1], j=triangles[:, 2], k=triangles[:, 3],
                         intensity=intensity,
                         colorbar=dict(
                             lenmode='fraction', len=0.6, thicknessmode='fraction', thickness=0.03, x=0.9
                             # orientation='h',y=1
                         )
                         # showscale=False,
                         ))

vertices2 = isos_br.points
triangles2 = isos_br.faces.reshape(-1, 4)
plot.add_trace(go.Mesh3d(x=vertices2[:, 0], y=vertices2[:, 1], z=vertices2[:, 2],
                         opacity=0.7, color='purple',
                         # colorscale='jet',
                         # colorscale='Viridis',
                         # cmax=450, cmin=150,
                         i=triangles2[:, 1], j=triangles2[:, 2], k=triangles2[:, 3],
                         # intensity=intensity,
                         showscale=False,
                         ))

vertices2 = isos_rho.points
triangles2 = isos_rho.faces.reshape(-1, 4)
# plot.add_trace(go.Mesh3d(x=vertices2[:, 0], y=vertices2[:, 1], z=vertices2[:, 2],
#                          opacity=0.3,color='azure',
#                          #colorscale='jet',
#                          # colorscale='Viridis',
#                          # cmax=450, cmin=150,
#                          i=triangles2[:, 1], j=triangles2[:, 2], k=triangles2[:, 3],
#                          # intensity=intensity,
#                          showscale=False,
#                          ))

tv, pv, rv = np.meshgrid(t_vr, p_vr, r_vr, indexing='xy')
rv = rv[p_index - 10:p_index + 10, t_index - 20:t_index + 20, :]
pv = pv[p_index - 10:p_index + 10, t_index - 20:t_index + 20, :]
tv = tv[p_index - 10:p_index + 10, t_index - 20:t_index + 20, :]

vr_plot = vr[p_index - 10:p_index + 10, t_index - 20:t_index + 20, :]
xv = rv * np.cos(pv) * np.sin(tv)
yv = rv * np.sin(pv) * np.sin(tv)
zv = rv * np.cos(tv)

psp_pos, _ = spice.spkpos('SPP', times, 'IAU_SUN', 'NONE', 'SUN')  # km
psp_pos = psp_pos.T / Rs
psp_pos = {'x': psp_pos[0], 'y': psp_pos[1], 'z': psp_pos[2]}
psp_pos = pd.DataFrame(data=psp_pos)
datetimes = spice.et2datetime(times)
datetimes = [dt.strftime('%Y%m%d%H%M%S') for dt in datetimes]
rs = np.sqrt(psp_pos['x'] ** 2 + psp_pos['y'] ** 2 + psp_pos['z'] ** 2)

from PIL import Image, ImageOps

sunim = Image.open('data/euvi_aia304_2012_carrington_print.jpeg')
sunimgray = ImageOps.grayscale(sunim)
sundata = np.array(sunimgray)
theta = np.linspace(0, 2 * np.pi, sundata.shape[1])
phi = np.linspace(0, np.pi, sundata.shape[0])

# sundata = np.squeeze(br[:,:,0]).T
# plt.figure()
# plt.pcolormesh(sundata)
# plt.colorbar()
# plt.show()
# print(sundata.shape)
# theta = p_br
# phi = t_br

tt, pp = np.meshgrid(theta, phi)
r = 3
x0 = r * np.cos(tt) * np.sin(pp)
y0 = r * np.sin(tt) * np.sin(pp)
z0 = r * np.cos(pp)

# plot.add_trace(go.Surface(x=x0, y=y0, z=z0, surfacecolor=sundata, colorscale='rdbu', opacity=1,showscale=True,cmax=3e5,cmin=-3e5))
plot.add_trace(go.Surface(x=x0, y=y0, z=z0, surfacecolor=sundata, colorscale='solar', opacity=1, showscale=False))

plot.add_trace(go.Scatter3d(x=psp_pos['x'], y=psp_pos['y'], z=psp_pos['z'],
                            mode='lines',
                            line=dict(color='black',
                                      width=8),
                            # name='Orbit of PSP (' + start_time + '~' + stop_time + ')',

                            # customdata=np.dstack((datetimes,rs)),
                            # hovertemplate='z1:<br><b>z2:%{z:.3f}</b><br>z3: %{customdata[1]:.3f} ',
                            # hovertemplate=
                            # "<b>%{customdata[0]}</b><br><br>" +
                            # "Position: [%{x},%{y},%{z}]<br>" +
                            # "Radius: %{customdata[1]:.3f}<br>" +
                            # "<extra></extra>",
                            ))
# obs_cross = spice.datetime2et(datetime.strptime('20210426T070000', '%Y%m%dT%H%M%S'))
# simu_cross = spice.datetime2et(datetime.strptime('20210428T220000','%Y%m%dT%H%M%S'))

marker_dts = [datetime(2021, 4, 24, 0, 0, 0), datetime(2021, 4, 24, 12, 0, 0), datetime(2021, 4, 25, 0, 0, 0),
              datetime(2021, 4, 25, 12, 0, 0),
              datetime(2021, 4, 26, 0, 0, 0), datetime(2021, 4, 26, 12, 0, 0), datetime(2021, 4, 27, 0, 0, 0),
              datetime(2021, 4, 27, 12, 0, 0),
              datetime(2021, 4, 28, 0, 0, 0), datetime(2021, 4, 28, 12, 0, 0), datetime(2021, 4, 29, 0, 0, 0),
              datetime(2021, 4, 29, 12, 0, 0),
              datetime(2021, 4, 30, 0, 0, 0), datetime(2021, 4, 30, 12, 0, 0), datetime(2021, 5, 1, 0, 0, 0),
              datetime(2021, 5, 1, 12, 0, 0),
              datetime(2021, 5, 2, 0, 0, 0), datetime(2021, 5, 2, 12, 0, 0), datetime(2021, 5, 3, 0, 0, 0),
              datetime(2021, 5, 3, 12, 0, 0),
              datetime(2021, 5, 4, 0, 0, 0), datetime(2021, 5, 4, 12, 0, 0), datetime(2021, 5, 5, 0, 0, 0),
              datetime(2021, 5, 5, 12, 0, 0),
              datetime(2021, 5, 6, 0, 0, 0), datetime(2021, 5, 6, 12, 0, 0)]
marker_dts = [datetime(2021, 4, 22, 0, 0, 0), datetime(2021, 4, 23, 0, 0, 0),
              datetime(2021, 4, 24, 0, 0, 0), datetime(2021, 4, 25, 0, 0, 0),
              datetime(2021, 4, 26, 0, 0, 0), datetime(2021, 4, 27, 0, 0, 0),
              datetime(2021, 4, 28, 0, 0, 0), datetime(2021, 4, 29, 0, 0, 0),
              datetime(2021, 4, 30, 0, 0, 0), datetime(2021, 5, 1, 0, 0, 0),
              datetime(2021, 5, 2, 0, 0, 0), datetime(2021, 5, 3, 0, 0, 0),
              datetime(2021, 5, 4, 0, 0, 0), datetime(2021, 5, 5, 0, 0, 0), ]
marker_dts = [datetime(2021, 4, 28, 0, 0, 0), datetime(2021, 4, 29, 0, 0, 0),
              datetime(2021, 4, 30, 0, 0, 0), datetime(2021, 5, 1, 0, 0, 0), ]
marker_dts = [datetime(2021, 4, 28, 15, 20), datetime(2021, 4, 29, 13, 40), datetime(2021, 4, 30, 4, 30)]
# marker_dts=[datetime(2018,10,24,0,0,0),datetime(2018,10,25,0,0,0),datetime(2018,10,26,0,0,0),datetime(2018,10,27,0,0,0),
#             datetime(2018,10,28,0,0,0),datetime(2018,10,29,0,0,0),datetime(2018,10,30,0,0,0),datetime(2018,10,31,0,0,0),
#             datetime(2018,11,1,0,0,0),datetime(2018,11,2,0,0,0),datetime(2018,11,3,0,0,0),datetime(2018,11,4,0,0,0),
#             datetime(2018,11,5,0,0,0),datetime(2018,11,6,0,0,0),datetime(2018,11,7,0,0,0),datetime(2018,11,8,0,0,0),
#             datetime(2018,11,9,0,0,0),datetime(2018,11,10,0,0,0),datetime(2018,11,11,0,0,0),datetime(2018,11,12,0,0,0),
#             datetime(2018,11,13,0,0,0),datetime(2018,11,14,0,0,0),datetime(2018,11,15,0,0,0),datetime(2018,11,16,0,0,0),]
# marker_dts = [datetime(2022,2,15),datetime(2022,2,16),datetime(2022,2,17),datetime(2022,2,18),datetime(2022,2,19),
#               datetime(2022,2,20),datetime(2022,2,21),datetime(2022,2,22),datetime(2022,2,23),datetime(2022,2,24),
#               datetime(2022,2,25),datetime(2022,2,26),datetime(2022,2,27),datetime(2022,2,28),datetime(2022,3,1),
#               datetime(2022,3,2),datetime(2022,3,3),datetime(2022,3,4),datetime(2022,3,5),datetime(2022,3,6),
#               datetime(2022,3,7),
#               ]
marker_ets = [spice.datetime2et(dt) for dt in marker_dts]
marker_txts = [dt.strftime('%m%d\n%H%M') for dt in marker_dts]

psp_obs_cross, _ = spice.spkpos('SPP', marker_ets, 'IAU_SUN', 'NONE', 'SUN')  # km
psp_obs_cross = np.array(psp_obs_cross.T / Rs)
# print(psp_obs_cross)
AU = 1.49597871e8
r_1au = AU / Rs
from plot_body_positions import rtp2xyz_in_Carrington

for i in range(len(marker_ets)):
    print(psp_obs_cross[:, i])
    r_psp_obs_cross, lon_psp_obs_cross, lat_psp_obs_cross = xyz2rtp_in_Carrington(psp_obs_cross[:, i], for_psi=True)
    print(r_psp_obs_cross, lon_psp_obs_cross, lat_psp_obs_cross)
    r_vect = np.linspace(r_psp_obs_cross, r_1au, num=100)
    # print(r_vect)
    r_ind = np.argmin(abs(r_psp_obs_cross - r_vr))
    p_ind = np.argmin(abs(lon_psp_obs_cross - p_vr))
    t_ind = np.argmin(abs(lat_psp_obs_cross - t_vr))

    vr_tmp = vr[p_ind, t_ind, r_ind]
    vr_vect = r_vect * 0 + vr_tmp
    lon_vect, lat_vect = parker_spiral(r_vect * Rs / AU, np.rad2deg(np.pi / 2 - lat_psp_obs_cross),
                                       np.rad2deg(lon_psp_obs_cross), vr_vect)
    print(lon_vect[0], lat_vect[0])
    lon_vect = np.deg2rad(lon_vect)
    lat_vect = np.deg2rad(lat_vect)
    print(lon_vect[0], lat_vect[0])
    x_vect, y_vect, z_vect = rtp2xyz_in_Carrington([r_vect, lon_vect, lat_vect], for_psi=False)
    plot.add_trace(go.Scatter3d(x=x_vect, y=y_vect, z=(z_vect * 0 + z_vect) / 2,
                                mode='lines', line=dict(color='white', width=5), ))
    print('----------')
# psp_obs_cross = {'x': psp_obs_cross[0], 'y': psp_obs_cross[1], 'z': psp_obs_cross[2]}
# psp_obs_cross = pd.DataFrame(data=psp_obs_cross)
plot.add_trace(go.Scatter3d(x=np.array(psp_obs_cross[0]), y=np.array(psp_obs_cross[1]), z=np.array(psp_obs_cross[2]),
                            mode='markers+text',
                            marker=dict(size=3,
                                        color='white',
                                        symbol='diamond'),
                            text=marker_txts, textfont=dict(size=10, color='white'),
                            # name='Orbit of PSP (' + start_time + '~' + stop_time + ')',
                            ))

# put_psp_stl_pos = psp_obs_cross[:,11]
# print(put_psp_stl_pos)
# lon = np.rad2deg(np.arccos(put_psp_stl_pos[0]/np.sqrt(put_psp_stl_pos[1]**2+put_psp_stl_pos[0]**2)))
# print(lon)
# if put_psp_stl_pos[1]<0:
#     lon = 360-lon
# print(clon+rot_angle)

# plot.add_trace(add_texture(put_psp_stl_pos,lon+180,scale=10))


# psp_simu_cross, _ = spice.spkpos('SPP', simu_cross, 'IAU_SUN', 'NONE', 'SUN')  # km
# psp_simu_cross = np.array(psp_simu_cross.T / Rs)
# print(psp_simu_cross)
# # psp_simu_cross = {'x': psp_simu_cross[0], 'y': psp_simu_cross[1], 'z': psp_simu_cross[2]}
# # psp_simu_cross = pd.DataFrame(data=psp_simu_cross)
# plot.add_trace(go.Scatter3d(x=np.array(psp_simu_cross[0]), y=np.array(psp_simu_cross[1]), z=np.array(psp_simu_cross[2]),
#                             mode='markers',
#                             marker=dict(size=5,
#                                         color='green',
#                                         symbol='diamond'),
#                             name='Orbit of PSP (' + start_time + '~' + stop_time + ')',))


plot.update_layout(
    title=dict(
        text="HCS Crossing",
        y=0.9, x=0.5,
        xanchor='center',
        yanchor='top'),
    scene=dict(
        xaxis_title='X (Rs)',
        yaxis_title='Y (Rs)',
        zaxis_title='Z (Rs)',
        # xaxis_range=[-80, 40],
        # yaxis_range=[-40, 80],
        # zaxis_range=[-25, 25],
        # aspectratio=dict(x=1.2,y=1.2,z=0.5)),
    ),
    showlegend=False,
    # legend=dict(
    #     orientation="h",
    #     yanchor="bottom",
    #     y=0.8,
    #     xanchor="right",
    #     x=1, ),
    template='seaborn',
    margin=dict(autoexpand=False, b=0, t=0)
)
py.plot(plot, filename='HCS_IH(' + start_time + '_' + stop_time + ').html')

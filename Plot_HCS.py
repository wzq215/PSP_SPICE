import pandas as pd
import h5py
import numpy as np
import spiceypy as spice
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import furnsh_kernels
import plotly.offline as py
import plotly.graph_objects as go
import pyvista
from ps_read_hdf_3d import ps_read_hdf_3d
from plot_body_positions import xyz2rtp_in_Carrington
Rs = 696300  # km
# ========Data Preparation=======

# set time range for PSP orbit
start_time = '2021-01-11'
stop_time = '2021-01-25'
start_dt = datetime.strptime(start_time, '%Y-%m-%d')
stop_dt = datetime.strptime(stop_time, '%Y-%m-%d')
utc = [start_dt.strftime('%b %d, %Y'), stop_dt.strftime('%b %d, %Y')]
etOne = spice.str2et(utc[0])
etTwo = spice.str2et(utc[1])

step = 100
times = [x * (etTwo - etOne) / step + etOne for x in range(step)]

psp_utc_str = '20210117T131000'

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
data_br = ps_read_hdf_3d(2239, 'corona', 'br002', periodicDim=3)
# data_br = h5py.File('simulation/20210117T131000/corona_h5/br002.h5')
r_br = np.array(data_br['scales1'])  # 201 in Rs, distance from sun
t_br = np.array(data_br['scales2'])  # 150 in rad, latitude
p_br = np.array(data_br['scales3'])  # 256 in rad, Carrington longitude
br = np.array(data_br['datas'])  # 1CU = 2.205G = 2.205e-4T = 2.205e5nT
br = br * 2.205e5  # nT
# print(t_br)
data_rho = ps_read_hdf_3d(2239, 'corona', 'rho002', periodicDim=3)#h5py.File('simulation/20210117T131000/corona_h5/rho002.h5')
r_rho = np.array(data_rho['scales1'])  # 201 in Rs, distance from sun
t_rho = np.array(data_rho['scales2'])  # 150 in rad, latitude
p_rho = np.array(data_rho['scales3'])  # 256 in rad, Carrington longitude
rho = np.array(data_rho['datas'])
rho = rho * 1e8  # cm^-3
# print(t_rho)
data_vr = ps_read_hdf_3d(2239, 'corona', 'vr002', periodicDim=3)
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
isos = mesh.contour(isosurfaces=1, rng=[0, 0])
# isos.plot(opacity=0.7)

# get Isosurface of rho*r^2
tv2, pv2, rv2 = np.meshgrid(t_rho, p_rho, r_rho, indexing='xy')

xv2 = rv2 * np.cos(pv2) * np.sin(tv2)
yv2 = rv2 * np.sin(pv2) * np.sin(tv2)
zv2 = rv2 * np.cos(tv2)
mesh2 = pyvista.StructuredGrid(xv2, yv2, zv2)
rholog = rho*rv2**2
print('min',np.nanmin(rholog))
print('max',np.nanmax(rholog))
mesh2.point_data['values'] = rholog.ravel(order='F')  # also the active scalars
isos2 = mesh2.contour(isosurfaces=1,rng=[8e5,8e5])
isos2.plot(opacity=0.9)


# Color HCS by Vr
vertices = isos.points
triangles = isos.faces.reshape(-1, 4)

vr_points = vertices[:, 0] * 0
for i in range(len(vertices)):

    point = np.array(vertices[i])
    r_p,p_p,t_p = xyz2rtp_in_Carrington(point,for_psi=True)

    r_ind = np.argmin(abs(r_p - r_vr))
    p_ind = np.argmin(abs(p_p - p_vr))
    t_ind = np.argmin(abs(t_p - t_vr))

    vr_points[i] = vr[p_ind, t_ind, r_ind] #* (r_p) ** 2

# intensity = np.log10(np.array(rho_points)).reshape(-1, 1)
intensity = np.array(vr_points).reshape(-1,1)


# ===========Plot============
plot = go.Figure()
plot.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                         opacity=1,colorscale='jet',
                         # colorscale='Viridis',
                         cmax=450, cmin=200,
                         i=triangles[:, 1], j=triangles[:, 2], k=triangles[:, 3],
                         intensity=intensity,
                         # showscale=False,
                         ))

vertices2 = isos2.points
triangles2 = isos2.faces.reshape(-1, 4)
plot.add_trace(go.Mesh3d(x=vertices2[:, 0], y=vertices2[:, 1], z=vertices2[:, 2],
                         opacity=0.7,color='grey',
                         #colorscale='jet',
                         # colorscale='Viridis',
                         # cmax=450, cmin=150,
                         i=triangles2[:, 1], j=triangles2[:, 2], k=triangles2[:, 3],
                         # intensity=intensity,
                         showscale=False,
                         ))

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

obs_cross = spice.datetime2et(datetime.strptime('20210117T133000', '%Y%m%dT%H%M%S'))
simu_cross = spice.datetime2et(datetime.strptime('20210117T200000','%Y%m%dT%H%M%S'))

plot.add_trace(go.Scatter3d(x=psp_pos['x'], y=psp_pos['y'], z=psp_pos['z'],
                            mode='lines',
                            line=dict(color='yellow',
                                      width=10),
                            name='Orbit of PSP (' + start_time + '~' + stop_time + ')',
                            ))
psp_obs_cross, _ = spice.spkpos('SPP', obs_cross, 'IAU_SUN', 'NONE', 'SUN')  # km
psp_obs_cross = np.array(psp_obs_cross.T / Rs)
# psp_obs_cross = {'x': psp_obs_cross[0], 'y': psp_obs_cross[1], 'z': psp_obs_cross[2]}
# psp_obs_cross = pd.DataFrame(data=psp_obs_cross)
plot.add_trace(go.Scatter3d(x=np.array(psp_obs_cross[0]), y=np.array(psp_obs_cross[1]), z=np.array(psp_obs_cross[2]),
                            mode='markers',
                            marker=dict(size=5,
                                        color='blue',
                                        symbol='diamond'),
                            name='Orbit of PSP (' + start_time + '~' + stop_time + ')',))

psp_simu_cross, _ = spice.spkpos('SPP', simu_cross, 'IAU_SUN', 'NONE', 'SUN')  # km
psp_simu_cross = np.array(psp_simu_cross.T / Rs)
# psp_simu_cross = {'x': psp_simu_cross[0], 'y': psp_simu_cross[1], 'z': psp_simu_cross[2]}
# psp_simu_cross = pd.DataFrame(data=psp_simu_cross)
plot.add_trace(go.Scatter3d(x=np.array(psp_simu_cross[0]), y=np.array(psp_simu_cross[1]), z=np.array(psp_simu_cross[2]),
                            mode='markers',
                            marker=dict(size=5,
                                        color='green',
                                        symbol='diamond'),
                            name='Orbit of PSP (' + start_time + '~' + stop_time + ')',))


plot.update_layout(
    title=dict(
        text="HCS Crossing",
        y=0.9, x=0.5,
        xanchor='center',
        yanchor='top'),
    scene=dict(
        xaxis_title='X (Rs)',
        yaxis_title='Y (Rs)',
        zaxis_title='Z (Rs)', ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1, ),
)
py.plot(plot, filename='test.html', image='svg')

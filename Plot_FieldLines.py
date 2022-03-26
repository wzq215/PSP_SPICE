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

start_time = '2021-01-16'
stop_time = '2021-01-19'
start_dt = datetime.strptime(start_time, '%Y-%m-%d')
stop_dt = datetime.strptime(stop_time, '%Y-%m-%d')
utc = [start_dt.strftime('%b %d, %Y'), stop_dt.strftime('%b %d, %Y')]
etOne = spice.str2et(utc[0])
etTwo = spice.str2et(utc[1])

# Epochs
step = 100
times = [x * (etTwo - etOne) / step + etOne for x in range(step)]

psp_utc_str = '20210117T131000'
Rs = 696300  # km
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

data_br = h5py.File('simulation/20210117T131000/corona_h5/br002.h5')
r_br = np.array(data_br['scales1'])  # 201 in Rs, distance from sun
t_br = np.array(data_br['scales2'])  # 150 in rad, latitude
p_br = np.array(data_br['scales3'])  # 256 in rad, Carrington longitude
br = np.array(data_br['datas'])  # 1CU = 2.205G = 2.205e-4T = 2.205e5nT
br = br * 2.205e5  # nT
print(t_br)
data_rho = h5py.File('simulation/20210117T131000/corona_h5/rho002.h5')
r_rho = np.array(data_rho['scales1'])  # 201 in Rs, distance from sun
t_rho = np.array(data_rho['scales2'])  # 150 in rad, latitude
p_rho = np.array(data_rho['scales3'])  # 256 in rad, Carrington longitude
rho = np.array(data_rho['datas'])
rho = rho * 1e8  # cm^-3
print(t_rho)
data_vr = h5py.File('simulation/20210117T131000/corona_h5/vr002.h5')
r_vr = np.array(data_vr['scales1'])  # 201 in Rs, distance from sun
t_vr = np.array(data_vr['scales2'])  # 150 in rad, latitude
p_vr = np.array(data_vr['scales3'])  # 256 in rad, Carrington longitude
vr = np.array(data_vr['datas'])
vr = vr * 481.3711  # km/s
print(t_vr)
p_index = abs(np.rad2deg(p_br) - subpnt_lat[0]).argmin()
t_index = abs(np.rad2deg(t_br) - subpnt_lat[1]).argmin()
r_index = abs(r_vr - psp_r).argmin()


tv, pv, rv = np.meshgrid(t_br, p_br, r_br, indexing='xy')


xv = rv * np.cos(pv) * np.sin(tv)
yv = rv * np.sin(pv) * np.sin(tv)
zv = rv * np.cos(tv)

mesh = pyvista.StructuredGrid(xv, yv, zv)
mesh.point_data['values'] = br.ravel(order='F')  # also the active scalars
isos = mesh.contour(isosurfaces=1, rng=[0, 0])
isos.plot(opacity=0.7)

plot = go.Figure()

vertices = isos.points
triangles = isos.faces.reshape(-1, 4)
plot.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                         opacity=0.8, colorscale='Viridis', cmax=15, cmin=-15,
                         i=triangles[:, 1], j=triangles[:, 2], k=triangles[:, 3],
                         intensity=vertices[:, 2],
                         showscale=False,
                         ))

rv = rv[p_index - 10:p_index + 10, t_index - 20:t_index + 20, :]
pv = pv[p_index - 10:p_index + 10, t_index - 20:t_index + 20, :]
tv = tv[p_index - 10:p_index + 10, t_index - 20:t_index + 20, :]
br_plot = br[p_index - 10:p_index + 10, t_index - 20:t_index + 20, :]
rho_plot = rho[p_index - 10:p_index + 10, t_index - 20:t_index + 20, :]
vr_plot = vr[p_index - 10:p_index + 10, t_index - 20:t_index + 20, :]
xv = rv * np.cos(pv) * np.sin(tv)
yv = rv * np.sin(pv) * np.sin(tv)
zv = rv * np.cos(tv)

# plot.add_trace(go.Scatter3d(x=xv.flatten(), y=yv.flatten(), z=zv.flatten(),
#                             mode='markers',
#                             marker=dict(size=2,
#                                         color=vr_plot.flatten(),
#                                         colorscale='rainbow',
#                                         showscale=True,
#                                         colorbar=dict(title='Density (cm^-3)',
#                                                       tickvals=[3, 4, 5, 6],
#                                                       ticktext=['10^3', '10^4', '10^5', '10^6']
#                                                       # dtick='log',
#                                                       # exponentformat='power',
#                                                       ),
#                                         cmax=6,
#                                         cmin=3,
#                                         opacity=0.1),
#                             name='Radial Magnetic Field',
#                             ))
plot.add_trace(go.Scatter3d(x=xv.flatten(), y=yv.flatten(), z=zv.flatten(),
                            mode='markers',
                            marker=dict(size=2,
                                        color=vr_plot.flatten(),
                                        colorscale='rainbow',
                                        showscale=True,
                                        colorbar=dict(title='Radial Velocity (cm^-3)',
                                                      # tickvals=[3, 4, 5, 6],
                                                      # ticktext=['10^3', '10^4', '10^5', '10^6']
                                                      # dtick='log',
                                                      # exponentformat='power',
                                                      ),
                                        # cmax=6,
                                        # cmin=3,
                                        opacity=0.1),
                            name='Radial Magnetic Field',
                            ))

psp_pos, _ = spice.spkpos('SPP', times, 'IAU_SUN', 'NONE', 'SUN')  # km
psp_pos = psp_pos.T / Rs
psp_pos = {'x': psp_pos[0], 'y': psp_pos[1], 'z': psp_pos[2]}
psp_pos = pd.DataFrame(data=psp_pos)

plot.add_trace(go.Scatter3d(x=psp_pos['x'], y=psp_pos['y'], z=psp_pos['z'],
                            mode='lines',
                            line=dict(color='darkblue',
                                      width=5),
                            name='Orbit of PSP (' + start_time + '~' + stop_time + ')',
                            ))

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

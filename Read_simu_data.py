import pandas as pd
# import h5py
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
from scipy.interpolate import griddata
# from load_psp_data import load_RTN_1min_data, load_spc_data,load_spe_data,load_spi_data
from plotly.subplots import make_subplots
from ps_read_hdf_3d import ps_read_hdf_3d

Rs = 696300  # km

start_time = '20210116'
stop_time = '20210120'
start_dt = datetime.strptime(start_time, '%Y%m%d')
stop_dt = datetime.strptime(stop_time, '%Y%m%d')
utc = [start_dt.strftime('%b %d, %Y'), stop_dt.strftime('%b %d, %Y')]
etOne = spice.str2et(utc[0])
etTwo = spice.str2et(utc[1])
# Epochs
step = 100
times = [x * (etTwo - etOne) / step + etOne for x in range(step)]

# read simulation data

# data_rho_corona = ps_read_hdf_3d(2239, 'corona', 'rho002', periodicDim=3)
# r_rho_corona = np.array(data_rho_corona['scales1'])  # in Rs, distance from sun
# t_rho_corona = np.array(data_rho_corona['scales2'])  # in rad, latitude
# p_rho_corona = np.array(data_rho_corona['scales3'])  # in rad, Carrington longitude
# rho_corona = np.array(data_rho_corona['datas'])  # 1CU = 10^8 cm^-3
# rho_corona = rho_corona * 1e8 * 1e6 # m^-3

# data_br = h5py.File('simulation/20210117T131000/corona_h5/br002.h5')
# data_rho = h5py.File('simulation/20210117T131000/corona_h5/rho002.h5')
data_br = ps_read_hdf_3d(2239, 'corona', 'br002', periodicDim=3)
data_rho = ps_read_hdf_3d(2239, 'corona', 'rho002', periodicDim=3)
r = np.array(data_br['scales1'])  # 201 in Rs, distance from sun
t = np.array(data_br['scales2'])  # 150 in rad, latitude
p = np.array(data_br['scales3'])  # 256 in rad, Carrington longitude
tv, pv, rv = np.meshgrid(t, p, r, indexing='xy')
print(np.shape(rv))
print(np.shape(tv))
print(np.shape(pv))

br = np.array(data_br['datas'])  # 1CU = 2.205G = 2.205e-4T = 2.205e5nT
br = br * 2.205e5  # nT
print(np.shape(br))

rho_r = np.array(data_rho['scales1'])
rho_t = np.array(data_rho['scales2'])
rho_p = np.array(data_rho['scales3'])
rho = np.array(data_rho['datas'])
rho_tv, rho_pv, rho_rv = np.meshgrid(rho_t,rho_p,rho_r,indexing ='xy')
rho = rho*1e8 # cm^-3

print(np.shape(rho_rv))
print(np.shape(rho_tv))
print(np.shape(rho_pv))
print(np.shape(rho))

# read psp data
# B_rtn=load_RTN_1min_data(start_time,stop_time)
# SPI = load_spi_data(start_time,stop_time)
# np.save('B_rtn.npy',B_rtn)
# np.save('SPI.npy',SPI)
# B_rtn = np.load('B_rtn.npy',allow_pickle=True)
# SPI = np.load('SPI.npy',allow_pickle=True)
# psp_utc_str = '20210117T131000'

# et = spice.datetime2et(datetime.strptime(psp_utc_str, '%Y%m%dT%H%M%S'))
# subpnt_psp = spice.subpnt('INTERCEPT/ELLIPSOID', 'SUN', et, 'IAU_SUN', 'None', 'SPP')
# subpnt_lat = np.rad2deg(spice.reclat(subpnt_psp[0])[1:])
# if subpnt_lat[0] < 0:
#     subpnt_lat[0] = subpnt_lat[0] + 360
# subpnt_lat[1] = 90 - subpnt_lat[1]
# psp_r = np.linalg.norm(subpnt_psp[0] - subpnt_psp[2], 2) / Rs
# print('-------PSP-------')
# print(psp_r)
# print(subpnt_lat[0])
# print(subpnt_lat[1])



# p_index = abs(np.rad2deg(p) - subpnt_lat[0]).argmin()
# t_index = abs(np.rad2deg(t) - subpnt_lat[1]).argmin()
# r_index = abs(r - psp_r).argmin()
# print('--------simu--------')
# print(r[r_index])
# print(np.rad2deg(p[p_index]))
# print(np.rad2deg(t[t_index]))
# print(br.shape)
#

# fig = plt.figure(figsize=(5, 5))
# ax = plt.gca(projection='polar')
#
# plt.pcolormesh(pv[:, t_index, :], rv[:, t_index, :], br[:, t_index, :])
#
# print(subpnt_lat[1])

# plt.colorbar()
# plt.clim(-200, 200)
# plt.scatter(np.deg2rad(subpnt_lat[0]), psp_r, c='k')
# # plt.ylabel('distance from sun (Rs)')
# plt.xlabel('Carrington Longitude (deg)')
# plt.title('Br (in code unit)(' + psp_utc_str + ')')
# plt.show()

# fig = plt.figure()
# plt.pcolormesh(np.rad2deg(pv[:, :, r_index]), np.rad2deg(tv[:, :, r_index]), br[:, :, r_index], cmap='seismic')
# plt.colorbar()
# plt.clim(-200, 200)
# plt.scatter(subpnt_lat[0], subpnt_lat[1], c='k')
# plt.gca().invert_yaxis()
# plt.xlabel('Carrington Longitude (deg)')
# plt.ylabel('Latitude (deg)')
# plt.title('Br (in code unit) (' + psp_utc_str + ')')
# plt.show()


'''模拟飞行器的观测结果'''
# 模拟密度和磁场的观测
# （1） 插值：飞行器位置对应一个模拟数据中的坐标
br_simu=[]
rho_simu=[]
'''x = np.linspace(-30, 30, 101)
y = np.linspace(-30, 30, 101)
z = np.linspace(-30, 30, 101)
xx, yy, zz = np.meshgrid(x, y, z)
points = np.zeros((256 * 150 * 201, 3))
points[:, 0] = xv.flatten()
points[:, 1] = yv.flatten()
points[:, 2] = zv.flatten()
values = br.flatten()
# 需要先从原网格插值到均匀网格
grid_data = griddata(points, values, (xx, yy, zz), method='nearest')'''

psp_p = []
psp_t = []
psp_r = []
from plot_body_positions import xyz2rtp_in_Carrington
for et in times:
    psp_pos, _ = spice.spkpos('SPP', et, 'IAU_SUN', 'NONE', 'SUN')  # km
    psp_pos = psp_pos.T / Rs
    r_psp = np.linalg.norm(psp_pos[0:3],2)
    if psp_pos[1]>0:
        p_psp = np.arccos(psp_pos[0]/r_psp)
    elif psp_pos[1]<=0:
        p_psp = 2*np.pi-np.arccos(psp_pos[0]/r_psp)
    t_psp = np.arccos(psp_pos[2] / r_psp)
    print('rtp',r_psp,t_psp,p_psp)
    r_psp,p_psp,t_psp = xyz2rtp_in_Carrington(psp_pos)
    t_psp = np.pi/2-t_psp
    print('xyz2rtp',r_psp,t_psp,p_psp)

    subpnt_psp = spice.subpnt('INTERCEPT/ELLIPSOID', 'SUN', et, 'IAU_SUN', 'None', 'SPP')
    subpnt_lat = np.rad2deg(spice.reclat(subpnt_psp[0])[1:])
    if subpnt_lat[0] < 0:
        subpnt_lat[0] = subpnt_lat[0] + 360
    subpnt_lat[1] = 90 - subpnt_lat[1]
    # psp_r = np.linalg.norm(subpnt_psp[0] - subpnt_psp[2], 2) / Rs
    psp_r.append(np.linalg.norm(subpnt_psp[0] - subpnt_psp[2], 2) / Rs)
    psp_p.append(np.deg2rad(subpnt_lat[0]))
    psp_t.append(np.deg2rad(subpnt_lat[1]))
    print('rtp_subpnt',np.linalg.norm(subpnt_psp[0] - subpnt_psp[2], 2) / Rs,np.deg2rad(subpnt_lat[1]),np.deg2rad(subpnt_lat[0]))

    p_index = abs(p-p_psp).argmin()
    t_index = abs(t-t_psp).argmin()
    r_index = abs(r-r_psp).argmin()
    br_simu.append(br[p_index,t_index,r_index])
    rhop_index = abs(rho_p-p_psp).argmin()
    rhot_index = abs(rho_t-t_psp).argmin()
    rhor_index = abs(rho_r-r_psp).argmin()
    rho_simu.append(rho[rhop_index,rhot_index,rhor_index])

from scipy import interpolate
tck = interpolate.splrep(times, br_simu)
# newt = np.linspace(times[0],times[-1],100)
br_bspline = interpolate.splev(times, tck)

tck = interpolate.splrep(times, rho_simu)
# newt = np.linspace(times[0],times[-1],100)
rho_bspline = interpolate.splev(times, tck)


# epoch_br_obs = B_rtn['epoch_mag_RTN_1min']
# br_obs = B_rtn['psp_fld_l2_mag_RTN_1min'][:, 0]
# epoch_np_obs = SPI['Epoch']
# np_obs = SPI['DENS']
# np.save('epoch_br_obs.npy',np.array(epoch_br_obs),allow_pickle=True)
# np.save('br_obs.npy',np.array(br_obs),allow_pickle=True)
# np.save('epoch_np_obs.npy',np.array(epoch_np_obs),allow_pickle=True)
# np.save('np_obs.npy',np.array(np_obs),allow_pickle=True)

epoch_br_obs = np.load('epoch_br_obs.npy',allow_pickle=True)
br_obs = np.load('br_obs.npy',allow_pickle=True)
epoch_np_obs = np.load('epoch_np_obs.npy',allow_pickle=True)
np_obs = np.load('np_obs.npy',allow_pickle=True)



fig=go.Figure()
fig = make_subplots(rows=2, cols=1,
                    subplot_titles=("B_R (obs vs model)", "N_p (obs vs model)"))
fig.add_trace(go.Scatter(x=epoch_br_obs, y=br_obs, name='Br_obs', mode='lines'), row=1, col=1)
fig.add_trace(go.Scatter(x=spice.et2datetime(times),y=br_bspline,name='Br_model',mode='markers',line_shape='linear'),row=1,col=1)
fig.add_trace(go.Scatter(x=epoch_np_obs,y=np_obs,name='Np_obs',mode='lines'),row=2,col=1)
fig.add_trace(go.Scatter(x=spice.et2datetime(times),y=np.array(rho_bspline),mode='markers',name='Np_model',line_shape='linear'),row=2,col=1)
fig.update_xaxes(tickformat="%m/%d %H:%M\n%Y", title_text="Epoch")

py.plot(fig)
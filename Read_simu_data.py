# import h5py
from datetime import datetime, timedelta
from ps_read_hdf_3d import ps_read_hdf_3d
import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
import pandas as pd
import sunpy.map
import furnsh_kernels
import plotly.graph_objects as go
import plotly.offline as py
from plotly.subplots import make_subplots

#

Rs = 696300  # km

# datetime_beg = datetime(2022, 4, 28, 12, 0, 0)
# datetime_end = datetime(2021, 4, 30, 12, 0, 0)
datetime_psp_beg = datetime(2022, 2, 25, 0, 0, 0)
datetime_psp_end = datetime(2022, 2, 26, 0, 0, 0)

timestr_psp_beg = datetime_psp_beg.strftime('%Y%m%dT%H%M%S')
timestr_psp_end = datetime_psp_end.strftime('%Y%m%dT%H%M%S')

datetime_beg = datetime(2022, 2, 16, 0, 0, 0)
datetime_end = datetime(2022, 3, 12, 0, 0, 0)

timestep = timedelta(minutes=15)
steps = (datetime_end - datetime_beg) // timestep + 1
dttimes = np.array([x * timestep + datetime_beg for x in range(steps)])
times = spice.datetime2et(dttimes)

# read simulation data
crid = 2254
data_br = ps_read_hdf_3d(crid, 'corona', 'br002', periodicDim=3)
br = np.array(data_br['datas'])  # 1CU = 2.205G = 2.205e-4T = 2.205e5nT
r = np.array(data_br['scales1'])  # 201 in Rs, distance from sun
t = np.array(data_br['scales2'])  # 150 in rad, latitude
p = np.array(data_br['scales3'])  # 256 in rad, Carrington longitude

tv, pv, rv = np.meshgrid(t, p, r, indexing='xy')
br = br * 2.205e5  # nT

data_rho = ps_read_hdf_3d(crid, 'helio', 'rho002', periodicDim=3)
rho_r = np.array(data_rho['scales1'])
rho_t = np.array(data_rho['scales2'])
rho_p = np.array(data_rho['scales3'])
rho = np.array(data_rho['datas'])
rho_tv, rho_pv, rho_rv = np.meshgrid(rho_t, rho_p, rho_r, indexing='xy')
rho = rho * 1e8  # cm^-3

data_bt = ps_read_hdf_3d(crid, 'helio', 'bt002', periodicDim=3)
data_bp = ps_read_hdf_3d(crid, 'corona', 'bp002', periodicDim=3)

bt = np.array(data_bt['datas'])  # 1CU = 2.205G = 2.205e-4T = 2.205e5nT
bt = bt * 2.205e5  # nT
bp = np.array(data_bp['datas'])  # 1CU = 2.205G = 2.205e-4T = 2.205e5nT
bp = bp * 2.205e5  # nT
bt = bt[:, 0:-1, :]
bp = bp[0:-1, :, :]
br = br[:, :, 0:-1]
r = r[0:-1]

b_abs = np.sqrt(br ** 2 + bt ** 2 + bp ** 2)

psp_data_df_pmom = pd.read_csv(
    'export/load_psp_data/psp_overview/pmom_(' + timestr_psp_beg + '-' + timestr_psp_end + ').csv')
epoch_np_obs = psp_data_df_pmom['epochp']
np_obs = psp_data_df_pmom['Np']

psp_data_df_mag = pd.read_csv(
    'export/load_psp_data/psp_overview/mag_(' + timestr_psp_beg + '-' + timestr_psp_end + ').csv')
epoch_br_obs = psp_data_df_mag['epochmag']
br_obs = psp_data_df_mag['Br']

# epoch_br_obs = np.load('epoch_br_obs.npy', allow_pickle=True)
# br_obs = np.load('br_obs.npy', allow_pickle=True)
# epoch_np_obs = np.load('epoch_np_obs.npy', allow_pickle=True)
# np_obs = np.load('np_obs.npy', allow_pickle=True)


'''模拟飞行器的观测结果'''
# 模拟密度和磁场的观测
# （1） 插值：飞行器位置对应一个模拟数据中的坐标
br_simu = []
rho_simu = []

psp_p = []
psp_t = []
psp_r = []
from plot_body_positions import xyz2rtp_in_Carrington

for et in times:
    psp_pos, _ = spice.spkpos('SPP', et, 'IAU_SUN', 'NONE', 'SUN')  # km
    psp_pos = psp_pos.T / Rs
    r_psp, p_psp, t_psp = xyz2rtp_in_Carrington(psp_pos, for_psi=True)

    psp_r.append(r_psp)
    psp_p.append(p_psp)
    psp_t.append(t_psp)

    p_index = abs(p - p_psp).argmin()
    t_index = abs(t - t_psp).argmin()
    r_index = abs(r - r_psp).argmin()
    print(p_index)
    br_simu.append(br[p_index, t_index, r_index])

    rhop_index = abs(rho_p - p_psp).argmin()
    rhot_index = abs(rho_t - t_psp).argmin()
    rhor_index = abs(rho_r - r_psp).argmin()
    rho_simu.append(rho[rhop_index, rhot_index, rhor_index])

from scipy import interpolate

tck = interpolate.splrep(times, br_simu)
br_bspline = interpolate.splev(times, tck)

tck = interpolate.splrep(times, rho_simu)
rho_bspline = interpolate.splev(times, tck)

fig = go.Figure()
fig = make_subplots(rows=2, cols=1,
                    subplot_titles=("B_R (obs vs model)", "N_p (obs vs model)"),
                    shared_xaxes=True)
fig.add_trace(go.Scatter(x=epoch_br_obs, y=br_obs, name='Br_obs', mode='lines'), row=1, col=1)
fig.add_trace(
    go.Scatter(x=spice.et2datetime(times), y=br_bspline, name='Br_model', mode='markers', line_shape='linear'), row=1,
    col=1)
fig.add_trace(go.Scatter(x=epoch_np_obs, y=np_obs, name='Np_obs', mode='lines'), row=2, col=1)
fig.add_trace(go.Scatter(x=spice.et2datetime(times), y=np.array(rho_bspline), mode='markers', name='Np_model',
                         line_shape='linear'), row=2, col=1)
fig.update_xaxes(tickformat="%m/%d %H:%M\n%Y", title_text="Epoch")

py.plot(fig)
# plt.figure()
# ax=plt.subplot(2,1,1)
# plt.plot(epoch_br_obs,br_obs,'k-',linewidth=1)
# dt = pd.to_datetime(df['epoch_simu'])
# # plt.scatter(df['epoch_simu'],df['Br_bspline'],'r')
# plt.scatter(spice.et2datetime(times),df['Br_bspline'],s=5,color='orangered')
# plt.ylabel('$B_r [nT]$')
# ax.spines['bottom'].set_position(('data',0))
# # ax.set_xticks([])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#
# ax = plt.subplot(2,1,2)
# plt.plot(epoch_np_obs,np_obs,'k-',linewidth=1,label='Observation')
# # plt.scatter(spice.et2datetime(times),np.array(rho_bspline),s=5,color='orangered',label='Simulation')
# plt.ylabel('$n_p [cm^{-3}]$')
# plt.xlabel('Time')
# plt.legend()
# # ax.spines['bottom'].set_position(('data',0))
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.show()

quit()


def get_rlonlat_psp_carr(dt_times):
    r_psp_carr = []
    lon_psp_carr = []
    lat_psp_carr = []
    for dt in dt_times:
        et = spice.datetime2et(dt)
        psp_pos, _ = spice.spkpos('SPP', et, 'IAU_SUN', 'NONE', 'SUN')  # km
        psp_pos = psp_pos.T / Rs
        r_psp = np.linalg.norm(psp_pos[0:3], 2)
        if psp_pos[1] > 0:
            p_psp = np.arccos(psp_pos[0] / r_psp)
        elif psp_pos[1] <= 0:
            p_psp = 2 * np.pi - np.arccos(psp_pos[0] / r_psp)
        t_psp = np.arccos(psp_pos[2] / r_psp)
        # print('rtp',r_psp,t_psp,p_psp)
        r_psp, p_psp, t_psp = xyz2rtp_in_Carrington(psp_pos, for_psi=True)
        # t_psp = np.pi/2-t_psp
        # print('xyz2rtp',r_psp,t_psp,p_psp)
        r_psp_carr.append(r_psp)
        lon_psp_carr.append(p_psp)
        lat_psp_carr.append(t_psp)
    return np.array(r_psp_carr), np.array(lon_psp_carr), np.array(lat_psp_carr)


r_psp_carr_br, lon_psp_carr_br, lat_psp_carr_br = get_rlonlat_psp_carr(epoch_br_obs)
r_psp_carr_np, lon_psp_carr_np, lat_psp_carr_np = get_rlonlat_psp_carr(epoch_np_obs)

pp, tt = np.meshgrid(np.rad2deg(p), 90 - np.rad2deg(t))

import pandas as pd

df_trace = pd.read_csv('E8trace0429.csv')

plt.figure()
# plt.subplot(2,1,1)
plt.pcolormesh(pp, tt, br[:, :, 0].T, cmap='seismic')
plt.colorbar(label='$B_r [nT]$')
plt.clim([-1e6, 1e6])
# plt.colorbar()
for i in range(len(times)):
    plt.plot([np.rad2deg(psp_p[i]), df_trace['MFL_photosphere_lon_deg'][i]],
             [90 - np.rad2deg(psp_t[i]), df_trace['MFL_photosphere_lat_deg'][i]], 'gray', linewidth=0.75)

plt.plot(np.rad2deg(psp_p), 90 - np.rad2deg(psp_t), linewidth=2, c='black', label='Sub-PSP Points')
plt.scatter(df_trace['MFL_photosphere_lon_deg'], df_trace['MFL_photosphere_lat_deg'], s=8, c='green', marker='o',
            label='Foot Points')  # c=np.arange(len(df_trace['Epoch']))
plt.gca().set_aspect(1)
# plt.scatter(np.rad2deg(psp_p[0:-1:8]),90-np.rad2deg(psp_t[0:-1:8]),marker='x',c='red')
print(dttimes[[7, 51, 81]])
plt.scatter(np.rad2deg(psp_p[7]), 90 - np.rad2deg(psp_t[7]), marker='x', c='red')
plt.scatter(np.rad2deg(psp_p[51]), 90 - np.rad2deg(psp_t[51]), marker='x', c='red')
plt.scatter(np.rad2deg(psp_p[81]), 90 - np.rad2deg(psp_t[81]), marker='x', c='red')
plt.title('HMI Magnetograph')
plt.xlabel('Carrington Longitude')
plt.ylabel('Carrington Latitude')
plt.legend()
plt.show()
# plt.subplot(2,1,2)
plt.figure()
plt.pcolormesh(pp, tt, br[:, :, 205].T, cmap='seismic')
plt.colorbar(label='$B_r [nT]$')
plt.scatter(np.rad2deg(lon_psp_carr_br), 90 - np.rad2deg(lat_psp_carr_br), c=br_obs * r_psp_carr_br ** 2 / 1e5, s=1,
            label='Sub-PSP Points')
plt.colorbar(label=r'$B_r r^2 [\times 10^5 nT Rs^2]$')
plt.clim([-1.2, 1.2])
plt.scatter(np.rad2deg(psp_p[0:-1:24]), 90 - np.rad2deg(psp_t[0:-1:24]), marker='x', c='black')
plt.gca().set_aspect(1)
plt.title('Simulated Magnetic Field at 20Rs')
plt.xlabel('Carrington Longitude')
plt.ylabel('Carrington Latitude')
plt.legend()
plt.show()

r_ind = 205

plt.figure()
plt.plot(epoch_np_obs, np_obs * r_psp_carr_np ** 2)
plt.show()
plt.figure()
# plt.subplot(3,1,1)
# plt.pcolormesh(pp,tt,br[:,:,r_ind].T,cmap='seismic')
# # plt.colorbar()
# plt.scatter(np.rad2deg(lon_psp_carr_br),90-np.rad2deg(lat_psp_carr_br),c=br_obs*r_psp_carr_br**2,s=1)
# plt.scatter(np.rad2deg(psp_p[0:-1:24]),90-np.rad2deg(psp_t[0:-1:24]),marker='x',c='black')
# # plt.clim(-1500,1500)
# # plt.colorbar()
# plt.xlim([30,160])
# plt.ylim([-45,45])
# plt.gca().set_aspect(1)
# plt.xticks([])
cmin = 0
cmax = 2e6
ymin = -40
ymax = 30
plt.subplot(2, 2, 2)
plt.pcolormesh(pp, tt, rho[1:, 1:, r_ind].T * 8 ** 2, cmap='seismic')
# print(rho[0:-1,0:-1,100])
plt.colorbar(label=r'$N_p r^2 [cm^{-3} Rs^2]$')
plt.clim([cmin, cmax])
plt.scatter(np.rad2deg(lon_psp_carr_np), 90 - np.rad2deg(lat_psp_carr_np), c=np_obs * r_psp_carr_np ** 2, s=1)
plt.clim([cmin, cmax])
plt.scatter(np.rad2deg(psp_p[0:-1:24]), 90 - np.rad2deg(psp_t[0:-1:24]), marker='x', c='black')
# plt.colorbar()
# plt.clim([1e7,5e7])
plt.xlim([90, 240])
plt.ylim([ymin, ymax])
plt.gca().set_aspect(1)
# plt.xticks([])
plt.ylabel('Carrington Latitude')

plt.subplot(2, 2, 4)
carrmap_data = np.load('carrmap6.npz')
ts = carrmap_data['arr_0']
ps = carrmap_data['arr_1']
carrmap = carrmap_data['arr_2']
plt.pcolormesh(ts, ps, carrmap.T, cmap='viridis')
plt.xlabel('Carrington Longitude')
plt.ylabel('Carrington Latitude')
plt.colorbar(label='Carrington Map')
plt.clim([5e-14, 2e-13])
plt.xlim([90, 240])
plt.ylim([ymin, ymax])
plt.gca().set_aspect(1)
# plt.show()

r_ind = 239
plt.subplot(2, 2, 1)
plt.pcolormesh(pp, tt, rho[1:, 1:, r_ind].T * 20 ** 2, cmap='seismic')
# plt.colorbar()
# print(rho[0:-1,0:-1,100])
plt.clim([cmin, cmax])

plt.scatter(np.rad2deg(lon_psp_carr_np), 90 - np.rad2deg(lat_psp_carr_np), c=np_obs * r_psp_carr_np ** 2, s=1)
plt.colorbar(label=r'$N_p r^2 [cm^{-3} Rs^2]$')
plt.clim([cmin, cmax])

plt.scatter(np.rad2deg(psp_p[0:-1:24]), 90 - np.rad2deg(psp_t[0:-1:24]), marker='x', c='black')
# plt.colorbar()
# plt.clim([1e7,5e7])
plt.xlim([40, 140])
plt.ylim([ymin, ymax])
plt.gca().set_aspect(1)
# plt.xticks([])
plt.ylabel('Carrington Latitude')

plt.subplot(2, 2, 3)
carrmap_data = np.load('carrmap20.npz')
ts = carrmap_data['arr_0']
ps = carrmap_data['arr_1']
carrmap = carrmap_data['arr_2']
plt.pcolormesh(ts, ps, carrmap.T, cmap='viridis')
plt.xlabel('Carrington Longitude')
plt.ylabel('Carrington Latitude')
plt.colorbar(label='Carrington Map')
plt.clim([5e-14, 2e-13])
plt.xlim([40, 140])
plt.ylim([ymin, ymax])
plt.gca().set_aspect(1)
plt.show()

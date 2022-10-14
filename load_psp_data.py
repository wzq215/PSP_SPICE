'''Created by Ziqi Wu. @21/07/02'''
import os
from datetime import datetime
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.offline as py
from matplotlib.ticker import AutoMinorLocator
from plotly.subplots import make_subplots
from spacepy import pycdf

os.environ["CDF_LIB"] = "/usr/local/cdf/lib"
psp_data_path = '/Users/ephe/PSP_Data_Analysis/Encounter08/'


def load_RTN_1min_data(start_time_str, stop_time_str):
    start_time = datetime.strptime(start_time_str, '%Y%m%d').toordinal()
    stop_time = datetime.strptime(stop_time_str, '%Y%m%d').toordinal()
    filelist = [psp_data_path + 'psp_fld_l2_mag_RTN_1min_' + datetime.fromordinal(x).strftime('%Y%m%d') + '_v02.cdf'
                for x in range(start_time, stop_time + 1)]
    print('Brtn 1min Files: ', filelist)
    data = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    # print(data)
    return data


def load_RTN_4sa_data(start_time_str, stop_time_str):
    start_time = datetime.strptime(start_time_str, '%Y%m%d').toordinal()
    stop_time = datetime.strptime(stop_time_str, '%Y%m%d').toordinal()
    filelist = [
        psp_data_path + 'psp_fld_l2_mag_rtn_4_sa_per_cyc_' + datetime.fromordinal(x).strftime('%Y%m%d') + '_v02.cdf'
        for x in range(start_time, stop_time + 1)]
    print('Brtn 1min Files: ', filelist)
    data = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    # print(data)
    return data


def load_RTN_data(start_time_str, stop_time_str):
    '''psp_fld_l2_mag_rtn_2021042800_v02.cdf'''
    cycles = [0, 6, 12, 18]
    start_time = datetime.strptime(start_time_str, '%Y%m%d%H')
    stop_time = datetime.strptime(stop_time_str, '%Y%m%d%H')
    # start_hour = start_time.hour
    # start_index = divmod(start_hour,6)[0]
    start_file_time = datetime(start_time.year, start_time.month, start_time.day, cycles[divmod(start_time.hour, 6)[0]])
    stop_file_time = datetime(stop_time.year, stop_time.month, stop_time.day, cycles[divmod(stop_time.hour, 6)[0]])
    if divmod(stop_time.hour, 6)[1] == 0:
        stop_file_time -= timedelta(hours=6)
    filelist = []
    tmp_time = start_file_time
    while tmp_time <= stop_file_time:
        filelist.append(psp_data_path + 'psp_fld_l2_mag_rtn_' + tmp_time.strftime('%Y%m%d%H') + '_v02.cdf')
        tmp_time += timedelta(hours=6)
    print('Brtn Files: ', filelist)
    data = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    # print(data)
    return data


# def load_spc_data(start_time_str, stop_time_str):
#     # psp/sweap/spc/psp_swp_spc_l3i_20210115_v02.cdf
#     start_time = datetime.strptime(start_time_str, '%Y%m%d').toordinal()
#     stop_time = datetime.strptime(stop_time_str, '%Y%m%d').toordinal()
#     filelist = [psp_data_path + 'psp_swp_spc_l3i_' + datetime.fromordinal(x).strftime('%Y%m%d') + '_v02.cdf'
#                 for x in range(start_time, stop_time)]
#     print('SPC Files: ', filelist)
#     data = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
#     # print(data)
#     return data


def load_spe_data(start_time_str, stop_time_str):
    # psp/sweap/spe/psp_swp_spa_sf0_L3_pad_20210115_v03.cdf
    start_time = datetime.strptime(start_time_str, '%Y%m%d').toordinal()
    stop_time = datetime.strptime(stop_time_str, '%Y%m%d').toordinal()
    filelist = [psp_data_path + 'psp_swp_spe_sf0_L3_pad_' + datetime.fromordinal(x).strftime('%Y%m%d') + '_v03.cdf'
                for x in range(start_time, stop_time + 1)]
    print('SPE Files: ', filelist)
    data = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    # print(data)
    return data


def load_spi_data(start_time_str, stop_time_str, inst=False, species='0'):
    # psp/sweap/spi/psp_swp_spi_sf00_L3_mom_INST_20210115_v03.cdf
    start_time = datetime.strptime(start_time_str, '%Y%m%d').toordinal()
    stop_time = datetime.strptime(stop_time_str, '%Y%m%d').toordinal()
    if inst:
        filelist = [psp_data_path + 'psp_swp_spi_sf0' + species + '_L3_mom_INST_' + datetime.fromordinal(x).strftime(
            '%Y%m%d') + '_v03.cdf'
                    for x in range(start_time, stop_time + 1)]
    else:
        filelist = [psp_data_path + 'psp_swp_spi_sf0' + species + '_L3_mom_' + datetime.fromordinal(x).strftime(
            '%Y%m%d') + '_v04.cdf'
                    for x in range(start_time, stop_time + 1)]
    print('SPI Files: ', filelist)
    data = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    print(data)
    return data


def calc_Z(epochmag, magx, magy, magz, epochpmom, vx, vy, vz, densp):
    # print(epochpmom)
    # print(epochmag)
    mu_0 = 4e-7 * np.pi
    m_p = 1.672621637e-27  # kg
    vx = np.interp(np.array((epochmag - epochpmom[0]) / timedelta(days=1), dtype='float64'),
                   np.array((epochpmom - epochpmom[0]) / timedelta(days=1), dtype='float64'), vx)
    vy = np.interp(np.array((epochmag - epochpmom[0]) / timedelta(days=1), dtype='float64'),
                   np.array((epochpmom - epochpmom[0]) / timedelta(days=1), dtype='float64'), vy)
    vz = np.interp(np.array((epochmag - epochpmom[0]) / timedelta(days=1), dtype='float64'),
                   np.array((epochpmom - epochpmom[0]) / timedelta(days=1), dtype='float64'), vz)
    n0 = np.nanmean(densp)
    # n0 = [np.nanmean((epochpmom >= (dt - window_length / 2)) & (epochpmom <= (dt + window_length / 2)))]
    vAx = magx * 1e-9 / np.sqrt(mu_0 * n0 * 1e6 * m_p) / 1000  # km/s
    vAy = magy * 1e-9 / np.sqrt(mu_0 * n0 * 1e6 * m_p) / 1000  # km/s
    vAz = magz * 1e-9 / np.sqrt(mu_0 * n0 * 1e6 * m_p) / 1000  # km/s
    vAx0, vAy0, vAz0 = np.nanmean(vAx), np.nanmean(vAy), np.nanmean(vAz)
    vx0, vy0, vz0 = np.nanmean(vx), np.nanmean(vy), np.nanmean(vz)
    dBx, dBy, dBz = vAx - vAx0, vAy - vAy0, vAz - vAz0
    dvx, dvy, dvz = vx - vx0, vy - vy0, vz - vz0
    # print([dBx, dBy, dBz])
    # print(np.linalg.norm(np.array([dBx, dBy, dBz]),axis=1))
    dvA = np.nanmean(np.linalg.norm(np.array([dBx, dBy, dBz]), axis=1))
    # print(dvA)
    dv = np.array([dvx, dvy, dvz])
    epoch = epochmag
    zpx, zpy, zpz = dvx - dBx, dvy - dBy, dvz - dBz
    zmx, zmy, zmz = dvx + dBx, dvy + dBy, dvz + dBz
    zplus = np.array([zpx, zpy, zpz])
    zminus = np.array([zmx, zmy, zmz])
    print(zplus)
    print(zminus)
    # print([vAx0,vAy0,vAz0])
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.plot(vAx)
    # plt.plot(vAy)
    # plt.plot(vAz)
    # plt.subplot(2,1,2)
    # plt.plot(dBx)
    # plt.plot(dBy)
    # plt.plot(dBz)
    # plt.show()
    return epoch, zplus, zminus, dvA, dv


def get_windowmean(epochmag, magx, magy, magz, epochpmom, vx, vy, vz, densp):
    vx = np.interp(np.array((epochmag - epochpmom[0]) / timedelta(days=1), dtype='float64'),
                   np.array((epochpmom - epochpmom[0]) / timedelta(days=1), dtype='float64'), vx)
    vy = np.interp(np.array((epochmag - epochpmom[0]) / timedelta(days=1), dtype='float64'),
                   np.array((epochpmom - epochpmom[0]) / timedelta(days=1), dtype='float64'), vy)
    vz = np.interp(np.array((epochmag - epochpmom[0]) / timedelta(days=1), dtype='float64'),
                   np.array((epochpmom - epochpmom[0]) / timedelta(days=1), dtype='float64'), vz)
    n0 = np.nanmean(densp)
    Bx0, By0, Bz0 = np.nanmean(magx), np.nanmean(magy), np.nanmean(magz)
    vx0, vy0, vz0 = np.nanmean(vx), np.nanmean(vy), np.nanmean(vz)
    B0 = [Bx0, By0, Bz0]
    v0 = [vx0, vy0, vz0]
    return n0, B0, v0


def calc_helicity(epochmag, magx, magy, magz, epochpmom, vx, vy, vz, densp, windowsecs=60):
    window_length = timedelta(seconds=windowsecs)
    mu_0 = 4e-7 * np.pi
    m_p = 1.672621637e-27  # kg

    n0 = densp * 0
    magx0, magy0, magz0 = densp * 0, densp * 0, densp * 0
    vx0, vy0, vz0 = densp * 0, densp * 0, densp * 0
    for i in range(len(epochpmom)):
        print(i / len(epochpmom) * 100, '%')
        dt = epochpmom[i]
        subepochpmom_index = (epochpmom >= (dt - window_length / 2)) & (epochpmom <= (dt + window_length / 2))
        subepochmag_index = (epochmag >= (dt - window_length / 2)) & (epochmag <= (dt + window_length / 2))
        subepochpmom = epochpmom[subepochpmom_index]
        subepochmag = epochmag[subepochmag_index]
        submagx, submagy, submagz = np.array(magx[subepochmag_index]), np.array(magy[subepochmag_index]), np.array(
            magz[subepochmag_index])
        subvx, subvy, subvz = np.array(vx[subepochpmom_index]), np.array(vy[subepochpmom_index]), np.array(
            vz[subepochpmom_index])
        subdensp = np.array(densp[subepochpmom_index])
        # _, zplus, zminus, dvA, _ = calc_Z(subepochmag, submagx, submagy, submagz, subepochpmom, subvx, subvy, subvz,
        #                                   subdensp)
        n0[i] = np.nanmean(subdensp)
        magx0[i], magy0[i], magz0[i] = np.nanmean(submagx), np.nanmean(submagy), np.nanmean(submagz)
        vx0[i], vy0[i], vz0[i] = np.nanmean(subvx), np.nanmean(subvy), np.nanmean(subvz)
    magx_p = np.interp(np.array((epochpmom - epochpmom[0]) / timedelta(days=1), dtype='float64'),
                       np.array((epochmag - epochpmom[0]) / timedelta(days=1), dtype='float64'), magx)
    magy_p = np.interp(np.array((epochpmom - epochpmom[0]) / timedelta(days=1), dtype='float64'),
                       np.array((epochmag - epochpmom[0]) / timedelta(days=1), dtype='float64'), magy)
    magz_p = np.interp(np.array((epochpmom - epochpmom[0]) / timedelta(days=1), dtype='float64'),
                       np.array((epochmag - epochpmom[0]) / timedelta(days=1), dtype='float64'), magz)
    vAx = magx_p * 1e-9 / np.sqrt(mu_0 * n0 * 1e6 * m_p) / 1000  # km/s
    vAy = magy_p * 1e-9 / np.sqrt(mu_0 * n0 * 1e6 * m_p) / 1000  # km/s
    vAz = magz_p * 1e-9 / np.sqrt(mu_0 * n0 * 1e6 * m_p) / 1000  # km/s

    vAx0 = magx0 * 1e-9 / np.sqrt(mu_0 * n0 * 1e6 * m_p) / 1000  # km/s
    vAy0 = magy0 * 1e-9 / np.sqrt(mu_0 * n0 * 1e6 * m_p) / 1000  # km/s
    vAz0 = magz0 * 1e-9 / np.sqrt(mu_0 * n0 * 1e6 * m_p) / 1000  # km/s

    dBx, dBy, dBz = vAx - vAx0, vAy - vAy0, vAz - vAz0
    dvx, dvy, dvz = vx - vx0, vy - vy0, vz - vz0

    zpx, zpy, zpz = dvx - dBx, dvy - dBy, dvz - dBz
    zmx, zmy, zmz = dvx + dBx, dvy + dBy, dvz + dBz
    zplus = np.array([zpx, zpy, zpz])
    zminus = np.array([zmx, zmy, zmz])
    print(zplus.shape)

    helicity = (np.abs(np.linalg.norm(zplus, axis=0)) ** 2 - np.abs(np.linalg.norm(zminus, axis=0)) ** 2) / (
            np.abs(np.linalg.norm(zplus, axis=0)) ** 2 + np.abs(np.linalg.norm(zminus, axis=0)) ** 2)

    norm_dB = np.sqrt(dBx ** 2 + dBy ** 2 + dBz ** 2)
    return helicity, norm_dB


def calc_anisotropic_temp(T_tensor, mag):
    T_para = []
    T_perp = []
    for i in range(len(T_tensor[:, 0])):
        T_arr = np.array(
            [[T_tensor[i, 0], T_tensor[i, 3], T_tensor[i, 4]], [T_tensor[i, 3], T_tensor[i, 1], T_tensor[i, 5]],
             [T_tensor[i, 4], T_tensor[i, 5], T_tensor[i, 2]]])
        mag_tmp = mag[i]
        # print(mag_tmp)
        # print('T_arr: ',T_arr)
        e1_mag = mag_tmp / np.linalg.norm(mag_tmp)
        e2_mag = np.cross(e1_mag, [1, 0, 0])
        e2_mag = e2_mag / np.linalg.norm(e2_mag)
        e3_mag = np.cross(e1_mag, e2_mag)
        E_mag = np.asmatrix(np.array([[e1_mag], [e2_mag], [e3_mag]]).T)
        T_mag = E_mag.T * np.asmatrix(T_arr) * E_mag
        # print(T_mag)
        T_para.append(T_mag[0, 0])
        T_perp.append((T_mag[1, 1] + T_mag[2, 2]) / 2)
    T_para = np.array(T_para)
    T_perp = np.array(T_perp)
    return T_para, T_perp


# %根据磁场矢量，计算平行温度和垂直温度
# T = [Ttensor(1) Ttensor(4) Ttensor(5);Ttensor(4) Ttensor(2) Ttensor(6);Ttensor(5) Ttensor(6) Ttensor(3)];
# E = eye(3);
# e1_mag = mag./norm(mag);
# e2_mag = cross(e1_mag,[1 0 0]);
# e2_mag = e2_mag./norm(e2_mag);
# e3_mag = cross(e1_mag,e2_mag);
# E_mag = [e1_mag' e2_mag' e3_mag'];
#
#          A = E*E_mag;
# T_mag = A'*T*A;
# Tparallel = T_mag(1,1);
# Tperp = (T_mag(2,2)+T_mag(3,3))/2;
# function [epoch, zplus, zminus, dvA, dv] = calc_Z(epochmag, magx, magy, magz, epochspc, vx, vy, vz, np)
# %This function used to calculate Z+ and Z- in one interval using the
# %average data from MAG and SPC
# % The MAG data are interpolated to the time of SPC
# %unit: epoch, B: nT, V: km/s, np: cm-3
# mu_0 = 4e-7*pi;
# m_p = 1.672621637e-27;
# % Bx = interp1(epochmag,magx,epochspc);
# % By = interp1(epochmag,magy,epochspc);
# % Bz = interp1(epochmag,magz,epochspc);
# vx = interp1(epochspc,vx,epochmag,'spline');
# vy = interp1(epochspc,vy,epochmag,'spline');
# vz = interp1(epochspc,vz,epochmag,'spline');
# n0 = mean(np);
# vAx = magx * 1e-9 / sqrt(mu_0 * n0 * 1e6 * m_p) / 1000; %km/s
# vAy = magy * 1e-9 / sqrt(mu_0 * n0 * 1e6 * m_p) / 1000; %km/s
# vAz = magz * 1e-9 / sqrt(mu_0 * n0 * 1e6 * m_p) / 1000; %km/s
# vAx0 = mean(vAx); vAy0 = mean(vAy); vAz0 = mean(vAz);
# vx0 = mean(vx); vy0 = mean(vy); vz0 = mean(vz);
# dBx = vAx-vAx0; dBy = vAy-vAy0; dBz = vAz-vAz0;
# dvx = vx-vx0; dvy = vy - vy0; dvz = vz-vz0;
# dvA = [dBx dBy dBz];
# dv = [dvx dvy dvz];
# epoch = epochmag;
# zpx = dvx - dBx; zpy = dvy - dBy; zpz = dvz - dBz;
# zmx = dvx + dBx; zmy = dvy + dBy; zmz = dvz + dBz;
# zplus = [zpx zpy zpz];
# zminus = [zmx zmy zmz];
# end


if __name__ == '__main__':

    # -----Set parameters-----
    plot_1min = True  # for mag file, True for mag_RTN_1min, False for mag_RTN
    mag_type = '4sa'
    inst = False  # for SPI file, True for instrument coordinates, False for Inertial
    style = 4  # Choose output styles [wait for filling]
    alpha = True

    '''
    Style 1: Plotly Figure. (a) |B|&|V|; (b) Br&Vr; (c) Bt&Vt; (d) Bn&Vn; (e) Np&Tp
    Style 2: Plotly Figure. (a) e-PAD; (b) e-norm_PAD; (c) |B|,Br&Vr; (d) Bt,Bn&beta; (e) Np&Tp
    Style 3: Matplotlib Figure. (a) e-PAD; (b) |B|,Br,Bt,Bn; (c) Vr; (d) Np&Tp; (e) lg(beta)
    Style 4: Matplotlib Figure. (a) |B|&Np; (b) Br&Vr; (c) Bt&Vt; (d) Bn&Vn; (e) Tp&lg(beta); (f) e-PAD; (g)e-norm_PAD
    Style 5: Matplotlib Figure. (a) e-norm_PAD; (b) |B|,Br,Bt,Bn; (c) Vr; (d) Np&Tp; (e) lg(beta)
    '''

    # -----Choose Time range-----
    beg_time = datetime(2021, 4, 28, 15, 20)
    # end_time = datetime(2021,4,28,16,0)
    end_time = datetime(2021, 4, 30, 4, 30)  # 04300430#04281523#04291340
    # beg_time = datetime(2021, 4, 28, 16, 0)
    # end_time = datetime(2021, 4, 28, 16, 10)
    # beg_time = datetime(2021,4,29,13,30)
    # end_time = datetime(2021,4,29,14,30)
    beg_time_str = beg_time.strftime('%Y%m%dT%H%M%S')
    end_time_str = end_time.strftime('%Y%m%dT%H%M%S')

    # ------Load electron PAD and preview-----
    spe_pad = load_spe_data(beg_time.strftime('%Y%m%d'), end_time.strftime('%Y%m%d'))
    epochpade = spe_pad['Epoch']
    timebinpade = (epochpade > beg_time) & (epochpade < end_time)
    epochpade = epochpade[timebinpade]
    EfluxVsPAE = spe_pad['EFLUX_VS_PA_E'][timebinpade, :, :]
    PitchAngle = spe_pad['PITCHANGLE'][timebinpade, :]
    Energy_val = spe_pad['ENERGY_VALS'][timebinpade, :]
    norm_EfluxVsPAE = EfluxVsPAE * 0
    for i in range(12):
        norm_EfluxVsPAE[:, i, :] = EfluxVsPAE[:, i, :] / np.nansum(EfluxVsPAE, 1)  # Calculate normalized PAD
    # choose energy channel
    i_energy = 8
    print(Energy_val[0, i_energy])
    enestr = str(Energy_val[0, i_energy])
    # choose coloraxis. zmin/max1 for PAD; zmin/max2 for norm_PAD
    zmin1 = 8
    zmax1 = 10
    zmin2 = -1.5
    zmax2 = -0
    # # plot preview. skip if all settled.
    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_energy])).T, cmap='jet')
    # plt.colorbar()
    # plt.clim([zmin1, zmax1])
    # plt.subplot(2, 1, 2)
    # plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(norm_EfluxVsPAE[:, :, i_energy])).T, cmap='jet')
    # plt.colorbar()
    # plt.clim([zmin2, zmax2])
    # plt.suptitle(enestr)
    # plt.show()
    # # exit()

    # -----Load mag data-----
    if mag_type == '1min':
        mag_RTN = load_RTN_1min_data(beg_time.strftime('%Y%m%d'), end_time.strftime('%Y%m%d'))

        epochmag = mag_RTN['epoch_mag_RTN_1min']
        timebinmag = (epochmag > beg_time) & (epochmag < end_time)
        epochmag = epochmag[timebinmag]

        Br = mag_RTN['psp_fld_l2_mag_RTN_1min'][timebinmag, 0]
        Bt = mag_RTN['psp_fld_l2_mag_RTN_1min'][timebinmag, 1]
        Bn = mag_RTN['psp_fld_l2_mag_RTN_1min'][timebinmag, 2]
        Babs = np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2)

        filename = 'figures/overviews/Overview_1min(' + beg_time_str + '-' + end_time_str + ').html'
    elif mag_type == 'rtn':

        mag_RTN = load_RTN_data(beg_time.strftime('%Y%m%d%H'), end_time.strftime('%Y%m%d%H'))

        epochmag = mag_RTN['epoch_mag_RTN']
        timebinmag = (epochmag > beg_time) & (epochmag < end_time)
        epochmag = epochmag[timebinmag]

        Br = mag_RTN['psp_fld_l2_mag_RTN'][timebinmag, 0]
        Bt = mag_RTN['psp_fld_l2_mag_RTN'][timebinmag, 1]
        Bn = mag_RTN['psp_fld_l2_mag_RTN'][timebinmag, 2]
        Babs = np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2)

        filename = 'figures/overviews/Overview(' + beg_time_str + '-' + end_time_str + ').html'
    elif mag_type == '4sa':

        mag_RTN = load_RTN_4sa_data(beg_time.strftime('%Y%m%d'), end_time.strftime('%Y%m%d'))

        epochmag = mag_RTN['epoch_mag_RTN_4_Sa_per_Cyc']
        timebinmag = (epochmag > beg_time) & (epochmag < end_time)
        epochmag = epochmag[timebinmag]

        Br = mag_RTN['psp_fld_l2_mag_RTN_4_Sa_per_Cyc'][timebinmag, 0]
        Bt = mag_RTN['psp_fld_l2_mag_RTN_4_Sa_per_Cyc'][timebinmag, 1]
        Bn = mag_RTN['psp_fld_l2_mag_RTN_4_Sa_per_Cyc'][timebinmag, 2]
        Babs = np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2)

    # -----Load SPI data-----
    if inst:
        # load proton moms
        pmom_SPI = load_spi_data(beg_time.strftime('%Y%m%d'), end_time.strftime('%Y%m%d'), inst=inst, species='0')
        epochpmom = pmom_SPI['Epoch']
        timebinpmom = (epochpmom > beg_time) & (epochpmom < end_time)
        epochpmom = epochpmom[timebinpmom]
        densp = pmom_SPI['DENS'][timebinpmom]
        vp_r = pmom_SPI['VEL'][timebinpmom, 0]
        vp_t = pmom_SPI['VEL'][timebinpmom, 1]
        vp_n = pmom_SPI['VEL'][timebinpmom, 2]
        Tp = pmom_SPI['TEMP'][timebinpmom]

        # load alpha moms
        amom_SPI = load_spi_data(beg_time.strftime('%Y%m%d'), end_time.strftime('%Y%m%d'), inst=inst, species='a')
        epochamom = amom_SPI['Epoch']
        timebinamom = (epochamom > beg_time) & (epochamom < end_time)
        epochamom = epochamom[timebinamom]
        densa = amom_SPI['DENS'][timebinamom]
        va_r = amom_SPI['VEL'][timebinamom, 0]
        va_t = amom_SPI['VEL'][timebinamom, 1]
        va_n = amom_SPI['VEL'][timebinamom, 2]
        Ta = amom_SPI['TEMP'][timebinamom]
    else:
        # load proton moms
        pmom_SPI = load_spi_data(beg_time.strftime('%Y%m%d'), end_time.strftime('%Y%m%d'), inst=inst, species='0')
        epochpmom = pmom_SPI['Epoch']
        timebinpmom = (epochpmom > beg_time) & (epochpmom < end_time)
        epochpmom = epochpmom[timebinpmom]
        densp = pmom_SPI['DENS'][timebinpmom]
        vp_r = pmom_SPI['VEL_RTN_SUN'][timebinpmom, 0]
        vp_t = pmom_SPI['VEL_RTN_SUN'][timebinpmom, 1]
        vp_n = pmom_SPI['VEL_RTN_SUN'][timebinpmom, 2]
        Tp = pmom_SPI['TEMP'][timebinpmom]
        EFLUX_VS_PHI_p = pmom_SPI['EFLUX_VS_PHI'][timebinpmom]
        PHI_p = pmom_SPI['PHI_VALS'][timebinpmom]
        T_tensor_p = pmom_SPI['T_TENSOR_INST'][timebinpmom]
        MagF_inst_p = pmom_SPI['MAGF_INST'][timebinpmom]
        print(np.array(epochpmom).shape)
        print(np.array(T_tensor_p).shape)
        print(np.array(MagF_inst_p).shape)

        # load alpha moms
        amom_SPI = load_spi_data(beg_time.strftime('%Y%m%d'), end_time.strftime('%Y%m%d'), inst=inst, species='a')
        epochamom = amom_SPI['Epoch']
        timebinamom = (epochamom > beg_time) & (epochamom < end_time)
        epochamom = epochamom[timebinamom]
        densa = amom_SPI['DENS'][timebinamom]
        va_r = amom_SPI['VEL_RTN_SUN'][timebinamom, 0]
        va_t = amom_SPI['VEL_RTN_SUN'][timebinamom, 1]
        va_n = amom_SPI['VEL_RTN_SUN'][timebinamom, 2]
        Ta = amom_SPI['TEMP'][timebinamom]
        EFLUX_VS_PHI_a = amom_SPI['EFLUX_VS_PHI'][timebinamom]
        PHI_a = amom_SPI['PHI_VALS'][timebinamom]
        T_tensor_a = amom_SPI['T_TENSOR_INST'][timebinpmom]
        MagF_inst_a = amom_SPI['MAGF_INST'][timebinpmom]

    # print(np.argmax(EFLUX_VS_PHI_a,axis=1))
    test_mom_data = False
    if test_mom_data:
        plt.subplot(2, 1, 1)
        plt.pcolormesh(epochpmom, np.array(PHI_p[0][:]), np.array(EFLUX_VS_PHI_p).T)
        plt.colorbar()
        plt.clim(1e11, 5e12)
        plt.ylabel('PHI')
        plt.title('EFLUX vs PHI (proton)')

        plt.subplot(2, 1, 2)
        plt.pcolormesh(epochamom, np.array(PHI_a[0][:]), np.log10(np.array(EFLUX_VS_PHI_a).T))
        # plt.plot(epochamom,np.array(PHI_a[:,np.argmax(EFLUX_VS_PHI_a,axis=1)]))
        plt.colorbar()
        # plt.clim(1e9,1e11)
        plt.clim(9, 10.5)
        plt.ylabel('PHI')
        plt.xlabel('Time')
        plt.title('EFLUX vs PHI (alpha)')
        plt.show()

    from plot_body_positions import get_rlonlat_psp_carr

    r_psp_carr_pmom, lon_psp_carr_pmom, lat_psp_carr_pmom = get_rlonlat_psp_carr(epochamom, for_psi=False)
    r_psp_carr_mag, lon_psp_carr_mag, lat_psp_carr_mag = get_rlonlat_psp_carr(epochmag, for_psi=False)

    # plt.figure()
    # transit_t = 4 / 3 * (r_psp_carr_pmom - 1) * 696300 / vp_r / 86400
    # plt.plot(epochpmom, transit_t)
    # plt.xlabel('Epoch')
    # plt.ylabel('Transition Time [day]')
    # plt.show()

    densp = densp[0:len(densa)]
    epochpmom = epochpmom[0:len(densa)]
    Tp = Tp[0:len(densa)]
    vp_r = vp_r[0:len(densa)]
    vp_t = vp_t[0:len(densa)]
    vp_n = vp_n[0:len(densa)]

    # ----- calculate secondary properties -----
    # interp total magnetic field to pmoms epoch and calculate beta.
    Babs_p = np.interp(np.array((epochpmom - epochpmom[0]) / timedelta(days=1), dtype='float64'),
                       np.array((epochmag - epochpmom[0]) / timedelta(days=1), dtype='float64'), Babs)
    Br_p = np.interp(np.array((epochpmom - epochpmom[0]) / timedelta(days=1), dtype='float64'),
                     np.array((epochmag - epochpmom[0]) / timedelta(days=1), dtype='float64'), Br)
    Bn_p = np.interp(np.array((epochpmom - epochpmom[0]) / timedelta(days=1), dtype='float64'),
                     np.array((epochmag - epochpmom[0]) / timedelta(days=1), dtype='float64'), Bn)
    # Bt_p = np.interp(np.array((epochpmom - epochpmom[0]) / timedelta(days=1), dtype='float64'),
    #                  np.array((epochmag - epochpmom[0]) / timedelta(days=1), dtype='float64'), Bt)

    Tp_para, Tp_perp = calc_anisotropic_temp(T_tensor_p, MagF_inst_p)
    Ta_para, Ta_perp = calc_anisotropic_temp(T_tensor_a, MagF_inst_a)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(epochpmom, Tp_para)
    plt.plot(epochpmom, Tp_perp)
    ax = plt.twinx()
    ax.plot(epochpmom, Tp_para / Tp_perp, 'k-')
    plt.subplot(2, 1, 2)
    plt.plot(epochamom, Ta_para)
    plt.plot(epochamom, Ta_perp)
    ax = plt.twinx()
    ax.plot(epochamom, Ta_para / Ta_perp, 'k-')
    ax.set_ylim([0, 10])
    plt.show()
    # quit()

    beta = 4.03e-11 * densp * Tp / ((Babs_p * 1e-5) ** 2)
    # plot thermal pressure P_t and magnetic pressure P_b
    e = 1.6022e-19
    P_t = densp * Tp * 1e6 * e * 1e9
    dP_t = P_t - np.mean(P_t)
    P_b = 3.93 * (Babs_p * 1e-9) ** 2 * 1.0133e5 * 1e9
    dP_b = P_b - np.mean(P_b)
    Valf = 2.18e1 * Babs_p * densp ** (-0.5)  # km/s
    Vap = np.sqrt((va_r - vp_r) ** 2 + (va_n - vp_n) ** 2 + (va_t - vp_t) ** 2)
    Vap = np.sqrt((va_r) ** 2 + (va_n) ** 2 + (va_t) ** 2) - np.sqrt((vp_r) ** 2 + (vp_n) ** 2 + (vp_t) ** 2)
    # print(Valf)
    # print(Vap)
    mp = 1.6726e-27
    ma = mp * 4
    boltzmann = 1.3807e-23
    tc = 133 * (np.sqrt(2 * Ta * e / ma + 2 * Tp * e / mp) * 1e-3) ** 3 / densp  # s
    Ac_ap = r_psp_carr_pmom / (np.sqrt((vp_r) ** 2 + (vp_n) ** 2 + (vp_t) ** 2) * tc) * 696300

    lambda_p = 9.42 + np.log((Tp * 1.1604e4) ** (3 / 2) / densp ** (1 / 2))
    Ac_pp = 1.31e7 * densp / (vp_r * (Tp * 1.1604e4) ** (3 / 2)) * r_psp_carr_pmom * 6.96e10 / 1.5e13 * lambda_p

    plt.figure()
    plt.plot(epochpmom, Ac_ap, label='Ac_alpha-proton')
    plt.plot(epochpmom, Ac_pp, label='Ac_proton-proton')
    plt.title('Ac ap/pp')
    plt.legend()
    plt.show()
    Ac = Ac_pp
    windowsecs = 60 * 20
    # plot helicity
    # helicity, norm_dB = calc_helicity(epochmag,Br,Bt,Bn,epochpmom,vp_r,vp_t,vp_n,densp,windowsecs=windowsecs)
    # np.savez('helicity_normdB'+str(windowsecs),epochpmom=epochpmom,helicity=helicity,windowsecs=windowsecs,norm_dB=norm_dB)
    with np.load('helicity_normdB' + str(windowsecs) + '.npz') as f:
        helicity, norm_dB = f['helicity'], f['norm_dB']
    np.load('helicity_normdB' + str(windowsecs) + '.npz')

    fig, axs = plt.subplots(4, 1, sharex=True)
    axs[0].plot(epochpmom, helicity, 'k-')
    axs[0].set_ylabel(r'$\sigma_c$')
    ax2 = axs[0].twinx()
    ax2.plot(epochpmom, norm_dB, 'r-')
    ax2.set_ylabel(r'$|\delta V_A|$', c='r')

    axs[1].plot(epochmag, Br, 'k-')
    axs[1].set_ylabel('$B_r [nT]$')
    ax2 = axs[1].twinx()
    ax2.plot(epochpmom, vp_r, 'r-')
    ax2.set_ylabel(r'$V_r [km/s]$', c='r')
    axs[1].tick_params(axis="x", which='both', direction="in", pad=-15)

    axs[2].plot(epochmag, Bt, 'k-')
    axs[2].set_ylabel(r'$B_t [nT]$')
    ax2 = axs[2].twinx()
    ax2.plot(epochpmom, vp_t, 'r-')
    ax2.set_ylabel(r'$V_t [km/s]$', c='r')
    axs[2].tick_params(axis="x", which='both', direction="in", pad=-15)

    axs[3].plot(epochmag, Bn, 'k-')
    axs[3].set_ylabel(r'$B_n [nT]$')
    ax2 = axs[3].twinx()
    ax2.plot(epochpmom, vp_n, 'r-')
    ax2.set_ylabel(r'$V_n [km/s]$', c='r')
    axs[3].tick_params(axis="x", which='both', direction="out", pad=-15)

    plt.show()
    # quit()

    # ----- plot previews -----
    # plot preview of proton/alpha moms
    if alpha:
        # plt.figure()
        # plt.subplot(3, 1, 1)
        # plt.plot(vp_r, 'k-')
        # plt.plot(va_r, 'b-')
        # plt.subplot(3, 1, 2)
        # plt.plot(Vap)
        # plt.subplot(3, 1, 3)
        # plt.plot(Vap / Valf)
        # plt.show()
        fig, axs = plt.subplots(4, 1, sharex=True)
        # plt.subplot(4, 1, 1)
        axs[0].plot(epochpmom, densp, 'k-', label=r'$Np$')
        axs[0].set_ylabel('$N_p$\n$[cm^-3]$')
        ax2 = axs[0].twinx()
        ax2.scatter(epochpmom, 100 * densa / densp, c='r', s=0.5, label=r'$AHe$')
        ax2.set_ylim([0, 3])
        ax2.set_ylabel(r'$A_{\alpha} [\%]$', c='r')
        axs[0].xaxis.set_minor_locator(AutoMinorLocator())
        axs[0].tick_params(axis="x", which='both', direction="in", pad=-15)
        # plt.subplot(4, 1, 2)
        axs[1].plot(epochpmom, Tp, 'k-', label=r'$Tp$')
        axs[1].set_ylabel('$T_p$\n$[eV]$')
        ax2 = axs[1].twinx()
        ax2.scatter(epochpmom, Ta / Tp, label=r'$Tα$', c='r', s=0.5)
        ax2.set_ylim([0, 20])
        ax2.set_ylabel(r'$T_{\alpha}/T_p$', c='r')
        axs[1].xaxis.set_minor_locator(AutoMinorLocator())
        axs[1].tick_params(axis="x", which='both', direction="in", pad=-15)
        # plt.subplot(4, 1, 3)
        axs[2].plot(epochpmom, vp_r, 'k-', label='Vpr')
        axs[2].set_ylabel('$V_{pr}$\n $[km/s]$')
        ax2 = axs[2].twinx()
        ax2.scatter(epochpmom, Vap / Valf, label='Vap/VA', c='r', s=0.5)
        ax2.set_ylabel(r'$V_{\alpha p}/V_A$', c='r')
        ax2.set_ylim([-1, 2])
        axs[2].xaxis.set_minor_locator(AutoMinorLocator())
        axs[2].tick_params(axis="x", which='both', direction="in", pad=-15)
        # plt.subplot(4, 1, 4)
        axs[3].plot(epochpmom, Ac, 'k-')
        axs[3].set_ylabel('Collisional Age')
        axs[3].set_xlabel('Time [mm-dd HH]')
        plt.show()
        # exit()

        plt.figure()
        plt.plot(epochpmom, vp_r * 1e5 * densp - ma * 1e3 * 6.6726e-8 * 2e33 * (1e6) ** 1.5 / (
                1.2e-23 * (r_psp_carr_pmom * 696300 * 1e5) ** 2))
        plt.show()

        # plt.figure()
        # plt.subplot(221)
        # plt.hist2d(np.log10(100 * densa / densp), Ta / Tp, bins=100, range=[[-2, 1], [0, 15]], density=True,
        #            norm=colors.LogNorm(), cmap='jet')
        # # plt.xscale('log')
        # plt.xlabel('AHe [%]')
        # plt.ylabel('Ta/Tp')
        # # plt.axis('square')
        # plt.colorbar()
        # plt.subplot(223)
        # plt.hist2d(np.log10(100 * densa / densp), vp_r, bins=100, range=[[-2, 1], [100, 400]], density=True,
        #            norm=colors.LogNorm(), cmap='jet')
        # plt.ylabel(r'$\Delta V/V_{A}$')
        # plt.xlabel('AHe')
        # plt.colorbar()
        # plt.subplot(222)
        # plt.hist2d(Vap / Valf, Ta / Tp, bins=100, range=[[0, 1.6], [0, 15]], density=True, norm=colors.LogNorm(),
        #            cmap='jet')
        # plt.xlabel(r'$\Delta V/V_{A}$')
        # plt.ylabel('Ta/Tp')
        # plt.colorbar()
        # plt.suptitle('Frequency')
        # plt.show()
        import pandas as pd

        # df = pd.DataFrame()
        # df['1/AHe'] = densp/densa/100
        # df['Tp/Ta'] = Tp/Ta
        # df['DVrVa'] = Vap/Valf
        # df['Ac'] = Ac
        # df['origins'] = [t > datetime(2021,4,29,13,40) for t in epochpmom]
        # # df['origins'] = [True for t in epochpmom]
        # # df['deltaV'] = va_r-vp_r
        # # df['Babs'] = Babs_p
        # # df['Br'] = Br_p
        # g = sns.pairplot(df,hue='origins',markers='x')
        # g.axes[0,0].set_xlim([0,50])
        # g.axes[0,0].set_ylim([0,50])
        # g.axes[1,1].set_xlim([1/20,10])
        # g.axes[1,1].set_ylim([1/20,10])
        # g.axes[2,2].set_xlim([0,3])
        # g.axes[2,2].set_ylim([0,3])
        # # g = sns.pairplot(df,kind=')
        # #
        #
        #
        # plt.savefig('pairplotalphafan.png')

        HCS_timebin = ((epochpmom > datetime(2021, 4, 29, 0, 55)) & (epochpmom < datetime(2021, 4, 29, 1, 55))) | \
                      ((epochpmom > datetime(2021, 4, 29, 8, 15)) & (epochpmom < datetime(2021, 4, 29, 8, 55))) | \
                      ((epochpmom > datetime(2021, 4, 29, 9, 23)) & (epochpmom < datetime(2021, 4, 29, 10, 23))) | \
                      ((epochpmom > datetime(2021, 4, 29, 13, 40)) & (epochpmom < datetime(2021, 4, 29, 13, 50)))
        SSW1_timebin = (epochpmom < datetime(2021, 4, 29, 13, 40)) & ~HCS_timebin
        SSW2_timebin = (epochpmom > datetime(2021, 4, 29, 13, 40)) & ~HCS_timebin

        df = pd.DataFrame()
        df['epoch'] = epochamom
        df[r'A_{\alpha}'] = 100 * densa / densp
        df[r'$T_{\alpha}/T_{p}$'] = Ta / Tp
        df[r'$dV_{\alpha p}/V_{A}$'] = Vap / Valf
        df['Ac'] = Ac
        df['type'] = 'SSW1'
        df['type'][HCS_timebin] = 'HCS'
        df['type'][SSW2_timebin] = 'SSW2'
        df['Proton Flux'] = vp_r * densp
        df['$V_pr$'] = vp_r
        df['$\sigma_c$'] = np.abs(helicity)
        df['$|\delta V_{A}|$'] = norm_dB
        df['Tp_para/perp'] = Tp_para / Tp_perp
        df['Ta_para/perp'] = Ta_para / Ta_perp
        df.to_csv('alpha_properties.csv')
        # df['origins'] = [True for t in epochpmom]
        # df['deltaV'] = va_r-vp_r
        # df['Babs'] = Babs_p
        # df['Br'] = Br_p
        # fig,axes=plt.subplots(2,4)
        # plt.subplot(4,2,1)
        # sns.displot(df, x=r'A_{\alpha}', hue="type", kind="kde", fill=True, row_order=0, col_order=0)
        # # plt.subplot(4,2,2)
        # sns.displot(df, x=r'$T_{\alpha}/T_{p}$', hue="type", kind="kde", fill=True, row_order=0, col_order=1)
        # # plt.subplot(4,2,3)
        # sns.displot(df, x=r'$dV_{\alpha p}/V_{A}$', hue="type", kind="kde", fill=True, row_order=0, col_order=2)
        # # plt.subplot(4,2,4)
        # sns.displot(df, x='Ac', hue="type", kind="kde", fill=True, row_order=0, col_order=3)
        # # plt.subplot(4,2,5)
        # sns.displot(df, x='$V_pr$', hue="type", kind="kde", fill=True, row_order=1, col_order=0)
        # # plt.subplot(4,2,6)
        # sns.displot(df, x='Proton Flux', hue="type", kind="kde", fill=True, row_order=1, col_order=1)
        # # plt.subplot(4,2,7)
        # sns.displot(df, x='$\sigma_c$', hue="type", kind="kde", fill=True, row_order=1, col_order=2)
        # # plt.subplot(4,2,8)
        # sns.displot(df, x='$|\delta V_{A}|$', hue="type", kind="kde", fill=True, row_order=1, col_order=3)
        # # plt.show()
        #
        # g = sns.pairplot(df, hue='type', markers='x')
        # # g = sns.pairplot(df,kind=')
        # g.axes[0, 0].set_xlim([0, 3])
        # g.axes[0, 0].set_ylim([0, 3])
        # g.axes[1, 1].set_xlim([0, 18])
        # g.axes[1, 1].set_ylim([0, 18])
        # g.axes[2, 2].set_xlim([-1, 3.5])
        # g.axes[2, 2].set_ylim([-1, 3.5])
        # # g.axes[4,4].set_xlim([100,400])
        # # g.axes[4,4].set_ylim([100,400])
        # g.axes[5, 5].set_xlim([0, 1])
        # g.axes[5, 5].set_ylim([0, 1])
        # g.axes[6, 6].set_xlim([0, 1500])
        # g.axes[6, 6].set_ylim([0, 1500])
        #
        # plt.savefig('pairplotalpha.png')

    # plt.figure()
    # plt.scatter(np.log10(100*densa/densp),Ta/Tp,c=abs(Bn_p),cmap='jet',s=1)
    # plt.show()

    # plt.figure()
    # plt.subplot(221)
    # plt.hist2d(Vap/Valf,np.log10(100*densa/densp),bins=100,range=[[0,2],[-2,2]],density=True,norm=colors.LogNorm(),cmap='jet')
    # plt.xlabel(r'$\Delta V/V_{A}$')
    # plt.ylabel('AHe')
    # plt.colorbar()
    # plt.subplot(222)
    # plt.hist2d(Vap/Valf,Ta/Tp,bins=100,range=[[0,2],[0,20]],density=True,norm=colors.LogNorm(),cmap='jet')
    # plt.xlabel(r'$\Delta V/V_{A}$')
    # plt.ylabel('Ta/Tp')
    # plt.colorbar()
    # # plt.subplot(222)
    # # plt.hist2d(r_psp_carr_pmom,np.log10(100*densa/densp),bins=100,range=[[15,50],[-2,2]],density=True,norm=colors.LogNorm(),cmap='jet')
    # # plt.xlabel('R')
    # # plt.ylabel('AHe')
    # # plt.colorbar()
    # plt.subplot(223)
    # plt.hist2d(Babs_p,np.log10(100*densa/densp),bins=100,range=[[0,500],[-2,2]],density=True,norm=colors.LogNorm(),cmap='jet')
    # plt.xlabel('Br')
    # plt.ylabel('AHe')
    # plt.colorbar()
    # plt.subplot(224)
    # plt.hist2d(np.rad2deg(lat_psp_carr_pmom),np.log10(100*densa/densp),bins=100,range=[[-4.5,4.5],[-2,2]],density=True,norm=colors.LogNorm(),cmap='jet')
    # plt.xlabel('Carr_lat')
    # plt.ylabel('AHe')
    # plt.colorbar()
    # plt.suptitle('Frequency')
    # plt.show()

    # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # c=ax.scatter(lat_psp_carr_pmom, r_psp_carr_pmom,c=np.array(100*densa/densp),s=1,cmap='jet',vmin=0.01,vmax=10,norm=colors.LogNorm())
    # fig.colorbar(c)
    # fig.show()

    # # plot preview of beta and scaled B/Np.
    # # get r_psp from spice for scaling. self-made function. skip.
    # from plot_body_positions import get_rlonlat_psp_carr
    # r_psp_carr_pmom, _, _ = get_rlonlat_psp_carr(epochpmom, for_psi=False)
    # r_psp_carr_mag, _, _ = get_rlonlat_psp_carr(epochmag, for_psi=False)
    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.plot(epochpmom, vp_r, label=r'$V_r$')
    # plt.legend()
    # ax2 = plt.twinx()
    # ax2.plot(epochpmom, densp * r_psp_carr_pmom ** 2, 'r-', label=r'$Np r^2$')
    # plt.legend()
    # plt.subplot(2, 1, 2)
    # plt.plot(epochpmom, Babs_p * r_psp_carr_pmom ** 2, label=r'$|B|r^2$')
    # plt.legend()
    # ax2 = plt.twinx()
    # ax2.plot(epochpmom, np.log10(beta), 'r-', label=r'$lg(\beta)$')
    # plt.legend()
    # plt.show()

    # # plot preview of P_t/P-b, check for slow/fase mode.
    # plt.figure()
    # plt.plot(epochpmom, dP_t, 'r-', label=r'\delta P_t')
    # plt.plot(epochpmom, dP_b, 'b-', label=r'\delta P_b')
    # plt.legend()
    # plt.show()

    if style == 1:
        '''Style 1: Plotly Figure. (a) |B|&|V|; (b) Br&Vr; (c) Bt&Vt; (d) Bn&Vn; (e) Np&Tp'''
        fig = make_subplots(rows=5, cols=1,
                            specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}],
                                   [{"secondary_y": True}], [{"secondary_y": True}]],
                            subplot_titles=("B_R & V_R", "B_T & V_T", "B_N & V_N"), shared_xaxes=True)
        fig.add_trace(
            go.Scatter(x=epochmag, y=np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2), name='|B|', mode='lines', line_color='blue'),
            row=1, col=1)
        fig.add_trace(go.Scatter(x=epochmag, y=Br, name='Br', mode='lines', line_color='blue'), row=2, col=1)
        fig.add_trace(go.Scatter(x=epochmag, y=Bt, name='Bt', mode='lines', line_color='blue'), row=3, col=1)
        fig.add_trace(go.Scatter(x=epochmag, y=Bn, name='Bn', mode='lines', line_color='blue'), row=4, col=1)
        fig.add_trace(go.Scatter(x=epochpmom, y=densp, name='np', mode='lines', line_color='blue'), row=5, col=1)

        fig.add_trace(go.Scatter(x=epochpmom, y=np.sqrt(vp_r ** 2 + vp_n ** 2 + vp_t ** 2), name='|V|', mode='lines',
                                 line_color='red'), row=1, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=epochpmom, y=vp_r, name='Vr', mode='lines', line_color='red'), row=2, col=1,
                      secondary_y=True)
        fig.add_trace(go.Scatter(x=epochpmom, y=vp_t, name='Vt', mode='lines', line_color='red'), row=3, col=1,
                      secondary_y=True)
        fig.add_trace(go.Scatter(x=epochpmom, y=vp_n, name='Vn', mode='lines', line_color='red'), row=4, col=1,
                      secondary_y=True)
        fig.add_trace(go.Scatter(x=epochpmom, y=Tp, name='Tp', mode='lines', line_color='red'), row=5, col=1,
                      secondary_y=True)

        fig.update_xaxes(tickformat="%m/%d %H:%M\n%Y", title_text="Epoch")
        fig.update_yaxes(title_text="|B| [nT]", row=1, col=1)
        fig.update_yaxes(title_text="Br [nT]", row=2, col=1)
        fig.update_yaxes(title_text="Bt [nT]", row=3, col=1)
        fig.update_yaxes(title_text="Bn [nT]", row=4, col=1)
        fig.update_yaxes(title_text="np", row=5, col=1)
        fig.update_yaxes(title_text="|V| [km/s]", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Vr [km/s]", row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Vt [km/s]", row=3, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Vn [km/s]", row=4, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Tp [km/s]", row=5, col=1, secondary_y=True)

        fig.update_layout(title=dict(text="Magnetic Field",
                                     y=0.9, x=0.5,
                                     xanchor='center',
                                     yanchor='top'),
                          template='simple_white', )
        filename = 'figures/overviews/Overview(' + beg_time_str + '-' + end_time_str + ').html'
        if plot_1min:
            filename = 'figures/overviews/Overview_1min(' + beg_time_str + '-' + end_time_str + ').html'
        py.plot(fig, filename='figures/overviews/Overview(' + beg_time_str + '-' + end_time_str + ').html')
    if style == 2:
        '''Style 2: Plotly Figure. (a) e-PAD; (b) e-norm_PAD; (c) |B|,Br&Vr; (d) Bt,Bn&beta; (e) Np&Tp'''
        fig = make_subplots(rows=5, cols=1,
                            specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}],
                                   [{"secondary_y": True}], [{"secondary_y": True}]],
                            subplot_titles=(
                                "PAD " + enestr + "eV", "Normalized PAD " + enestr + "eV", "|B|, B_R & V_R", "Bt, Bn",
                                "np & Tp"), shared_xaxes=True)

        fig.add_trace(
            go.Heatmap(x=epochpade, y=PitchAngle[0, :], z=np.log10(EfluxVsPAE[:, :, i_energy].T), colorscale='jet',
                       zmin=zmin1, zmax=zmax1, showscale=False), row=1, col=1)

        fig.add_trace(
            go.Heatmap(x=epochpade, y=PitchAngle[0, :], z=np.log10(norm_EfluxVsPAE[:, :, i_energy].T), colorscale='jet',
                       zmin=zmin2, zmax=zmax2, showscale=False), row=2, col=1)

        fig.add_trace(go.Scatter(x=epochmag, y=np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2), name='|B|', mode='lines',
                                 line_color='black'), row=3, col=1)
        fig.add_trace(go.Scatter(x=epochmag, y=Br, name='Br', mode='lines', line_color='blue'), row=3, col=1)
        fig.add_trace(go.Scatter(x=epochpmom, y=vp_r, name='Vr', mode='lines', line_color='red'), row=3, col=1,
                      secondary_y=True)

        fig.add_trace(go.Scatter(x=epochmag, y=Bt, name='Bt', mode='lines', line_color='green'), row=4, col=1)
        fig.add_trace(go.Scatter(x=epochmag, y=Bn, name='Bn', mode='lines', line_color='red'), row=4, col=1)
        fig.add_trace(go.Scatter(x=epochpmom, y=np.log10(beta), name=r'\beta', mode='lines', line_color='black'), row=4,
                      col=1, secondary_y=True)

        fig.add_trace(go.Scatter(x=epochpmom, y=densp, name='np', mode='lines', line_color='blue'), row=5, col=1)
        fig.add_trace(go.Scatter(x=epochpmom, y=Tp, name='Tp', mode='lines', line_color='red'), row=5, col=1,
                      secondary_y=True)

        fig.update_xaxes(tickformat="%m/%d %H:%M\n%Y", title_text="Epoch", row=5, col=1)
        fig.update_yaxes(title_text="PA", row=1, col=1)
        fig.update_yaxes(title_text="PA", row=2, col=1)
        fig.update_yaxes(title_text="B [nT]", row=3, col=1)
        fig.update_yaxes(title_text="Btn [nT]", row=4, col=1)
        fig.update_yaxes(title_text=r"\beta", row=4, col=1, secondary_y=True)
        fig.update_yaxes(title_text="np", row=5, col=1)
        fig.update_yaxes(title_text="Vr [km/s]", row=3, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Tp", row=5, col=1, secondary_y=True)

        fig.update_layout(title=dict(text="HCS Crossing Encounter 08",
                                     y=0.9, x=0.5,
                                     xanchor='center',
                                     yanchor='top'),
                          template='simple_white', )
        filename = 'figures/overviews/Overview_PAD_(' + beg_time_str + '-' + end_time_str + ').html'
        if plot_1min:
            filename = 'figures/overviews/Overview_PAD' + enestr + '_1min(' + beg_time_str + '-' + end_time_str + ').html'
        py.plot(fig, filename=filename)
    if style == 3:
        '''Style 3: Matplotlib Figure. (a) e-PAD; (b) |B|,Br,Bt,Bn; (c) Vr; (d) Np&Tp; (e) lg(beta)'''
        fig, axs = plt.subplots(5, 1, sharex=True)

        pos = axs[0].pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_energy])).T,
                                cmap='jet', vmax=zmax1, vmin=zmin1)
        axs[0].set_ylabel('Pitch Angle \n[deg]')
        axs[0].xaxis.set_minor_locator(AutoMinorLocator())
        axs[0].tick_params(axis="x", which='both', direction="in", pad=-15)

        axs[1].plot(epochmag, Bt, 'r-')
        axs[1].plot(epochmag, Bn, 'b-')
        axs[1].plot(epochmag, np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2), 'm-')
        axs[1].plot(epochmag, Br, 'k-')
        axs[1].set_ylabel('B\n[nT]')
        axs[1].xaxis.set_minor_locator(AutoMinorLocator())
        axs[1].tick_params(axis="x", which='both', direction="in", pad=-15)

        axs[2].plot(epochpmom, vp_r, 'k-')
        axs[2].set_ylabel('$V_r$\n$[km/s]$')
        axs[2].xaxis.set_minor_locator(AutoMinorLocator())
        axs[2].tick_params(axis="x", which='both', direction="in", pad=-15)

        axs[3].set_ylabel('$N_p$\n$[cm^{-3}]$')
        axs[3].xaxis.set_minor_locator(AutoMinorLocator())
        axs[3].tick_params(axis="x", which='both', direction="in", pad=-15)
        ax2 = axs[3].twinx()
        ax2.plot(epochpmom, Tp, 'r-')
        ax2.set_ylim([0, 240])
        ax2.set_ylabel('$T_p$\n$[eV]$', color='r')
        axs[3].plot(epochpmom, densp, 'k-')

        axs[4].plot(epochpmom, np.log10(beta), 'k-')
        axs[4].set_ylabel(r'$\lg (\beta)$')
        axs[4].xaxis.set_minor_locator(AutoMinorLocator())
        axs[4].set_xlabel('Time [mm-dd HH]')

        plt.show()
    if style == 4:
        '''Style 4: Matplotlib Figure. (a) |B|&Np; (b) Br&Vr; (c) Bt&Vt; (d) Bn&Vn; (e) Tp&lg(beta); (f) e-PAD; (g)e-norm_PAD'''
        fig, axs = plt.subplots(7, 1, sharex=True)

        axs[0].plot(epochmag, Babs, 'k-')
        axs[0].set_ylabel('$|B| [nT]$')
        ax2 = axs[0].twinx()
        ax2.plot(epochpmom, densp, 'r-')
        ax2.set_ylabel(r'$N_p [cm^{-3}]$', c='r')
        axs[0].tick_params(axis="x", which='both', direction="in", pad=-15)

        axs[1].plot(epochmag, Br, 'k-')
        axs[1].set_ylabel('$B_r [nT]$')
        ax2 = axs[1].twinx()
        ax2.plot(epochpmom, vp_r, 'r-')
        ax2.set_ylabel(r'$V_r [km/s]$', c='r')
        axs[1].tick_params(axis="x", which='both', direction="in", pad=-15)

        axs[2].plot(epochmag, Bt, 'k-')
        axs[2].set_ylabel(r'$B_t [nT]$')
        ax2 = axs[2].twinx()
        ax2.plot(epochpmom, vp_t, 'r-')
        ax2.set_ylabel(r'$V_t [km/s]$', c='r')
        axs[2].tick_params(axis="x", which='both', direction="in", pad=-15)

        axs[3].plot(epochmag, Bn, 'k-')
        axs[3].set_ylabel(r'$B_n [nT]$')
        ax2 = axs[3].twinx()
        ax2.plot(epochpmom, vp_n, 'r-')
        ax2.set_ylabel(r'$V_n [km/s]$', c='r')
        axs[3].tick_params(axis="x", which='both', direction="in", pad=-15)

        axs[4].plot(epochpmom, Tp, 'k-')
        axs[4].set_ylabel(r'$T_p [eV]$')
        ax2 = axs[4].twinx()
        ax2.plot(epochpmom, np.log10(beta), 'r-')
        ax2.set_ylabel(r'$\lg (\beta)$', c='r')
        axs[4].tick_params(axis="x", which='both', direction="in", pad=-15)

        pos = axs[5].pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(norm_EfluxVsPAE[:, :, i_energy])).T,
                                cmap='jet', vmax=zmax2, vmin=zmin2)
        axs[5].set_ylabel('Pitch Angle \n[deg]')
        axs[5].xaxis.set_minor_locator(AutoMinorLocator())
        axs[5].tick_params(axis="x", which='both', direction="in", pad=-15)

        pos = axs[6].pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_energy])).T,
                                cmap='jet', vmax=zmax1, vmin=zmin1)
        axs[6].set_ylabel('Pitch Angle \n[deg]')
        axs[6].xaxis.set_minor_locator(AutoMinorLocator())
        axs[6].set_xlabel('Time [mm-dd HH]')
        plt.show()
    if style == 5:
        '''Style 5: Matplotlib Figure. (a) e-norm_PAD; (b) |B|,Br,Bt,Bn; (c) Vr; (d) Np&Tp; (e) lg(beta)'''
        fig, axs = plt.subplots(5, 1, sharex=True)
        pos = axs[0].pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_energy])).T,
                                cmap='jet', vmax=zmax1, vmin=zmin1)
        axs[0].set_ylabel('Pitch Angle \n[deg]')
        axs[0].xaxis.set_minor_locator(AutoMinorLocator())
        # axs[0].title(enestr)
        axs[0].tick_params(axis="x", which='both', direction="in", pad=-15)

        axs[1].plot(epochmag, Bt, 'r-', label='Bt')
        axs[1].plot(epochmag, Bn, 'b-', label='Bn')
        axs[1].plot(epochmag, np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2), 'm-', label='|B|')
        axs[1].plot(epochmag, Br, 'k-', label='Br')
        axs[1].set_ylabel('B\n[nT]')
        # axs[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        #               ncol=2, mode="expand", borderaxespad=0.)
        axs[1].legend(loc='right')
        axs[1].xaxis.set_minor_locator(AutoMinorLocator())
        axs[1].tick_params(axis="x", which='both', direction="in", pad=-15)

        axs[2].plot(epochpmom, vp_r, 'k-')
        axs[2].set_ylabel('$V$\n$[km/s]$')
        axs[2].xaxis.set_minor_locator(AutoMinorLocator())
        axs[2].tick_params(axis="x", which='both', direction="in", pad=-15)

        axs[3].plot(epochpmom, densp, 'k-')
        axs[3].set_ylabel('$N_p$\n$[cm^{-3}]$')
        axs[3].xaxis.set_minor_locator(AutoMinorLocator())
        axs[3].tick_params(axis="x", which='both', direction="in", pad=-15)
        ax2 = axs[3].twinx()
        ax2.plot(epochpmom, Tp, 'r-')
        ax2.set_ylim([0, 500])
        ax2.set_ylabel('$T_p$\n$[eV]$', color='r')

        axs[4].plot(epochpmom, np.log10(beta), 'k-')
        axs[4].set_ylabel(r'$\lg (\beta)$')
        axs[4].xaxis.set_minor_locator(AutoMinorLocator())
        axs[4].set_xlabel('Time')
        # ax2 = axs[4].twinx()
        # ax2.plot(epochpmom,densa*100/densp,'r-')
        # ax2.set_ylim([0,2])
        # ax2.set_ylabel(r'$A_{\alpha}%$',color='r')

        plt.show()
    if style == 6:
        # '''Style 5: Matplotlib Figure. (a) e-norm_PAD; (b) |B|,Br,Bt,Bn; (c) Vr; (d) Np&Tp; (e) lg(beta)'''
        fig, axs = plt.subplots(6, 1, sharex=True)

        axs[0].plot(epochpmom, r_psp_carr_pmom, 'k-', linewidth=1)
        ax2 = axs[0].twinx()
        ax2.plot(epochpmom, np.rad2deg(lon_psp_carr_pmom), 'r-', linewidth=1)
        ax2.set_ylabel('Carrington\n Longitude [deg]')
        axs[0].set_ylabel('Radial\n Distance [Rs]')

        pos = axs[1].pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(norm_EfluxVsPAE[:, :, i_energy])).T,
                                cmap='jet', vmax=zmax2, vmin=zmin2)
        axs[1].set_ylabel('Pitch Angle \n[deg]')
        axs[1].xaxis.set_minor_locator(AutoMinorLocator())
        axs[1].tick_params(axis="x", which='both', direction="in", pad=-15)

        axs[2].plot(epochmag, Bt, 'r-', linewidth=.5)
        axs[2].plot(epochmag, Bn, 'b-', linewidth=.5)
        axs[2].plot(epochmag, np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2), 'm-', linewidth=1)
        axs[2].plot(epochmag, Br, 'k-', linewidth=1)
        axs[2].set_ylabel('$B_{rtn}$\n$[nT]$')
        axs[2].xaxis.set_minor_locator(AutoMinorLocator())
        axs[2].tick_params(axis="x", which='both', direction="in", pad=-15)

        axs[3].plot(epochpmom, vp_t + 200, 'r-', linewidth=.5)
        axs[3].plot(epochpmom, vp_n + 200, 'b-', linewidth=.5)
        axs[3].plot(epochpmom, vp_r, 'k-', linewidth=1)
        axs[3].set_ylabel('$V_{p, rtn}$\n$[km/s]$')
        axs[3].xaxis.set_minor_locator(AutoMinorLocator())
        axs[3].tick_params(axis="x", which='both', direction="in", pad=-15)
        # ax2 = axs[2].twinx()
        # ax2.plot(epochamom, va_r, 'r-')
        # # ax2.set_ylim([0, 210])
        # ax2.set_ylabel(r'$V\alpha_r [km/s]$', color='r')

        axs[4].plot(epochpmom, densp, 'k-', linewidth=1)
        axs[4].set_ylabel('$N_p$ \n$[cm^{-3}]$')
        axs[4].xaxis.set_minor_locator(AutoMinorLocator())
        axs[4].tick_params(axis="x", which='both', direction="in", pad=-15)
        ax2 = axs[4].twinx()
        ax2.plot(epochpmom, Tp, 'r-', linewidth=1)
        ax2.set_ylim([0, 100])
        ax2.set_ylabel('$T_p$ \n $[eV]$', color='r')

        axs[5].set_ylabel(r'$N_{\alpha}$' + '\n $[cm^{-3}]$')
        axs[5].xaxis.set_minor_locator(AutoMinorLocator())
        # axs[4].tick_params(axis="x", which='both', direction="in", pad=-15)
        axs[5].set_xlabel('Time [mm-dd HH]')
        ax2 = axs[5].twinx()
        ax2.plot(epochamom, Ta, 'r-', linewidth=.5)
        ax2.set_ylim([0, 600])
        axs[5].plot(epochamom, densa, 'k-', linewidth=1)
        ax2.set_ylabel(r'$T_{\alpha}$' + '\n $[eV]$', color='r')

        # axs[5].plot(epochpmom, helicity, 'k-', linewidth=1)
        # axs[5].set_ylabel(r'$\sigma_c$')
        # axs[5].xaxis.set_minor_locator(AutoMinorLocator())
        # axs[5].set_xlabel('Time [mm-dd HH]')
        # ax2 = axs[5].twinx()
        # ax2.plot(epochpmom, norm_dB, 'r-',linewidth=1)
        # ax2.set_ylabel(r'$|\delta V_A|$'+'\n $[km/s]$', c='r')
        # ax2 = axs[5].twinx()
        # ax2.plot(epochpmom, Ta / Tp, 'r-')
        # ax2.set_ylim([0, 20])
        # ax2.set_ylabel(r'$T\alpha/Tp$', color='r')
        plt.show()

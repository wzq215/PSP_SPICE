'''Created by Ziqi Wu. @21/07/02
Revised @23/02/14'''
import os
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.offline as py
import matplotlib
from matplotlib.ticker import AutoMinorLocator
from plotly.subplots import make_subplots
from spacepy import pycdf
import scipy.constants as C
# --- Disable codes in need of rlonlat_psp ---
from plot_body_positions import get_rlonlat_psp_carr
import furnsh_kernels

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
    print('Brtn 4sa Files: ', filelist)
    data = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    # print(data)
    return data


def load_RTN_data(start_time_str, stop_time_str):
    '''psp_fld_l2_mag_rtn_2021042800_v02.cdf'''
    cycles = [0, 6, 12, 18]
    start_time = datetime.strptime(start_time_str, '%Y%m%d%H')
    stop_time = datetime.strptime(stop_time_str, '%Y%m%d%H')
    start_file_time = datetime(start_time.year, start_time.month, start_time.day, cycles[divmod(start_time.hour, 6)[0]])
    stop_file_time = datetime(stop_time.year, stop_time.month, stop_time.day, cycles[divmod(stop_time.hour, 6)[0]])

    if divmod(stop_time.hour, 6)[1] == 0:
        if stop_file_time != start_file_time:
            stop_file_time -= timedelta(hours=6)
    filelist = []
    tmp_time = start_file_time

    while tmp_time <= stop_file_time:
        print('hi')
        filelist.append(psp_data_path + 'psp_fld_l2_mag_rtn_' + tmp_time.strftime('%Y%m%d%H') + '_v02.cdf')
        tmp_time += timedelta(hours=6)
    print('Brtn Files: ', filelist)
    data = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    # print(data)
    return data


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
    # print(data)
    return data


def load_fld_qtn_data(start_time_str, stop_time_str):
    start_time = datetime.strptime(start_time_str, '%Y%m%d').toordinal()
    stop_time = datetime.strptime(stop_time_str, '%Y%m%d').toordinal()


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
    dvA = np.nanmean(np.linalg.norm(np.array([dBx, dBy, dBz]), axis=1))
    dv = np.array([dvx, dvy, dvz])
    epoch = epochmag
    zpx, zpy, zpz = dvx - dBx, dvy - dBy, dvz - dBz
    zmx, zmy, zmz = dvx + dBx, dvy + dBy, dvz + dBz
    zplus = np.array([zpx, zpy, zpz])
    zminus = np.array([zmx, zmy, zmz])
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


def do_mag_MVA(epochmag, bx, by, bz, epochpmom, vx, vy, vz, preview=False):
    M = np.array([[np.nanmean(bx ** 2) - np.nanmean(bx) ** 2, np.nanmean(bx * by) - np.nanmean(bx) * np.nanmean(by),
                   np.nanmean(bx * bz) - np.nanmean(bx) * np.nanmean(bz)],
                  [np.nanmean(by * bx) - np.nanmean(by) * np.nanmean(bx), np.nanmean(by ** 2) - np.nanmean(by) ** 2,
                   np.nanmean(by * bz) - np.nanmean(by) * np.nanmean(bz)],
                  [np.nanmean(bz * bx) - np.nanmean(bz) * np.nanmean(bx),
                   np.nanmean(bz * by) - np.nanmean(bz) * np.nanmean(by),
                   np.nanmean(bz ** 2) - np.nanmean(bz) ** 2]])

    [w, v] = np.linalg.eig(M)
    arg_w = np.argsort(w)

    e_N = v[arg_w[0]]
    e_M = v[arg_w[1]]
    e_L = v[arg_w[2]]

    print('e_L: ', e_L)
    print('e_M: ', e_M)
    print('e_N: ', e_N)

    b_L = bx * e_L[0] + by * e_L[1] + bz * e_L[2]
    b_M = bx * e_M[0] + by * e_M[1] + bz * e_M[2]
    b_N = bx * e_N[0] + by * e_N[1] + bz * e_N[2]

    v_L = vx * e_L[0] + vy * e_L[1] + vz * e_L[2]
    v_M = vx * e_M[0] + vy * e_M[1] + vz * e_M[2]
    v_N = vx * e_N[0] + vy * e_N[1] + vz * e_N[2]

    if preview == True:
        plt.figure()
        plt.subplot(311)

        plt.plot(epochmag, by, 'r-', label='By', linewidth=0.5)
        plt.plot(epochmag, bz, 'b-', label='Bz', linewidth=0.5)
        plt.plot(epochmag, bx, 'k-', label='Bx', linewidth=0.5)
        plt.legend()
        plt.ylabel('Bxyz [nT]')

        plt.subplot(312)
        plt.plot(epochmag, b_M, 'r-', label='BM', linewidth=0.5)
        plt.plot(epochmag, b_N, 'b-', label='BN', linewidth=0.5)
        plt.plot(epochmag, b_L, 'k-', label='BL', linewidth=0.5)
        plt.legend()
        plt.ylabel('BLMN [nT]')

        plt.subplot(313)
        # plt.plot(epochpmom,v_L,'k-',label='VL')
        plt.plot(epochpmom, v_M, 'r-', label='VM', linewidth=0.5)
        plt.plot(epochpmom, v_N, 'b-', label='VN', linewidth=0.5)
        plt.legend()
        plt.ylabel('VMN [km/s]')
        ax = plt.twinx()
        ax.plot(epochpmom, v_L, 'k-', label='VL', linewidth=0.5)
        plt.legend()
        plt.ylabel('VL [km/s]')
        plt.show()

    return w[arg_w], v[arg_w], b_L, b_M, b_N, v_L, v_M, v_N


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


if __name__ == '__main__':

    # %
    # -----Set parameters-----
    mag_type = '4sa'  # for mag file, '1min', '4sa', 'rtn'
    inst = False  # for SPI file, True for instrument coordinates, False for Inertial
    style = 6  # Choose output styles [wait for filling]
    alpha = True

    '''
    Style 1: Plotly Figure. (a) |B|&|V|; (b) Br&Vr; (c) Bt&Vt; (d) Bn&Vn; (e) Np&Tp
    Style 2: Plotly Figure. (a) e-PAD; (b) e-norm_PAD; (c) |B|,Br&Vr; (d) Bt,Bn&beta; (e) Np&Tp
    Style 3: Matplotlib Figure. (a) e-PAD; (b) |B|,Br,Bt,Bn; (c) Vr; (d) Np&Tp; (e) lg(beta)
    Style 4: Matplotlib Figure. (a) |B|&Np; (b) Br&Vr; (c) Bt&Vt; (d) Bn&Vn; (e) Tp&lg(beta); (f) e-PAD; (g)e-norm_PAD
    Style 5: Matplotlib Figure. (a) e-norm_PAD; (b) |B|,Br,Bt,Bn; (c) Vr; (d) Np&Tp; (e) lg(beta)
    Style 6: Matplotlib Figure. (a) Carrington_Longitude & Radial Distance (b) e-norm_PAD; (c) |B|,Br,Bt,Bn; (d) Vr,Vt,Vn; (d) Np&Tp; (e) Na&Ta
    '''

    # -----Choose Time range-----
    # # E13; Perihelion@220906
    # beg_time = datetime(2022,9,6,17)
    # end_time = datetime(2022,9,6,18,15)

    # # E12;
    # beg_time = datetime(2022,6,2,17,0)
    # end_time = datetime(2022,6,2,18,0)
    #
    # beg_time = datetime(2022,6,9,4)
    # end_time = datetime(2022,6,9,11)
    #
    # beg_time = datetime(2022,5,20,0)
    # end_time = datetime(2022,5,20,12)
    #
    # # E11;
    # beg_time = datetime(2022,2,25,12,26)
    # end_time = datetime(2022,2,25,12,37)
    #
    # beg_time = datetime(2022,2,17,13,48)
    # end_time = datetime(2022,2,18,1,40)
    #
    # # E10;
    # beg_time = datetime(2021,11,22,0)
    # end_time = datetime(2021,11,22,4)
    #
    # beg_time = datetime(2021,11,29,19)
    # end_time = datetime(2021,11,30,23)
    #
    #
    #
    # # beg_time = datetime(2022,9,12,7)
    # # end_time = datetime(2022,9,12,21,40)
    #
    # E8
    # beg_time = datetime(2021,4,28,12)
    # end_time = datetime(2021,4,30,12)
    # E8 Alpha Abundance
    beg_time = datetime(2021, 4, 28, 15)
    end_time = datetime(2021, 4, 30, 5)
    # # E9
    # beg_time = datetime(2021,8,9,0)
    # end_time = datetime(2021,8,10,23,59)
    # # E10
    # beg_time = datetime(2021,11,20,12)
    # end_time = datetime(2021,11,22,12)
    # # E11
    # beg_time = datetime(2022,2,24,18)
    # end_time = datetime(2022,2,26,18)
    # # E12
    # beg_time = datetime(2022,6,1)
    # end_time = datetime(2022,6,3)
    # # E13
    # beg_time = datetime(2022,9,5)
    # end_time = datetime(2022,9,7)
    # Encounter_str = '13'

    # beg_time = datetime(2021,8,21,12)
    # end_time = datetime(2021,8,22,23)
    # beg_time = datetime(2021,8,9,22,30)
    # end_time = datetime(2021,8,10,4)
    # beg_time = datetime(2022,3,10,5,0,0)
    # end_time = datetime(2022,3,11,19,0,0)
    # beg_time = datetime(2022,3,11,6,15,0)
    # end_time = datetime(2022,3,11,12,0,0)
    # beg_time = datetime(2022, 2, 25, 12, 15, 0)
    # end_time = datetime(2022, 2, 25, 12, 50, 0)
    # beg_time = datetime(2022, 2, 17, 15, 0, 0)
    # end_time = datetime(2022, 2, 18, 2, 0, 0)
    # beg_time = datetime()

    beg_time_str = beg_time.strftime('%Y%m%dT%H%M%S')
    end_time_str = end_time.strftime('%Y%m%dT%H%M%S')

    # %%    # ------Load electron PAD and preview-----
    spe_pad = load_spe_data(beg_time.strftime('%Y%m%d'), end_time.strftime('%Y%m%d'))
    epochpade = spe_pad['Epoch']
    timebinpade = (epochpade > beg_time) & (epochpade < end_time)
    epochpade = epochpade[timebinpade]
    EfluxVsPAE = spe_pad['EFLUX_VS_PA_E'][timebinpade, :, :]
    PitchAngle = spe_pad['PITCHANGLE'][timebinpade, :]
    Energy_val = spe_pad['ENERGY_VALS'][timebinpade, :]
    norm_EfluxVsPAE = EfluxVsPAE * 0
    EfluxVsE = spe_pad['EFLUX_VS_ENERGY']

    for i in range(12):
        norm_EfluxVsPAE[:, i, :] = EfluxVsPAE[:, i, :] / np.nansum(EfluxVsPAE, 1)  # Calculate normalized PAD
    # ! choose energy channel
    i_energy = 8
    print('PAD Energy:', Energy_val[0, i_energy])
    enestr = '%.2f' % Energy_val[0, i_energy]
    # ! choose coloraxis. zmin/max1 for PAD; zmin/max2 for norm_PAD
    zmin1 = 9.
    zmax1 = 10.5
    zmin2 = -1.5
    zmax2 = -0.
    # ! preview PAD and modify energy channel & colorbar. if unnecessary, switch to False.
    preview_PAD = True
    if preview_PAD:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_energy])).T, cmap='jet')
        plt.colorbar()
        plt.clim([zmin1, zmax1])
        plt.subplot(2, 1, 2)
        plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(norm_EfluxVsPAE[:, :, i_energy])).T, cmap='jet')
        plt.colorbar()
        plt.clim([zmin2, zmax2])
        plt.suptitle(enestr)
        plt.show()
    preview_PAD_diff_enes = False
    if preview_PAD_diff_enes:
        dv = 0
        plt.figure()
        plt.subplot(911)
        i_ene = 10
        plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_ene])).T, cmap='jet')
        plt.colorbar()
        plt.ylabel('%d' % Energy_val[0, i_ene] + '[eV]')
        plt.clim([8.3 - dv, 9.3 - dv])

        plt.subplot(912)
        i_ene = 11
        plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_ene])).T, cmap='jet')
        plt.colorbar()
        plt.ylabel('%d' % Energy_val[0, i_ene] + '[eV]')
        plt.clim([8.7 - dv, 9.7 - dv])

        plt.subplot(913)
        i_ene = 12
        plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_ene])).T, cmap='jet')
        plt.colorbar()
        plt.ylabel('%d' % Energy_val[0, i_ene] + '[eV]')
        plt.clim([9. - dv, 10. - dv])

        plt.subplot(914)
        i_ene = 13
        plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_ene])).T, cmap='jet')
        plt.colorbar()
        plt.ylabel('%d' % Energy_val[0, i_ene] + '[eV]')
        plt.clim([9.3 - dv, 10.3 - dv])

        plt.subplot(915)
        i_ene = 14
        plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_ene])).T, cmap='jet')
        plt.colorbar()
        plt.ylabel('%d' % Energy_val[0, i_ene] + '[eV]')
        plt.clim([9.5 - dv, 10.5 - dv])

        plt.subplot(916)
        i_ene = 15
        plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_ene])).T, cmap='jet')
        plt.colorbar()
        plt.ylabel('%d' % Energy_val[0, i_ene] + '[eV]')
        plt.clim([9.6 - dv, 10.6 - dv])

        plt.subplot(917)
        i_ene = 16
        plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_ene])).T, cmap='jet')
        plt.colorbar()
        plt.ylabel('%d' % Energy_val[0, i_ene] + '[eV]')
        plt.clim([9.7 - dv, 10.7 - dv])

        plt.subplot(918)
        i_ene = 17
        plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_ene])).T, cmap='jet')
        plt.colorbar()
        plt.ylabel('%d' % Energy_val[0, i_ene] + '[eV]')
        plt.clim([9.7 - dv, 10.7 - dv])

        plt.subplot(919)
        i_ene = 18
        plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_ene])).T, cmap='jet')
        plt.colorbar()
        plt.ylabel('%d' % Energy_val[0, i_ene] + '[eV]')
        plt.clim([9.8 - dv, 10.8 - dv])

        plt.show()
    preview_norm_PAD_diff_enes = False
    if preview_norm_PAD_diff_enes:
        v_min = -1.5
        v_max = -0
        plt.figure()
        plt.subplot(911)
        i_ene = 10
        plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(norm_EfluxVsPAE[:, :, i_ene])).T, cmap='jet')
        plt.colorbar()
        plt.ylabel('%d' % Energy_val[0, i_ene] + '[eV]')
        plt.clim([v_min, v_max])

        plt.subplot(912)
        i_ene = 11
        plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(norm_EfluxVsPAE[:, :, i_ene])).T, cmap='jet')
        plt.colorbar()
        plt.ylabel('%d' % Energy_val[0, i_ene] + '[eV]')
        plt.clim([v_min, v_max])

        plt.subplot(913)
        i_ene = 12
        plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(norm_EfluxVsPAE[:, :, i_ene])).T, cmap='jet')
        plt.colorbar()
        plt.ylabel('%d' % Energy_val[0, i_ene] + '[eV]')
        plt.clim([v_min, v_max])

        plt.subplot(914)
        i_ene = 13
        plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(norm_EfluxVsPAE[:, :, i_ene])).T, cmap='jet')
        plt.colorbar()
        plt.ylabel('%d' % Energy_val[0, i_ene] + '[eV]')
        plt.clim([v_min, v_max])

        plt.subplot(915)
        i_ene = 14
        plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(norm_EfluxVsPAE[:, :, i_ene])).T, cmap='jet')
        plt.colorbar()
        plt.ylabel('%d' % Energy_val[0, i_ene] + '[eV]')
        plt.clim([v_min, v_max])

        plt.subplot(916)
        i_ene = 15
        plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(norm_EfluxVsPAE[:, :, i_ene])).T, cmap='jet')
        plt.colorbar()
        plt.ylabel('%d' % Energy_val[0, i_ene] + '[eV]')
        plt.clim([v_min, v_max])

        plt.subplot(917)
        i_ene = 16
        plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(norm_EfluxVsPAE[:, :, i_ene])).T, cmap='jet')
        plt.colorbar()
        plt.ylabel('%d' % Energy_val[0, i_ene] + '[eV]')
        plt.clim([v_min, v_max])

        plt.subplot(918)
        i_ene = 17
        plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(norm_EfluxVsPAE[:, :, i_ene])).T, cmap='jet')
        plt.colorbar()
        plt.ylabel('%d' % Energy_val[0, i_ene] + '[eV]')
        plt.clim([v_min, v_max])

        plt.subplot(919)
        i_ene = 20
        plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(norm_EfluxVsPAE[:, :, i_ene])).T, cmap='jet')
        plt.colorbar()
        plt.ylabel('%d' % Energy_val[0, i_ene] + '[eV]')
        plt.clim([v_min, v_max])

        plt.show()

    # %%  -----Load mag data-----
    if mag_type == '1min':
        mag_RTN = load_RTN_1min_data(beg_time.strftime('%Y%m%d'), end_time.strftime('%Y%m%d'))

        epochmag = mag_RTN['epoch_mag_RTN_1min']
        timebinmag = (epochmag > beg_time) & (epochmag < end_time)
        epochmag = epochmag[timebinmag]

        Br = mag_RTN['psp_fld_l2_mag_RTN_1min'][timebinmag, 0]
        Bt = mag_RTN['psp_fld_l2_mag_RTN_1min'][timebinmag, 1]
        Bn = mag_RTN['psp_fld_l2_mag_RTN_1min'][timebinmag, 2]
        Babs = np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2)

    elif mag_type == 'rtn':
        mag_RTN = load_RTN_data(beg_time.strftime('%Y%m%d%H'), end_time.strftime('%Y%m%d%H'))

        epochmag = mag_RTN['epoch_mag_RTN']
        timebinmag = (epochmag > beg_time) & (epochmag < end_time)
        epochmag = epochmag[timebinmag]

        Br = mag_RTN['psp_fld_l2_mag_RTN'][timebinmag, 0]
        Bt = mag_RTN['psp_fld_l2_mag_RTN'][timebinmag, 1]
        Bn = mag_RTN['psp_fld_l2_mag_RTN'][timebinmag, 2]
        Babs = np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2)

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
        T_tensor_a = amom_SPI['T_TENSOR_INST'][timebinamom]
        MagF_inst_a = amom_SPI['MAGF_INST'][timebinamom]

    # %% Check if max_flux lies in the FOV of SPAN.Switch to False if unnecessary.
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
    # %%
    r_psp_carr_pmom, lon_psp_carr_pmom, lat_psp_carr_pmom = get_rlonlat_psp_carr(epochamom, for_psi=False)
    r_psp_carr_mag, lon_psp_carr_mag, lat_psp_carr_mag = get_rlonlat_psp_carr(epochmag, for_psi=False)

    # unify epoch
    densp = densp[0:len(densa)]
    epochpmom = epochpmom[0:len(densa)]
    Tp = Tp[0:len(densa)]
    vp_r = vp_r[0:len(densa)]
    vp_t = vp_t[0:len(densa)]
    vp_n = vp_n[0:len(densa)]
    # %%
    # ----- calculate secondary properties -----
    mp = 1.6726e-27
    ma = mp * 4
    boltzmann = 1.3807e-23
    e = 1.6022e-19

    # interp total magnetic field to pmoms epoch and calculate beta.
    Babs_p = np.interp(np.array((epochpmom - epochpmom[0]) / timedelta(days=1), dtype='float64'),
                       np.array((epochmag - epochpmom[0]) / timedelta(days=1), dtype='float64'), Babs)
    Br_p = np.interp(np.array((epochpmom - epochpmom[0]) / timedelta(days=1), dtype='float64'),
                     np.array((epochmag - epochpmom[0]) / timedelta(days=1), dtype='float64'), Br)
    Bn_p = np.interp(np.array((epochpmom - epochpmom[0]) / timedelta(days=1), dtype='float64'),
                     np.array((epochmag - epochpmom[0]) / timedelta(days=1), dtype='float64'), Bn)
    Bt_p = np.interp(np.array((epochpmom - epochpmom[0]) / timedelta(days=1), dtype='float64'),
                     np.array((epochmag - epochpmom[0]) / timedelta(days=1), dtype='float64'), Bt)

    # calculate plasma beta
    beta = 4.03e-11 * densp * Tp / ((Babs_p * 1e-5) ** 2)
    # calculate thermal pressure P_t and magnetic pressure P_b
    P_t = densp * Tp * 1e6 * e * 1e9
    dP_t = P_t - np.mean(P_t)
    P_b = 3.93 * (Babs_p * 1e-9) ** 2 * 1.0133e5 * 1e9
    dP_b = P_b - np.mean(P_b)
    # calculate alfven veolocity
    Valf = 2.18e1 * Babs_p * densp ** (-0.5)  # km/s
    # calculate (Va-Vp)/Va
    Vap = np.sqrt((va_r - vp_r) ** 2 + (va_n - vp_n) ** 2 + (va_t - vp_t) ** 2)
    Vap = np.sqrt((va_r) ** 2 + (va_n) ** 2 + (va_t) ** 2) - np.sqrt((vp_r) ** 2 + (vp_n) ** 2 + (vp_t) ** 2)
    # %%
    do_MVA = False
    if do_MVA:
        # Do MVA Test
        # mva_beg = datetime(2021,8,21,20,0,0)
        # mva_end = datetime(2021,8,22,16,0,0)
        # mva_beg = datetime(2022, 2, 25, 12, 25, 0)
        # mva_end = datetime(2022, 2, 25, 12, 40, 0)
        # mva_beg = datetime(2022, 2, 17, 15, 0, 0)
        # mva_end = datetime(2022, 2, 18, 2, 0, 0)
        # mva_beg = datetime(2022, 3, 11, 6, 15, 0)
        # mva_end = datetime(2022, 3, 11, 12, 0, 0)
        # mva_beg = datetime(2022, 3, 10, 5, 0, 0)
        # mva_end = datetime(2022, 3, 11, 19, 0, 0)
        # mva_beg = datetime(2022,9,6,17,25)
        # mva_end = datetime(2022,9,6,17,45)
        mva_beg = datetime(2022, 6, 2, 17)
        mva_end = datetime(2022, 6, 2, 18)
        mva_beg = datetime(2022, 6, 9, 7, 20)
        mva_end = datetime(2022, 6, 9, 8, 0)
        mva_beg = datetime(2022, 5, 20, 4)
        mva_end = datetime(2022, 5, 20, 6)
        mva_beg = datetime(2022, 2, 17, 15)
        mva_end = datetime(2022, 2, 18, 1)
        mva_beg = datetime(2022, 2, 25, 12, 26)
        mva_end = datetime(2022, 2, 25, 12, 37)
        mva_beg = datetime(2021, 11, 22, 2, 15)
        mva_end = datetime(2021, 11, 22, 2, 45)
        mva_magbin = (epochmag > mva_beg) & (epochmag < mva_end)
        mva_pmombin = (epochpmom > mva_beg) & (epochpmom < mva_end)
        mva_epochmag = epochmag[mva_magbin]
        mva_epochpmom = epochpmom[mva_pmombin]
        mva_Br = np.array(Br[mva_magbin])
        mva_Bt = np.array(Bt[mva_magbin])
        mva_Bn = np.array(Bn[mva_magbin])
        mva_Vr = np.array(vp_r[mva_pmombin])
        mva_Vt = np.array(vp_t[mva_pmombin])
        mva_Vn = np.array(vp_n[mva_pmombin])
        w, v, b_L, b_M, b_N, v_L, v_M, v_N = do_mag_MVA(mva_epochmag, mva_Br, mva_Bt, mva_Bn, mva_epochpmom, mva_Vr,
                                                        mva_Vt, mva_Vn, preview=True)

    # %%
    calc_anisotropy = False
    if calc_anisotropy:
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
    # %%
    calc_AC = True
    if calc_AC:
        ln_lambda = 21.
        e_a = 2 * C.e / np.sqrt(4 * np.pi * C.epsilon_0)
        e_p = C.e / np.sqrt(4 * np.pi * C.epsilon_0)
        m_a = 4 * C.m_p
        mu_ap = m_a * C.m_p / (m_a + C.m_p)

        rho_ap = (densp * C.m_p + densa * m_a) * 1e6  # kg*m^-3
        w_ap = np.sqrt(2 * Ta * C.e / (4 * C.m_p) + 2 * Tp * C.e / C.m_p)  # m/s

        time_sw = r_psp_carr_pmom * 696300 / (vp_r)

        collision_freq_H1985 = 4 * np.pi * ln_lambda * e_a ** 2 * e_p ** 2 / (m_a * C.m_p) * rho_ap / mu_ap / (
                    w_ap ** 3)
        collision_freq_H1987 = 32 / 3 * np.sqrt(np.pi) * ln_lambda * e_a ** 2 * e_p ** 2 / (
                    m_a * C.m_p) * densp * 1e6 / (w_ap ** 3)
        # collision_freq_H1987 = 1./(133*(w_ap*1e-3)**3/densp)
        Ac_H1985 = time_sw * collision_freq_H1985
        Ac_H1987 = time_sw * collision_freq_H1987

        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(epochpmom, rho_ap / (mu_ap * w_ap ** 3))
        plt.title('rho w_ap^-3 mu^-1')
        plt.subplot(3, 1, 2)
        plt.semilogy(epochpmom, collision_freq_H1985, linewidth=2, label='H1985_tc^-1')
        plt.semilogy(epochpmom, collision_freq_H1987, '--', linewidth=0.5, label='H1987_tc^-1')
        plt.ylabel('Collision Frequency (Hz)')
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.semilogy(epochpmom, Ac_H1985, linewidth=2, label='H1985')
        plt.semilogy(epochpmom, Ac_H1987, '--', linewidth=0.5, label='H1987')
        plt.ylabel('Collisional Age')
        plt.legend()
        plt.show()

        # %%

        lambda_p = 9.42 + np.log((Tp * 1.1604e4) ** (3 / 2) / densp ** (1 / 2))
        tc = 133 * (np.sqrt(2 * Ta * C.e / (4 * C.m_p) + 2 * Tp * C.e / C.m_p) * 1e-3) ** 3 / densp  # s
        Ac_ap = r_psp_carr_pmom / (np.sqrt((vp_r) ** 2 + (vp_n) ** 2 + (vp_t) ** 2) * tc) * 696300

        lambda_p = 9.42 + np.log((Tp * 1.1604e4) ** (3 / 2) / densp ** (1 / 2))

        plt.figure()
        plt.plot(lambda_p)
        plt.show()
        Ac_pp = 1.31e7 * densp / (vp_r * (Tp * 1.1604e4) ** (3 / 2)) * r_psp_carr_pmom * 6.96e10 / 1.5e13 * lambda_p

        rho_ab = (densp * mp + densa * ma)
        thermal_ene_ab = (np.sqrt(2 * Ta * e / ma + 2 * Tp * e / mp) * 1e-3) ** 3
        mass_ab = ma * mp / (ma + mp)
        simple_collisional_rate = rho_ab / (thermal_ene_ab * mass_ab)
        aa_rate = 4 * np.pi * lambda_p * (
                    C.e ** 2 / (4 * np.pi * C.epsilon_0) / C.m_p) ** 2 * simple_collisional_rate * 1e-3
        Ac_ap_H1985 = r_psp_carr_pmom / (np.sqrt((vp_r) ** 2 + (vp_n) ** 2 + (vp_t) ** 2)) / aa_rate * 696300

        # simple_collisional_rate_1987 = densp*mp/thermal_ene_ab/mass_ab*lambda_p
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.scatter(epochpmom, simple_collisional_rate, s=0.1)
        plt.title('rho w_ap^-3 mu^-1')
        plt.subplot(3, 1, 2)
        plt.scatter(epochpmom, densp / ((np.sqrt(2 * Tp * e / mp) * 1e-3) ** 3) * np.sqrt(ma / mp), s=0.1)
        plt.title('Np w_p^-3 (ma/mp)^1/2')
        # plt.plot(simple_collisional_rate_1987)
        plt.subplot(3, 1, 3)
        plt.scatter(epochpmom,
                    simple_collisional_rate / (densp / ((np.sqrt(2 * Tp * e / mp) * 1e-3) ** 3) * np.sqrt(ma / mp)),
                    s=0.1)
        plt.title('Ratio')
        plt.show()
        plt.figure()
        plt.plot(epochpmom, Ac_ap, label='Ac_alpha-proton')
        plt.plot(epochpmom, Ac_ap_H1985, label='Hernandez1985')
        # plt.plot(epochpmom, Ac_pp, label='Ac_proton-proton')
        plt.title('Ac ap/pp')
        plt.legend()
        plt.show()
        Ac = Ac_ap
    # %%
    calc_hel = True
    windowsecs = 60 * 20
    if calc_hel:
        helicity, norm_dB = calc_helicity(epochmag, Br, Bt, Bn, epochpmom, vp_r, vp_t, vp_n, densp,
                                          windowsecs=windowsecs)
        np.savez('helicity_normdB' + str(windowsecs), epochpmom=epochpmom, helicity=helicity, windowsecs=windowsecs,
                 norm_dB=norm_dB)
    with np.load('helicity_normdB' + str(windowsecs) + '.npz') as f:
        helicity, norm_dB = f['helicity'], f['norm_dB']

    # %%   # ----- plot previews -----
    # plot preview of proton/alpha moms
    if alpha:
        fig, axs = plt.subplots(6, 1, sharex=True)

        axs[0].scatter(epochpmom, 100 * densa / densp, c='k', s=0.1, label=r'$AHe$')
        axs[0].set_xlim([epochpmom[0], epochpmom[-1]])
        axs[0].set_ylim([-0.1, 3])
        axs[0].set_ylabel(r'$A_{\alpha} [\%]$', c='k')
        # ax2 = axs[0].twinx()
        # ax2.plot(epochpmom, densp, 'coral', label=r'$Np$',linewidth=1.)
        # ax2.set_ylabel('$N_p$\n$[cm^-3]$',c='coral')
        axs[0].xaxis.set_minor_locator(AutoMinorLocator())
        axs[0].tick_params(axis="x", which='both', direction="in", pad=-15)

        axs[1].scatter(epochpmom, Vap / Valf, label='Vap/VA', c='k', s=0.1)
        axs[1].set_ylabel(r'$V_{\alpha p}/V_A$', c='k')
        axs[1].set_ylim([-1, 2])
        axs[1].set_xlim([epochpmom[0], epochpmom[-1]])
        # ax2 = axs[1].twinx()
        # ax2.plot(epochpmom, vp_r, 'coral', label='Vpr',linewidth=1.)
        # ax2.set_ylabel('$V_{pr}$\n $[km/s]$',c='coral')
        axs[1].xaxis.set_minor_locator(AutoMinorLocator())
        axs[1].tick_params(axis="x", which='both', direction="in", pad=-15)

        axs[2].scatter(epochpmom, Ta / Tp, label=r'$Tα$', c='k', s=0.1)
        axs[2].set_ylim([-0.1, 20])
        axs[2].set_ylabel(r'$T_{\alpha}/T_p$', c='k')
        axs[2].set_xlim([epochpmom[0], epochpmom[-1]])
        # ax2 = axs[2].twinx()
        # ax2.plot(epochpmom, Tp, 'coral', label=r'$Tp$',linewidth=1.)
        # ax2.set_ylabel('$T_p$\n$[eV]$', c='coral')
        axs[2].xaxis.set_minor_locator(AutoMinorLocator())
        axs[2].tick_params(axis="x", which='both', direction="in", pad=-15)

        axs[3].scatter(epochpmom, Ac, c='k', s=0.1)
        axs[3].set_ylabel('Collisional Age')
        axs[3].set_xlim([epochpmom[0], epochpmom[-1]])

        axs[5].scatter(epochpmom, abs(helicity), c='k', s=0.1)
        axs[5].set_ylabel(r'$\sigma_c$')
        axs[5].set_xlim([epochpmom[0], epochpmom[-1]])

        axs[4].plot(epochpmom, norm_dB, 'k', linewidth=0.5)
        axs[4].set_ylabel(r'$|\delta V_A|$')
        axs[5].set_xlabel('Time [mm-dd HH]')
        axs[4].set_xlim([epochpmom[0], epochpmom[-1]])

        plt.show()

        # plt.figure()
        # plt.plot(epochpmom, vp_r * 1e5 * densp - ma * 1e3 * 6.6726e-8 * 2e33 * (1e6) ** 1.5 / (
        #         1.2e-23 * (r_psp_carr_pmom * 696300 * 1e5) ** 2))
        # plt.show()
        #
        # HCS_timebin = ((epochpmom > datetime(2021, 4, 29, 0, 55)) & (epochpmom < datetime(2021, 4, 29, 1, 55))) | \
        #               ((epochpmom > datetime(2021, 4, 29, 8, 15)) & (epochpmom < datetime(2021, 4, 29, 8, 55))) | \
        #               ((epochpmom > datetime(2021, 4, 29, 9, 23)) & (epochpmom < datetime(2021, 4, 29, 10, 23))) | \
        #               ((epochpmom > datetime(2021, 4, 29, 13, 40)) & (epochpmom < datetime(2021, 4, 29, 13, 50)))
        # SSW1_timebin = (epochpmom < datetime(2021, 4, 29, 13, 40)) & ~HCS_timebin
        # SSW2_timebin = (epochpmom > datetime(2021, 4, 29, 13, 40)) & ~HCS_timebin
        #
        # df = pd.DataFrame()
        # df['epoch'] = epochamom
        # df[r'A_{\alpha}'] = 100 * densa / densp
        # df[r'$T_{\alpha}/T_{p}$'] = Ta / Tp
        # df[r'$dV_{\alpha p}/V_{A}$'] = Vap / Valf
        # df['Ac'] = Ac
        # df['type'] = 'SSW1'
        # df['type'][HCS_timebin] = 'HCS'
        # df['type'][SSW2_timebin] = 'SSW2'
        # df['Proton Flux'] = vp_r * densp
        # df['$V_pr$'] = vp_r
        # df['$\sigma_c$'] = np.abs(helicity)
        # df['$|\delta V_{A}|$'] = norm_dB
        # df['Tp_para/perp'] = Tp_para / Tp_perp
        # df['Ta_para/perp'] = Ta_para / Ta_perp
        # df.to_csv('alpha_properties.csv')
    # %%
    preview_helicity = False
    if preview_helicity == True:
        fig, axs = plt.subplots(4, 1, sharex=True)
        axs[0].plot(epochpmom, helicity, 'steelblue-')
        axs[0].set_ylabel(r'$\sigma_c$')
        ax2 = axs[0].twinx()
        ax2.plot(epochpmom, norm_dB, 'coral')
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
    # %%
    save_overview = False
    if save_overview == True:
        df_pmoms = pd.DataFrame()
        df_pmoms['epochp'] = epochpmom
        df_pmoms['Vr'] = vp_r
        df_pmoms['Vt'] = vp_t
        df_pmoms['Vn'] = vp_n
        df_pmoms['Np'] = densp
        df_pmoms['Tp'] = Tp
        df_pmoms.to_csv('export/load_psp_data/psp_overview/pmom_(' + beg_time_str + '-' + end_time_str + ').csv')

        df_mag = pd.DataFrame()
        df_mag['epochmag'] = epochmag
        df_mag['Btot'] = Babs
        df_mag['Br'] = Br
        df_mag['Bt'] = Bt
        df_mag['Bn'] = Bn
        df_mag.to_csv('export/load_psp_data/psp_overview/mag_(' + beg_time_str + '-' + end_time_str + ').csv')

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
        filename = 'figures/overviews/Overview_' + mag_type + '_(' + beg_time_str + '-' + end_time_str + ').html'
        fig.add_annotation(x=0.5, y=-.17, text='test_annotation', xref='paper', yref='paper', showarrow=False)
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
        filename = 'figures/overviews/Overview_PAD_' + mag_type + '_(' + beg_time_str + '-' + end_time_str + ').html'
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
        pos = axs[0].pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(norm_EfluxVsPAE[:, :, i_energy])).T,
                                cmap='jet', vmax=zmax2, vmin=zmin2)
        axs[0].set_ylabel('$e^-$ (' + enestr + ' eV) \n Pitch Angle [deg]')
        axs[0].xaxis.set_minor_locator(AutoMinorLocator())
        # axs[0].title(enestr)
        axs[0].tick_params(axis="x", which='both', direction="in", pad=-15)

        axs[1].plot(epochmag, Br, 'k-', label='Br', zorder=4)
        axs[1].plot(epochmag, Bt, 'r-', label='Bt', zorder=1)
        axs[1].plot(epochmag, Bn, 'b-', label='Bn', zorder=2)
        axs[1].plot(epochmag, np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2), 'm-', label='|B|', zorder=3)

        axs[1].set_ylabel('B\n[nT]')
        # axs[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        #               ncol=2, mode="expand", borderaxespad=0.)
        axs[1].legend(loc=2, bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
        axs[1].xaxis.set_minor_locator(AutoMinorLocator())
        axs[1].tick_params(axis="x", which='both', direction="in", pad=-15)

        axs[2].plot(epochpmom, vp_r, 'k-')
        axs[2].set_ylabel('$V$\n$[km/s]$')
        axs[2].xaxis.set_minor_locator(AutoMinorLocator())
        axs[2].tick_params(axis="x", which='both', direction="in", pad=-15)

        axs[3].plot(epochpmom, densp, 'k-', zorder=2)
        axs[3].set_ylabel('$N_p$\n$[cm^{-3}]$')
        axs[3].xaxis.set_minor_locator(AutoMinorLocator())
        axs[3].tick_params(axis="x", which='both', direction="in", pad=-15)
        ax2 = axs[3].twinx()
        ax2.plot(epochpmom, Tp, 'r-', zorder=1)
        ax2.set_zorder = 1
        # ax2.set_ylim([0, 500])
        ax2.set_ylabel('$T_p$\n$[eV]$', color='r')

        axs[4].plot(epochpmom, np.log10(beta), 'k-')
        # axs[4].set_ylabel(r'$\lg (\beta)$')
        axs[4].set_ylabel(r'$\lg\beta$')
        axs[4].xaxis.set_minor_locator(AutoMinorLocator())
        axs[4].set_xlabel('Time')
        # ax2 = axs[4].twinx()
        # ax2.plot(epochpmom,densa*100/densp,'r-')
        # ax2.set_ylim([0,2])
        # ax2.set_ylabel(r'$A_{\alpha}%$',color='r')

        plt.show()
    if style == 6:
        '''Style 6: Matplotlib Figure.
        (a) Carrington_Longitude & Radial Distance (b) e-norm_PAD;
        (c) |B|,Br,Bt,Bn; (d) Vr,Vt,Vn; (d) Np&Tp; (e) Na&Ta'''

        fig, axs = plt.subplots(6, 1, sharex=True)

        axs[0].plot(epochpmom, r_psp_carr_pmom, 'k-', linewidth=1)
        ax2 = axs[0].twinx()
        ax2.plot(epochpmom, np.rad2deg(lon_psp_carr_pmom), 'r-', linewidth=1)
        ax2.set_ylabel('Carrington\n Longitude [deg]')
        axs[0].set_ylabel('Radial\n Distance [Rs]')

        pos = axs[1].pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_energy])).T,
                                cmap='jet', vmax=zmax1, vmin=zmin1)
        axs[1].set_ylabel('Pitch Angle \n[deg]')
        axs[1].xaxis.set_minor_locator(AutoMinorLocator())
        axs[1].tick_params(axis="x", which='both', direction="in", pad=-15)
        axpos = axs[1].get_position()
        pad = 0.01
        width = 0.01
        caxpos = matplotlib.transforms.Bbox.from_extents(
            axpos.x1 + pad,
            axpos.y0,
            axpos.x1 + pad + width,
            axpos.y1
        )
        cax = axs[1].figure.add_axes(caxpos)
        cbar = plt.colorbar(pos, cax=cax)

        # axs[2].plot(epochmag, Bt, 'r-', linewidth=.5,label='Bt')
        # axs[2].plot(epochmag, Bn, 'b-', linewidth=.5,label='Bn')
        axs[2].plot(epochmag, np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2), 'r', linewidth=1, label='Btot')
        axs[2].plot(epochmag, Br, 'k-', linewidth=1, label='Br')
        axs[2].set_ylabel('$B$\n$[nT]$')
        axs[2].xaxis.set_minor_locator(AutoMinorLocator())
        axs[2].tick_params(axis="x", which='both', direction="in", pad=-15)
        axs[2].legend(loc=2, bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)

        # axs[3].plot(epochpmom, vp_t, 'r-', linewidth=.5,label='Vt')
        # axs[3].plot(epochpmom, vp_n, 'b-', linewidth=.5,label='Vp')
        axs[3].plot(epochpmom, vp_r, 'k-', linewidth=1)
        axs[3].set_ylabel('$V_{p}$\n$[km/s]$')
        axs[3].xaxis.set_minor_locator(AutoMinorLocator())
        axs[3].tick_params(axis="x", which='both', direction="in", pad=-15)
        # axs[3].legend(loc=2, bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)

        axs[4].plot(epochpmom, densp, 'k-', linewidth=1)
        axs[4].set_ylabel('$N_p$ \n$[cm^{-3}]$')
        axs[4].xaxis.set_minor_locator(AutoMinorLocator())
        # axs[4].tick_params(axis="x", which='both', direction="in", pad=-15)
        # axs[4].set_ylim([0,100])
        ax2 = axs[4].twinx()
        ax2.plot(epochpmom, Tp, 'r-', linewidth=1)
        # ax2.set_ylim([0, 500])
        ax2.set_ylabel('$T_p$ \n $[eV]$', color='r')

        axs[5].set_ylabel(r'$N_{\alpha}$' + '\n $[cm^{-3}]$')
        axs[5].xaxis.set_minor_locator(AutoMinorLocator())
        # axs[4].tick_params(axis="x", which='both', direction="in", pad=-15)
        axs[5].set_xlabel('Time [mm-dd HH]')
        ax2 = axs[5].twinx()
        ax2.plot(epochamom, Ta, 'r-', linewidth=.25)
        ax2.set_ylim([0, 600])
        axs[5].plot(epochamom, densa, 'k-', linewidth=1)
        ax2.set_ylabel(r'$T_{\alpha}$' + '\n $[eV]$', color='r')

        plt.show()
    if style == 7:
        '''Style 5: Matplotlib Figure. (a) e-norm_PAD; (b) |B|,Br,Bt,Bn; (c) Vr; (d) Np&Tp; (e) lg(beta)'''
        fig, axs = plt.subplots(5, 1, sharex=True)

        axs[0].plot(epochmag, Bt, 'r-', label='Bt')
        axs[0].plot(epochmag, Bn, 'b-', label='Bn')
        axs[0].plot(epochmag, np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2), 'm-', label='|B|')
        axs[0].plot(epochmag, Br, 'k-', label='Br')
        axs[0].set_ylabel('B\n[nT]')
        axs[0].legend(loc='right')
        axs[0].xaxis.set_minor_locator(AutoMinorLocator())
        axs[0].tick_params(axis="x", which='both', direction="in", pad=-15)

        axs[1].plot(epochpmom, np.log10(beta), 'k-')
        axs[1].plot(epochpmom, np.log10(beta * 0 + 1.), 'r--')
        axs[1].set_ylabel(r'\beta')
        axs[1].xaxis.set_minor_locator(AutoMinorLocator())
        # axs[1].set_xlabel('Time')

        # i_ene = 8
        # i_ax = 2
        # zmax1 = 5.5
        # zmin1 = 4.5
        # pos = axs[i_ax].pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_ene])).T,
        #                         cmap='jet', vmax=zmax1, vmin=zmin1)
        # axs[i_ax].set_ylabel('%d'%Energy_val[0, i_ene]+'[eV]')
        # axs[i_ax].xaxis.set_minor_locator(AutoMinorLocator())
        # axs[i_ax].tick_params(axis="x", which='both', direction="in", pad=-15)
        #
        # i_ene = 10
        # i_ax = 3
        # zmax1 = 6
        # zmin1 = 5
        # pos = axs[i_ax].pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_ene])).T,
        #                            cmap='jet', vmax=zmax1, vmin=zmin1)
        # axs[i_ax].set_ylabel('%d'%Energy_val[0, i_ene]+'[eV]')
        # axs[i_ax].xaxis.set_minor_locator(AutoMinorLocator())
        # axs[i_ax].tick_params(axis="x", which='both', direction="in", pad=-15)

        # i_ene = 12
        # i_ax = 2
        # zmax1 = 6
        # zmin1 = 5
        # pos = axs[i_ax].pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_ene])).T,
        #                            cmap='jet', vmax=zmax1, vmin=zmin1)
        # axs[i_ax].set_ylabel('%d'%Energy_val[0, i_ene]+'[eV]')
        # axs[i_ax].xaxis.set_minor_locator(AutoMinorLocator())
        # axs[i_ax].tick_params(axis="x", which='both', direction="in", pad=-15)

        i_ene = 14
        i_ax = 2
        zmax1 = 7
        zmin1 = 5.5
        pos = axs[i_ax].pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_ene])).T,
                                   cmap='jet', vmax=zmax1, vmin=zmin1)
        axs[i_ax].set_ylabel('%d' % Energy_val[0, i_ene] + '[eV]')
        axs[i_ax].xaxis.set_minor_locator(AutoMinorLocator())
        axs[i_ax].tick_params(axis="x", which='both', direction="in", pad=-15)

        # i_ene = 16
        # i_ax = 4
        # zmax1 = 7
        # zmin1 = 6
        # pos = axs[i_ax].pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_ene])).T,
        #                            cmap='jet', vmax=zmax1, vmin=zmin1)
        # axs[i_ax].set_ylabel('%d'%Energy_val[0, i_ene]+'[eV]')
        # axs[i_ax].xaxis.set_minor_locator(AutoMinorLocator())
        # axs[i_ax].tick_params(axis="x", which='both', direction="in", pad=-15)

        i_ene = 18
        i_ax = 3
        zmax1 = 8
        zmin1 = 6.5
        pos = axs[i_ax].pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_ene])).T,
                                   cmap='jet', vmax=zmax1, vmin=zmin1)
        axs[i_ax].set_ylabel('%d' % Energy_val[0, i_ene] + '[eV]')
        axs[i_ax].xaxis.set_minor_locator(AutoMinorLocator())
        # axs[i_ax].tick_params(axis="x", which='both', direction="out", pad=-15)

        # i_ene = 20
        # i_ax = 6
        # zmax1 = 8.5
        # zmin1 = 6.5
        # pos = axs[i_ax].pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_ene])).T,
        #                            cmap='jet', vmax=zmax1, vmin=zmin1)
        # axs[i_ax].set_ylabel('%d'%Energy_val[0, i_ene]+'[eV]')
        # axs[i_ax].xaxis.set_minor_locator(AutoMinorLocator())

        i_ene = 22
        i_ax = 4
        zmax1 = 8.5
        zmin1 = 7.
        pos = axs[i_ax].pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_ene])).T,
                                   cmap='jet', vmax=zmax1, vmin=zmin1)
        axs[i_ax].set_ylabel('%d' % Energy_val[0, i_ene] + '[eV]')
        axs[i_ax].xaxis.set_minor_locator(AutoMinorLocator())

        plt.show()
    elif style == 8:
        '''Style 1: Plotly Figure. (a) |B|&|V|; (b) Br&Vr; (c) Bt&Vt; (d) Bn&Vn; (e) Np&Tp'''
        fig = make_subplots(rows=6, cols=1,
                            specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}],
                                   [{"secondary_y": True}],
                                   [{"secondary_y": True}], [{"secondary_y": True}]],
                            subplot_titles=("PSP position in Carrington Coordinates",
                                            "e-PAD" + ' (%d' % Energy_val[0, i_energy] + 'eV)', "B_R & V_R",
                                            "B_T & V_T", "B_N & V_N", "Np & Tp"), shared_xaxes=True)

        fig.add_trace(go.Scatter(x=epochpmom, y=r_psp_carr_pmom, name='R_{sc}', mode='lines', line_color='blue'), row=1,
                      col=1)
        fig.add_trace(
            go.Heatmap(x=epochpade, y=PitchAngle[0, :], z=np.log10(EfluxVsPAE[:, :, i_energy].T), colorscale='jet',
                       zmin=zmin1, zmax=zmax1, showscale=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=epochmag, y=Br, name='Br', mode='lines', line_color='blue'), row=3, col=1)
        fig.add_trace(go.Scatter(x=epochmag, y=Bt, name='Bt', mode='lines', line_color='blue'), row=4, col=1)
        fig.add_trace(go.Scatter(x=epochmag, y=Bn, name='Bn', mode='lines', line_color='blue'), row=5, col=1)
        fig.add_trace(go.Scatter(x=epochpmom, y=densp, name='np', mode='lines', line_color='blue'), row=6, col=1)

        fig.add_trace(
            go.Scatter(x=epochpmom, y=np.rad2deg(lon_psp_carr_pmom), name='Lon_{sc}', mode='lines', line_color='red'),
            row=1, col=1, secondary_y=True)
        # fig.add_trace(
        #     go.Scatter(x=epochpmom, y=np.sqrt(vp_r ** 2 + vp_n ** 2 + vp_t ** 2), name='|V|', mode='lines',
        #                line_color='red'), row=3, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=epochpmom, y=vp_r, name='Vr', mode='lines', line_color='red'), row=3, col=1,
                      secondary_y=True)
        fig.add_trace(go.Scatter(x=epochpmom, y=vp_t, name='Vt', mode='lines', line_color='red'), row=4, col=1,
                      secondary_y=True)
        fig.add_trace(go.Scatter(x=epochpmom, y=vp_n, name='Vn', mode='lines', line_color='red'), row=5, col=1,
                      secondary_y=True)
        fig.add_trace(go.Scatter(x=epochpmom, y=Tp, name='Tp', mode='lines', line_color='red'), row=6, col=1,
                      secondary_y=True)

        fig.update_xaxes(tickformat="%m/%d %H:%M\n%Y", title_text="Epoch", row=6, col=1)

        fig.update_yaxes(title_text="Radial<br />Distance<br />(Rs)", color='blue', row=1, col=1, )
        fig.update_yaxes(title_text="Pitch<br />Angle<br />(deg)", color='blue', row=2, col=1)
        fig.update_yaxes(title_text="Br (nT)", color='blue', row=3, col=1)
        fig.update_yaxes(title_text="Bt (nT)", color='blue', row=4, col=1)
        fig.update_yaxes(title_text="Bn (nT)", color='blue', row=5, col=1)
        fig.update_yaxes(title_text='Np (cm^-3)', color='blue', row=6, col=1)

        fig.update_yaxes(title_text="Carr. Longitude<br />(deg)", color='red', row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Vr (km/s)", color='red', row=3, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Vt (km/s)", color='red', row=4, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Vn (km/s)", color='red', row=5, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Tp (eV)", color='red', row=6, col=1, secondary_y=True)

        fig.update_layout(title=dict(text="Encounter " + Encounter_str + " Overview",
                                     y=0.95, x=0.45,
                                     xanchor='center',
                                     yanchor='top', font=dict(size=20)),
                          template='simple_white', )
        filename = 'figures/overviews/Overview_' + mag_type + '_(' + beg_time_str + '-' + end_time_str + ').html'
        # fig.add_annotation(x=0.5, y=-.17, text='test_annotation', xref='paper', yref='paper', showarrow=False)
        py.plot(fig, filename='figures/overviews/Overview(' + beg_time_str + '-' + end_time_str + ').html')

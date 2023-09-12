import matplotlib.transforms

from load_psp_data import load_spi_data, load_RTN_data, load_RTN_1min_data, load_RTN_4sa_data, load_spe_data
from datetime import datetime, timedelta
import numpy as np
from plot_body_positions import get_rlonlat_psp_carr, get_rlonlat_solo_carr, get_rlonlat_earth_carr
from wind_overview import read_mag_data_wind, read_pmom_data_wind, read_epad_data_wind
from ace_overview import read_mag_data_ace, read_pmom_data_ace
from sta_overview import read_mag_data_sta, read_pmom_data_sta
import matplotlib.pyplot as plt
import pyspedas
from spacepy import pycdf
import os

os.environ['SOLO_DATA_DIR'] = '/Users/ephe/SolO_Data_Analysis'
Rs = 696300
AU = 1.5e8


# def read_mag_data_wind(beg_time,end_time,mag_type=''):
#     beg_time_str = beg_time.strftime('%Y-%m-%d/%H:%M:%S')
#     end_time_str = end_time.strftime('%Y-%m-%d/%H:%M:%S')
#     filelist = pyspedas.wind.mfi(trange=[beg_time_str,end_time_str],datatype='h3-rtn',downloadonly=True)
#     mag_wind = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
#     epochmag = mag_wind['Epoch']
#     timebinmag = (epochmag > beg_time) & (epochmag < end_time)
#     epochmag = epochmag[timebinmag]
#     Br = mag_wind['BRTN'][timebinmag,0]
#     Bt = mag_wind['BRTN'][timebinmag,1]
#     Bn = mag_wind['BRTN'][timebinmag,2]
#     Babs = np.sqrt(Br**2+Bt**2+Bn**2)
#     return epochmag,Br,Bt,Bn,Babs

# def read_pmom_data_wind(beg_time,end_time):
#     beg_time_str = beg_time.strftime('%Y-%m-%d/%H:%M:%S')
#     end_time_str = end_time.strftime('%Y-%m-%d/%H:%M:%S')
#     filelist = pyspedas.wind.threedp(trange=[beg_time_str, end_time_str], datatype='3dp_plsp', downloadonly=True)
#     pmom_wind = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
#     epochpmom = pmom_wind['Epoch']
#     timebinpmom = (epochpmom > beg_time) & (epochpmom < end_time)
#     epochpmom = epochpmom[timebinpmom]
#     densp = pmom_wind['MOM.P.DENSITY'][timebinpmom]
#     vr = -pmom_wind['MOM.P.VELOCITY'][timebinpmom,0]
#     vt = -pmom_wind['MOM.P.VELOCITY'][timebinpmom,1]
#     vn = pmom_wind['MOM.P.VELOCITY'][timebinpmom,2]
#     T = pmom_wind['MOM.P.AVGTEMP'][timebinpmom]
#     return epochpmom,densp,vr,vt,vn,T

# def read_epad_data_wind(beg_time,end_time):
#     beg_time_str = beg_time.strftime('%Y-%m-%d/%H:%M:%S')
#     end_time_str = end_time.strftime('%Y-%m-%d/%H:%M:%S')
#     filelist = pyspedas.wind.swe(trange=[beg_time_str, end_time_str], datatype='h3', downloadonly=True)
#     pad_wind = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
#     epochpad = pad_wind['Epoch']
#     timebinpad = (epochpad > beg_time) & (epochpad < end_time)
#     epochpad = epochpad[timebinpad]
#     pad_e6 = pad_wind['f_pitch_E06']
#     ene_e6 = '193.4 eV'
#     pa = np.linspace(0,180,30)
#
#     return epochpad,ene_e6,pa,pad_e6

# def read_mag_data_ace(beg_time,end_time):
#     beg_time_str = beg_time.strftime('%Y-%m-%d/%H:%M:%S')
#     end_time_str = end_time.strftime('%Y-%m-%d/%H:%M:%S')
#     filelist = pyspedas.ace.mfi(trange=[beg_time_str, end_time_str], datatype='h0',downloadonly=True)
#     mag_ace = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
#     epochmag = mag_ace['Epoch']
#     timebinmag = (epochmag > beg_time) & (epochmag < end_time)
#     epochmag = epochmag[timebinmag]
#     Br = -mag_ace['BGSEc'][timebinmag,0]
#     Bt = -mag_ace['BGSEc'][timebinmag,1]
#     Bn = mag_ace['BGSEc'][timebinmag,2]
#     Babs = np.sqrt(Br**2+Bt**2+Bn**2)
#     return epochmag,Br,Bt,Bn,Babs
#
# def read_pmom_data_ace(beg_time,end_time):
#     beg_time_str = beg_time.strftime('%Y-%m-%d/%H:%M:%S')
#     end_time_str = end_time.strftime('%Y-%m-%d/%H:%M:%S')
#     filelist = pyspedas.ace.swe(trange=[beg_time_str, end_time_str], datatype='k0', downloadonly=True)
#     pmom_ace = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
#     epochpmom = pmom_ace['Epoch']
#     timebinpmom = (epochpmom > beg_time) & (epochpmom < end_time)
#     epochpmom = epochpmom[timebinpmom]
#     densp = pmom_ace['Np'][timebinpmom]
#     vp = pmom_ace['Vp'][timebinpmom]
#     T = pmom_ace['Tpr'][timebinpmom]
#     return epochpmom,densp,vp,T


def read_mag_data_solo(beg_time, end_time, mag_type='rtn-normal'):
    beg_time_str = beg_time.strftime('%Y-%m-%d/%H:%M:%S')
    end_time_str = end_time.strftime('%Y-%m-%d/%H:%M:%S')
    filelist = pyspedas.solo.mag(trange=[beg_time_str, end_time_str], datatype=mag_type, downloadonly=True)
    mag_RTN = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])

    epochmag = mag_RTN['EPOCH']
    timebinmag = (epochmag > beg_time) & (epochmag < end_time)
    epochmag = epochmag[timebinmag]
    Br = mag_RTN['B_RTN'][timebinmag, 0]
    Bt = mag_RTN['B_RTN'][timebinmag, 1]
    Bn = mag_RTN['B_RTN'][timebinmag, 2]
    Babs = np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2)

    return epochmag, Br, Bt, Bn, Babs


def read_swa_data_solo(beg_time, end_time, swa_type='pas-grnd-mom'):
    beg_time_str = beg_time.strftime('%Y-%m-%d/%H:%M:%S')
    end_time_str = end_time.strftime('%Y-%m-%d/%H:%M:%S')
    filelist = pyspedas.solo.swa(trange=[beg_time_str, end_time_str], datatype=swa_type, downloadonly=True)
    pmom_SWA = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])

    epochpmom = pmom_SWA['Epoch']
    timebinpmom = (epochpmom > beg_time) & (epochpmom < end_time)
    epochpmom = epochpmom[timebinpmom]
    dens = pmom_SWA['N'][timebinpmom]
    vr = pmom_SWA['V_RTN'][timebinpmom, 0]
    vt = pmom_SWA['V_RTN'][timebinpmom, 1]
    vn = pmom_SWA['V_RTN'][timebinpmom, 2]
    T = pmom_SWA['T'][timebinpmom]

    return epochpmom, dens, vr, vt, vn, T


def read_mag_data_psp(beg_time, end_time, mag_type='1min'):
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
    return epochmag, Br, Bt, Bn, Babs


def read_spi_data_psp(beg_time, end_time, is_alpha=False, is_inst=False):
    if is_inst and not is_alpha:
        # load proton moms
        pmom_SPI = load_spi_data(beg_time.strftime('%Y%m%d'), end_time.strftime('%Y%m%d'), inst=is_inst, species='0')
        epochpmom = pmom_SPI['Epoch']
        timebinpmom = (epochpmom > beg_time) & (epochpmom < end_time)
        epochmom = epochpmom[timebinpmom]
        dens = pmom_SPI['DENS'][timebinpmom]
        v_r = pmom_SPI['VEL'][timebinpmom, 0]
        v_t = pmom_SPI['VEL'][timebinpmom, 1]
        v_n = pmom_SPI['VEL'][timebinpmom, 2]
        T = pmom_SPI['TEMP'][timebinpmom]
    elif is_inst and is_alpha:
        # load alpha moms
        amom_SPI = load_spi_data(beg_time.strftime('%Y%m%d'), end_time.strftime('%Y%m%d'), inst=is_inst, species='a')
        epochamom = amom_SPI['Epoch']
        timebinamom = (epochamom > beg_time) & (epochamom < end_time)
        epochmom = epochamom[timebinamom]
        dens = amom_SPI['DENS'][timebinamom]
        v_r = amom_SPI['VEL'][timebinamom, 0]
        v_t = amom_SPI['VEL'][timebinamom, 1]
        v_n = amom_SPI['VEL'][timebinamom, 2]
        T = amom_SPI['TEMP'][timebinamom]
    elif not is_inst and not is_alpha:
        # load proton moms
        pmom_SPI = load_spi_data(beg_time.strftime('%Y%m%d'), end_time.strftime('%Y%m%d'), inst=is_inst, species='0')
        epochpmom = pmom_SPI['Epoch']
        timebinpmom = (epochpmom > beg_time) & (epochpmom < end_time)
        epochmom = epochpmom[timebinpmom]
        dens = pmom_SPI['DENS'][timebinpmom]
        v_r = pmom_SPI['VEL_RTN_SUN'][timebinpmom, 0]
        v_t = pmom_SPI['VEL_RTN_SUN'][timebinpmom, 1]
        v_n = pmom_SPI['VEL_RTN_SUN'][timebinpmom, 2]
        T = pmom_SPI['TEMP'][timebinpmom]
    elif not is_inst and is_alpha:
        # load alpha moms
        amom_SPI = load_spi_data(beg_time.strftime('%Y%m%d'), end_time.strftime('%Y%m%d'), inst=is_inst, species='a')
        epochamom = amom_SPI['Epoch']
        timebinamom = (epochamom > beg_time) & (epochamom < end_time)
        epochmom = epochamom[timebinamom]
        dens = amom_SPI['DENS'][timebinamom]
        v_r = amom_SPI['VEL_RTN_SUN'][timebinamom, 0]
        v_t = amom_SPI['VEL_RTN_SUN'][timebinamom, 1]
        v_n = amom_SPI['VEL_RTN_SUN'][timebinamom, 2]
        T = amom_SPI['TEMP'][timebinamom]
    return epochmom, dens, v_r, v_t, v_n, T


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


if __name__ == '__main__':
    # %%
    mag_type = '4sa'  # for mag file, '1min', '4sa', 'rtn'
    n_Encounter = 11

    if n_Encounter == 13:
        n_Cases = 2
        cases = list({} for i in range(n_Cases))
        # i=0
        # cases[0]['beg_time'] = datetime(2022, 9, 5, 14, 0, 0)
        # cases[0]['end_time'] = datetime(2022, 9, 6, 4, 0, 0)
        # cases[0]['mid_time'] = datetime(2022, 9, 5, 20, 30, 0, 0)
        # cases[0]['observer'] = 'psp'

        i = 0
        cases[0]['beg_time'] = datetime(2022, 9, 6, 17, 0, 0)
        cases[0]['end_time'] = datetime(2022, 9, 6, 18, 15, 0)
        cases[0]['mid_time'] = datetime(2022, 9, 6, 17, 31, 0, 0)
        cases[0]['observer'] = 'psp'

        # cases[2]['beg_time'] = datetime(2022, 8, 29, 1, 0, 0)
        # cases[2]['end_time'] = datetime(2022, 8, 29, 6, 0, 0)
        # cases[2]['mid_time'] = datetime(2022, 8, 29, 2, 45, 0, 0)
        # cases[2]['observer'] = 'psp'

        i = 1
        cases[i]['beg_time'] = datetime(2022, 9, 12, 7, 0, 0)
        cases[i]['end_time'] = datetime(2022, 9, 12, 21, 40, 0)
        cases[i]['mid_time'] = datetime(2022, 9, 12, 15, 37, 0)
        cases[i]['observer'] = 'psp'

        full_start_time = datetime(2022, 8, 25)
        full_stop_time = datetime(2022, 9, 15)

    elif n_Encounter == 12:
        n_Cases = 2
        cases = list({} for i in range(n_Cases))
        cases[0]['beg_time'] = datetime(2022, 6, 2, 17, 0, 0)
        cases[0]['end_time'] = datetime(2022, 6, 2, 18, 0, 0)
        cases[0]['mid_time'] = datetime(2022, 6, 2, 17, 33, 0)
        cases[0]['observer'] = 'psp'

        cases[1]['beg_time'] = datetime(2022, 6, 9, 6, 15, 0)
        cases[1]['end_time'] = datetime(2022, 6, 9, 9, 0, 0)
        cases[1]['mid_time'] = datetime(2022, 6, 9, 7, 32, 0)
        cases[1]['observer'] = 'psp'

        # cases[2]['beg_time'] = datetime(2022, 5, 20, 0, 0, 0)
        # cases[2]['end_time'] = datetime(2022, 5, 20, 12, 0, 0)
        # cases[2]['mid_time'] = datetime(2022, 5, 20, 5, 30, 0)
        # cases[2]['observer'] = 'psp'

        full_start_time = datetime(2022, 5, 20)
        full_stop_time = datetime(2022, 6, 10)

    elif n_Encounter == 11:
        n_Cases = 7
        cases = list({} for i in range(n_Cases))
        i = 1
        cases[i]['beg_time'] = datetime(2022, 2, 17, 13, 48, 0)
        cases[i]['end_time'] = datetime(2022, 2, 18, 1, 30, 0)
        cases[i]['mid_time'] = datetime(2022, 2, 17, 17, 10, 0)
        cases[i]['observer'] = 'psp'
        i = 0
        cases[i]['beg_time'] = datetime(2022, 2, 25, 12, 28, 0)
        cases[i]['end_time'] = datetime(2022, 2, 25, 12, 37, 0)
        cases[i]['mid_time'] = datetime(2022, 2, 25, 12, 33, 40)
        cases[i]['observer'] = 'psp'

        cases[2]['beg_time'] = datetime(2022, 3, 10, 18, 0, 0)
        cases[2]['end_time'] = datetime(2022, 3, 11, 18, 0, 0)
        cases[2]['mid_time'] = datetime(2022, 3, 11, 0, 30, 0)
        cases[2]['observer'] = 'psp'

        cases[3]['beg_time'] = datetime(2022, 2, 28, 10)
        cases[3]['end_time'] = datetime(2022, 3, 3, 2)
        cases[3]['mid_time'] = datetime(2022, 3, 1, 2, 20)
        cases[3]['observer'] = 'solo'

        cases[5]['beg_time'] = datetime(2022, 3, 3)
        cases[5]['end_time'] = datetime(2022, 3, 6)
        cases[5]['mid_time'] = datetime(2022, 3, 3, 16, 30)
        cases[5]['observer'] = 'wind'

        cases[6]['beg_time'] = datetime(2022, 3, 3)
        cases[6]['end_time'] = datetime(2022, 3, 6)
        cases[6]['mid_time'] = datetime(2022, 3, 3, 16, 30)
        cases[6]['observer'] = 'ace'

        cases[4]['beg_time'] = datetime(2022, 2, 28, 1)
        cases[4]['end_time'] = datetime(2022, 3, 3, 4)
        cases[4]['mid_time'] = datetime(2022, 2, 28, 19, 55)
        cases[4]['observer'] = 'sta'

        full_start_time = datetime(2022, 2, 16)
        full_stop_time = datetime(2022, 3, 12)

    elif n_Encounter == 10:
        n_Cases = 2
        cases = list({} for i in range(n_Cases))
        cases[0]['beg_time'] = datetime(2021, 11, 22, 0, 0, 0)
        cases[0]['end_time'] = datetime(2021, 11, 22, 4, 0, 0)
        cases[0]['mid_time'] = datetime(2021, 11, 22, 2, 8, 0)
        cases[0]['observer'] = 'psp'

        # cases[1]['beg_time'] = datetime(2021, 11, 22, 0, 0, 0)
        # cases[1]['end_time'] = datetime(2021, 11, 22, 4, 0, 0)
        # cases[1]['mid_time'] = datetime(2021, 11, 22, 2, 0, 0)
        # cases[1]['observer'] = 'psp'

        cases[1]['beg_time'] = datetime(2021, 11, 29, 19, 0, 0)
        cases[1]['end_time'] = datetime(2021, 11, 30, 23, 0, 0)
        cases[1]['mid_time'] = datetime(2021, 11, 30, 8, 0, 0)
        cases[1]['observer'] = 'psp'

        full_start_time = datetime(2021, 11, 10, 0)
        full_stop_time = datetime(2021, 12, 1, 0)

    elif n_Encounter == 9:
        n_Cases = 3
        cases = list({} for i in range(n_Cases))
        cases[1]['beg_time'] = datetime(2021, 8, 1, 12, 0, 0)
        cases[1]['end_time'] = datetime(2021, 8, 2, 20, 0, 0)
        cases[1]['mid_time'] = datetime(2021, 8, 2, 0, 30, 0)
        cases[1]['observer'] = 'psp'

        cases[0]['beg_time'] = datetime(2021, 8, 10, 0, 0, 0)
        cases[0]['end_time'] = datetime(2021, 8, 10, 2, 30, 0)
        cases[0]['mid_time'] = datetime(2021, 8, 10, 1, 0, 0)
        cases[0]['observer'] = 'psp'

        cases[2]['beg_time'] = datetime(2021, 8, 21, 12, 0, 0)
        cases[2]['end_time'] = datetime(2021, 8, 23, 6, 0, 0)
        cases[2]['mid_time'] = datetime(2021, 8, 22, 4, 30, 0)
        cases[2]['observer'] = 'psp'

        full_start_time = datetime(2021, 8, 1)
        full_stop_time = datetime(2021, 8, 24)

    elif n_Encounter == 8:
        n_Cases = 2
        cases = list({} for i in range(n_Cases))
        cases[0]['beg_time'] = datetime(2021, 4, 29, 0, 0, 0)
        cases[0]['end_time'] = datetime(2021, 4, 29, 12, 0, 0)
        cases[0]['mid_time'] = datetime(2021, 4, 29, 8, 30, 0)
        cases[0]['observer'] = 'psp'

        cases[1]['beg_time'] = datetime(2021, 5, 12, 11, 30, 0)
        cases[1]['end_time'] = datetime(2021, 5, 12, 18, 0, 0)
        cases[1]['mid_time'] = datetime(2021, 5, 12, 13, 35, 0)
        cases[1]['observer'] = 'psp'

        # cases[2]['beg_time'] = datetime(2021, 4, 24, 0, 0, 0)
        # cases[2]['end_time'] = datetime(2021, 4, 25, 22, 0, 0)
        # cases[2]['mid_time'] = datetime(2021, 4, 24, 12, 0, 0)
        # cases[2]['observer'] = 'psp'

        full_start_time = datetime(2021, 4, 18)
        full_stop_time = datetime(2021, 5, 13)

    elif n_Encounter == 7:
        n_Cases = 3
        cases = list({} for i in range(n_Cases))
        cases[0]['beg_time'] = datetime(2021, 1, 17, 10, 0, 0)
        cases[0]['end_time'] = datetime(2021, 1, 17, 18, 0, 0)
        cases[0]['mid_time'] = datetime(2021, 1, 17, 14, 30, 0)
        cases[0]['observer'] = 'psp'

        cases[1]['beg_time'] = datetime(2021, 1, 12, 12, 0, 0)
        cases[1]['end_time'] = datetime(2021, 1, 13, 12, 0, 0)
        cases[1]['mid_time'] = datetime(2021, 1, 12, 19, 0, 0)
        cases[1]['observer'] = 'psp'

        cases[2]['beg_time'] = datetime(2021, 1, 22, 14, 0, 0)
        cases[2]['end_time'] = datetime(2021, 1, 24, 20, 0, 0)
        cases[2]['mid_time'] = datetime(2021, 1, 25, 10, 0, 0)
        cases[2]['observer'] = 'psp'

        full_start_time = datetime(2021, 1, 12)
        full_stop_time = datetime(2021, 1, 25)

    # %%
    for i in range(n_Cases):
        if cases[i]['observer'] == 'psp':
            cases[i]['epochmag'], cases[i]['Br'], cases[i]['Bt'], cases[i]['Bn'], cases[i]['Babs'] \
                = read_mag_data_psp(cases[i]['beg_time'], cases[i]['end_time'], mag_type='4sa')
            cases[i]['epochpmom'], cases[i]['densp'], cases[i]['vp_r'], cases[i]['vt_r'], cases[i]['vn_r'], cases[i][
                'Tp'] \
                = read_spi_data_psp(cases[i]['beg_time'], cases[i]['end_time'], is_alpha=False, is_inst=False)
            cases[i]['r_sc_carr_mag'], cases[i]['lon_sc_carr_mag'], cases[i]['lat_sc_carr_mag'] \
                = get_rlonlat_psp_carr(cases[i]['epochmag'], for_psi=False)
        elif cases[i]['observer'] == 'solo':
            cases[i]['epochmag'], cases[i]['Br'], cases[i]['Bt'], cases[i]['Bn'], cases[i]['Babs'] \
                = read_mag_data_solo(cases[i]['beg_time'], cases[i]['end_time'], mag_type='rtn-normal')
            cases[i]['epochpmom'], cases[i]['densp'], cases[i]['vp_r'], cases[i]['vt_r'], cases[i]['vn_r'], cases[i][
                'Tp'] \
                = read_swa_data_solo(cases[i]['beg_time'], cases[i]['end_time'], swa_type='pas-grnd-mom')
            cases[i]['r_sc_carr_mag'], cases[i]['lon_sc_carr_mag'], cases[i]['lat_sc_carr_mag'] \
                = get_rlonlat_solo_carr(cases[i]['epochmag'], for_psi=False)
        elif cases[i]['observer'] == 'wind':
            cases[i]['epochmag'], cases[i]['Br'], cases[i]['Bt'], cases[i]['Bn'], cases[i]['Babs'] \
                = read_mag_data_wind(cases[i]['beg_time'], cases[i]['end_time'])
            cases[i]['epochpmom'], cases[i]['densp'], cases[i]['vp_r'], cases[i]['vp_t'], cases[i]['vp_n'], cases[i][
                'Tp'] \
                = read_pmom_data_wind(cases[i]['beg_time'], cases[i]['end_time'])
            cases[i]['r_sc_carr_mag'], cases[i]['lon_sc_carr_mag'], cases[i]['lat_sc_carr_mag'] \
                = get_rlonlat_earth_carr(cases[i]['epochmag'], for_psi=False)
        elif cases[i]['observer'] == 'ace':
            cases[i]['epochmag'], cases[i]['Br'], cases[i]['Bt'], cases[i]['Bn'], cases[i]['Babs'] \
                = read_mag_data_ace(cases[i]['beg_time'], cases[i]['end_time'])
            cases[i]['epochpmom'], cases[i]['densp'], cases[i]['vp_r'], cases[i]['Tp'] \
                = read_pmom_data_ace(cases[i]['beg_time'], cases[i]['end_time'])
            cases[i]['vp_t'] = cases[i]['vp_r'] * 0.
            cases[i]['vp_n'] = cases[i]['vp_r'] * 0.
            cases[i]['r_sc_carr_mag'], cases[i]['lon_sc_carr_mag'], cases[i]['lat_sc_carr_mag'] \
                = get_rlonlat_earth_carr(cases[i]['epochmag'], for_psi=False)
        elif cases[i]['observer'] == 'sta':
            cases[i]['epochmag'], cases[i]['Br'], cases[i]['Bt'], cases[i]['Bn'], cases[i]['Babs'] \
                = read_mag_data_sta(cases[i]['beg_time'], cases[i]['end_time'])

            cases[i]['epochpmom'], cases[i]['densp'], cases[i]['vp_r'], cases[i]['vp_t'], cases[i]['vp_n'], cases[i][
                'Tp'], cases[i]['r_sc_carr_pmom'], cases[i]['lon_sc_carr_pmom'], cases[i]['lat_sc_carr_pmom'] \
                = read_pmom_data_sta(cases[i]['beg_time'], cases[i]['end_time'])

    # %%
    # i = 5
    # fig, axs = plt.subplots(5,1,sharex=True)
    #
    # axs[0].plot(cases[i]['epochmag'], cases[i]['r_sc_carr_mag'], 'k-',linewidth=1)
    # axs[0].set_ylabel('Radial \n Distance (Rs)')
    # ax2 = axs[0].twinx()
    # ax2.plot(cases[i]['epochmag'],np.rad2deg(cases[i]['lon_sc_carr_mag']),'r-',linewidth=1)
    # ax2.set_ylabel('Carrington \n Longitude (deg)',color='r')
    #
    # if cases[i]['observer']=='wind':
    #     epochpad,ene,pa,pad = read_epad_data_wind(cases[i]['beg_time'],cases[i]['end_time'])
    #     pad_ax = axs[1].pcolormesh(epochpad, pa, np.log10(np.array(pad)).T, cmap='jet')
    #     axs[1].set_ylabel('Pitch Angle \n (deg)')
    #     pad_axpos = axs[1].get_position()
    #     pad_caxpos = matplotlib.transforms.Bbox.from_extents(
    #         pad_axpos.x1 + 0.01, pad_axpos.y0, pad_axpos.x1 + 0.01 + 0.01, pad_axpos.y1)
    #     cax = axs[1].figure.add_axes(pad_caxpos)
    #     cbar = plt.colorbar(pad_ax, cax=cax)
    # elif cases[i]['observer']=='ace':
    #     axs[1].text(0.5,0.5,'NO DATA')
    #
    # axs[2].plot(cases[i]['epochmag'], cases[i]['Babs'], 'm-',linewidth=1,label='Btot')
    # axs[2].plot(cases[i]['epochmag'], cases[i]['Br'],'k-',linewidth=1,label='Br')
    # axs[2].set_ylim([-10,10])
    # axs[2].set_ylabel('$B_{r,tot}$\n$(nT))')
    # ax2 = axs[2].twinx()
    # ax2.plot(cases[i]['epochpmom'],cases[i]['vp_r'],'r-',linewidth=1)
    # ax2.set_ylabel('$V_{pr}$\n$(km/s)$',color='r')
    #
    #
    # axs[3].plot(cases[i]['epochmag'], cases[i]['Bt'], 'r-', linewidth=1, label='Bt')
    # axs[3].plot(cases[i]['epochmag'], cases[i]['Bn'], 'b-', linewidth=1, label='Bn')
    # axs[3].set_ylim([-10, 10])
    # axs[3].set_ylabel('$B_{tn}$\n$(nT))')
    #
    # axs[4].plot(cases[i]['epochpmom'],cases[i]['densp'],'k-',linewidth=1,label='Np')
    # axs[4].set_ylabel('$N_p$\n$(cm^{-3})$')
    # axs[4].set_xlabel('Time')
    # ax2 = axs[4].twinx()
    # ax2.plot(cases[i]['epochpmom'],cases[i]['Tp'],'r-',linewidth=1,label='Tp')
    # ax2.set_ylabel('$T_p$\n$(eV)$',color='r')
    #
    # plt.show()

    # %%

    timestep = timedelta(hours=12)
    steps = (full_stop_time - full_start_time) // timestep + 1
    dttimes = np.array([x * timestep + full_start_time for x in range(steps)])
    dttimes_str = [dt.strftime('%m%dT%H') for dt in dttimes[0:-1:24]]

    full_r_psp, full_lon_psp, full_lat_psp = get_rlonlat_psp_carr(dttimes, for_psi=False)
    full_r_solo, full_lon_solo, full_lat_solo = get_rlonlat_solo_carr(dttimes, for_psi=False)
    full_r_earth, full_lon_earth, full_lat_earth = get_rlonlat_earth_carr(dttimes, for_psi=False)

    # %%
    r_ss = 2.5

    for i in range(n_Cases):
        ind_mag = np.argmin(np.abs((cases[i]['epochmag'] - cases[i]['mid_time']) / timedelta(seconds=1)))
        ind_pmom = np.argmin(np.abs((cases[i]['epochpmom'] - cases[i]['mid_time']) / timedelta(seconds=1)))
        if cases[i]['observer'] == 'sta':
            r_vect = np.linspace(cases[i]['r_sc_carr_pmom'][ind_pmom], r_ss, num=100)
            r_beg = cases[i]['r_sc_carr_pmom'][ind_pmom]
            lon_beg = np.rad2deg(cases[i]['lon_sc_carr_pmom'][ind_pmom])
            lat_beg = np.rad2deg(cases[i]['lat_sc_carr_pmom'][ind_pmom])
        else:
            r_vect = np.linspace(cases[i]['r_sc_carr_mag'][ind_mag], r_ss, num=100)
            r_beg = cases[i]['r_sc_carr_mag'][ind_mag]
            lon_beg = np.rad2deg(cases[i]['lon_sc_carr_mag'][ind_mag])
            lat_beg = np.rad2deg(cases[i]['lat_sc_carr_mag'][ind_mag])

        vr_beg = cases[i]['vp_r'][ind_pmom]

        print(
            f'Start Point of Parker Spiral for Case {i}: '
            f'vr_beg = {vr_beg} km/s, r_beg = {r_beg} Rs, lon_beg = {lon_beg} deg, lat_beg= {lat_beg} deg.')
        cases[i]['ps_lon'], cases[i]['ps_lat'] = parker_spiral(r_vect * Rs / AU, lat_beg, lon_beg, r_vect * 0 + vr_beg)
        cases[i]['ps_r'] = r_vect
        cases[i]['ps_r_beg'] = r_beg
        cases[i]['ps_vr_beg'] = vr_beg

    # %%
    fig, axs = plt.subplots(n_Cases, 1)
    for i in range(n_Cases):
        if cases[i]['observer'] == 'sta':
            mid_lon = np.rad2deg(
                cases[i]['lon_sc_carr_pmom'][np.argmin(np.abs(cases[i]['epochpmom'] - cases[i]['mid_time']))])
            lon_mag = np.interp(
                np.array((cases[i]['epochmag'] - cases[i]['epochpmom'][0]) / timedelta(days=1), dtype='float64'),
                np.array((cases[i]['epochpmom'] - cases[i]['epochpmom'][0]) / timedelta(days=1), dtype='float64'),
                cases[i]['lon_sc_carr_pmom'])
            lon_pmom = cases[i]['lon_sc_carr_pmom']
            cases[i]['Br'][cases[i]['Br'] > 17.] = np.nan

            axs[i].plot(np.rad2deg(lon_mag), cases[i]['Br'])

            # axs[i].plot(np.rad2deg(lon_pmom),cases[i]['densp'])

        else:
            mid_lon = np.rad2deg(
                cases[i]['lon_sc_carr_mag'][np.argmin(np.abs(cases[i]['epochmag'] - cases[i]['mid_time']))])
            lon_mag = cases[i]['lon_sc_carr_mag']
            lon_pmom = np.interp(
                np.array((cases[i]['epochpmom'] - cases[i]['epochpmom'][0]) / timedelta(days=1), dtype='float64'),
                np.array((cases[i]['epochmag'] - cases[i]['epochpmom'][0]) / timedelta(days=1), dtype='float64'),
                cases[i]['lon_sc_carr_mag'])
            axs[i].plot(np.rad2deg(lon_mag), cases[i]['Br'])
            # axs[i].plot(np.rad2deg(lon_pmom),cases[i]['densp'])

        axs[i].plot([mid_lon, mid_lon], [np.nanmin(cases[i]['Br']), np.nanmax(cases[i]['Br'])], 'r--')
        axs[i].set_title('R~%d Rs' % cases[i]['ps_r_beg'])
        axs[i].set_ylabel('B_R [nT]')

        # axs[i].plot([mid_lon, mid_lon], [np.nanmin(cases[i]['densp']), np.nanmax(cases[i]['densp'])], 'r--')
        # axs[i].set_title('R~%d Rs' % cases[i]['ps_r_beg'])
        # axs[i].set_ylabel('Np ($cm^{-3}$)')
    # axs[n_Cases-1].set_xlabel('Carrington Longitude [deg]')
    axs[n_Cases - 1].set_xlabel('Carrington Longitude [deg]')

    plt.show()
    # %%
    fig, axs = plt.subplots(n_Cases, 1)
    for i in range(n_Cases):
        if cases[i]['observer'] == 'sta':
            mid_lon = np.rad2deg(
                cases[i]['lon_sc_carr_pmom'][np.argmin(np.abs(cases[i]['epochpmom'] - cases[i]['mid_time']))])
            lon_mag = np.interp(
                np.array((cases[i]['epochmag'] - cases[i]['epochpmom'][0]) / timedelta(days=1), dtype='float64'),
                np.array((cases[i]['epochpmom'] - cases[i]['epochpmom'][0]) / timedelta(days=1), dtype='float64'),
                cases[i]['lon_sc_carr_pmom'])
            lon_pmom = cases[i]['lon_sc_carr_pmom']

            # axs[i].plot(np.rad2deg(lon_mag),cases[i]['Br'])
            axs[i].plot(np.rad2deg(lon_pmom), cases[i]['vp_r'])

        else:
            mid_lon = np.rad2deg(
                cases[i]['lon_sc_carr_mag'][np.argmin(np.abs(cases[i]['epochmag'] - cases[i]['mid_time']))])
            lon_mag = cases[i]['lon_sc_carr_mag']
            lon_pmom = np.interp(
                np.array((cases[i]['epochpmom'] - cases[i]['epochpmom'][0]) / timedelta(days=1), dtype='float64'),
                np.array((cases[i]['epochmag'] - cases[i]['epochpmom'][0]) / timedelta(days=1), dtype='float64'),
                cases[i]['lon_sc_carr_mag'])
            # axs[i].plot(np.rad2deg(lon_mag), cases[i]['Br'])
            axs[i].plot(np.rad2deg(lon_pmom), cases[i]['vp_r'])

        # axs[i].plot([mid_lon, mid_lon], [np.nanmin(cases[i]['Br']), np.nanmax(cases[i]['Br'])], 'r--')
        # axs[i].set_title('R~%d Rs' % cases[i]['ps_r_beg'])
        # axs[i].set_ylabel('B_R [nT]')

        axs[i].plot([mid_lon, mid_lon], [np.nanmin(cases[i]['vp_r']), np.nanmax(cases[i]['vp_r'])], 'r--')
        axs[i].set_title('R~%d Rs' % cases[i]['ps_r_beg'])
        axs[i].set_ylabel('Vp ($km/s$)')
    # axs[n_Cases - 1].set_xlabel('Carrington Longitude [deg]')
    axs[n_Cases - 1].set_xlabel('Carrington Longitude [deg]')

    plt.show()

    # %%
    # fig, axs = plt.subplots(3, 1)
    # axs[0].quiver(np.rad2deg(lon_psp_carr_mag_C2), np.rad2deg(lat_psp_carr_mag_C2), Bt_C2, Bn_C2, np.sign(Br_C2))
    #
    # axs[1].quiver(np.rad2deg(lon_psp_carr_mag_C1), np.rad2deg(lat_psp_carr_mag_C1), Bt_C1, Bn_C1, np.sign(Br_C1))
    #
    # axs[2].quiver(np.rad2deg(lon_psp_carr_mag_C3), np.rad2deg(lat_psp_carr_mag_C3), Bt_C3, Bn_C3, np.sign(Br_C3))
    #
    # plt.show()

    # %%
    clist = ['r', 'g', 'b', 'm', 'c', 'y', 'k']
    plt.figure()
    ax = plt.subplot(111, projection='polar')
    ax.plot(full_lon_psp, full_r_psp * np.cos(full_lat_psp), 'k', label='PSP orbit', linewidth=2, alpha=0.5)
    ax.plot(full_lon_solo, full_r_solo * np.cos(full_lat_solo), 'k', label='SO orbit', linewidth=2, alpha=0.5)
    ax.plot(full_lon_earth, full_r_earth * np.cos(full_lat_earth), 'k', label='EARTH orbit', linewidth=2, alpha=0.5)

    for i in range(n_Cases):
        if cases[i]['observer'] == 'sta':
            ax.plot(cases[i]['lon_sc_carr_pmom'], cases[i]['r_sc_carr_pmom'] * np.cos(cases[i]['lat_sc_carr_pmom']),
                    clist[i], label=f'Case {i + 1} [~%d Rs]' % cases[i]['ps_r_beg'])
        else:
            ax.plot(cases[i]['lon_sc_carr_mag'], cases[i]['r_sc_carr_mag'] * np.cos(cases[i]['lat_sc_carr_mag']),
                    clist[i], label=f'Case {i + 1} [~%d Rs]' % cases[i]['ps_r_beg'])
        ax.plot(np.deg2rad(cases[i]['ps_lon']), cases[i]['ps_r'] * np.cos(np.deg2rad(cases[i]['ps_lat'])),
                clist[i] + '--',
                label=r'$V_{beg}=%d km/s$' % cases[i]['ps_vr_beg'])
    ax.set_title('Carrington Coordinates')
    # ax.set_rmax(140)
    # ax.set_thetamin(30)
    # ax.set_thetamax(210)
    # ax.set_theta_offset(-np.pi/6)
    plt.legend(ncol=2)
    plt.show()

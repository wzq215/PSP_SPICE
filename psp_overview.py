from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
# import pyspedas
from spacepy import pycdf
from plot_body_positions import get_rlonlat_psp_carr
from load_psp_data import load_spi_data, load_RTN_data, load_RTN_1min_data, load_RTN_4sa_data, load_spe_data
import os
import matplotlib.transforms

Rs = 696300
AU = 1.5e8


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


def read_pad_data_psp(beg_time, end_time):
    spe_pad = load_spe_data(beg_time.strftime('%Y%m%d'), end_time.strftime('%Y%m%d'))
    epochpade = spe_pad['Epoch']
    timebinpade = (epochpade > beg_time) & (epochpade < end_time)
    epochpade = epochpade[timebinpade]
    EfluxVsPAE = spe_pad['EFLUX_VS_PA_E'][timebinpade, :, :]
    PitchAngle = spe_pad['PITCHANGLE'][timebinpade, :]
    Energy_val = spe_pad['ENERGY_VALS'][timebinpade, :]
    # norm_EfluxVsPAE = EfluxVsPAE * 0
    # EfluxVsE = spe_pad['EFLUX_VS_ENERGY'][timebinpad]
    i_energy = 13
    ene = '%d' % Energy_val[0, i_energy] + ' eV'
    pa = PitchAngle[0][:]
    pad = EfluxVsPAE[:, :, i_energy]

    return epochpade, ene, pa, pad


if __name__ == '__main__':
    n_Cases = 1
    cases = list({} for i in range(n_Cases))
    i = 0
    # cases[i]['beg_time'] = datetime(2022, 2, 17, 13, 48, 0)
    # cases[i]['end_time'] = datetime(2022, 2, 18, 1, 40, 0)
    # cases[i]['mid_time'] = datetime(2022, 2, 17, 17, 7, 0)
    # cases[i]['observer'] = 'psp'

    # cases[i]['beg_time'] = datetime(2022, 2, 25, 12, 26, 0)
    # cases[i]['end_time'] = datetime(2022, 2, 25, 12, 37, 0)
    # cases[i]['mid_time'] = datetime(2022, 2, 25, 12, 33, 40)
    # cases[i]['observer'] = 'psp'
    cases[i]['beg_time'] = datetime(2022, 3, 10, 11, 50, 0)
    cases[i]['end_time'] = datetime(2022, 3, 11, 23, 59, 0)
    cases[i]['mid_time'] = datetime(2022, 3, 10, 22, 30, 0)
    cases[i]['observer'] = 'psp'

    cases[i]['epochmag'], cases[i]['Br'], cases[i]['Bt'], cases[i]['Bn'], cases[i]['Babs'] \
        = read_mag_data_psp(cases[i]['beg_time'], cases[i]['end_time'], mag_type='1min')
    cases[i]['epochpmom'], cases[i]['densp'], cases[i]['vp_r'], cases[i]['vp_t'], cases[i]['vp_n'], cases[i]['Tp'] \
        = read_spi_data_psp(cases[i]['beg_time'], cases[i]['end_time'])
    cases[i]['r_sc_carr_mag'], cases[i]['lon_sc_carr_mag'], cases[i]['lat_sc_carr_mag'] \
        = get_rlonlat_psp_carr(cases[i]['epochmag'], for_psi=False)
    epochpad, ene, pa, pad = read_pad_data_psp(cases[i]['beg_time'], cases[i]['end_time'])
    # %%
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(7, 7))
    plt.subplots_adjust(top=0.95, bottom=0.1, right=0.85, left=0.15, hspace=0, wspace=0)

    axs[0].plot(cases[i]['epochmag'], cases[i]['r_sc_carr_mag'], 'k-', linewidth=1)
    axs[0].set_ylabel('Radial \n Distance (Rs)')
    ax2 = axs[0].twinx()
    ax2.plot(cases[i]['epochmag'], np.rad2deg(cases[i]['lon_sc_carr_mag']), 'r-', linewidth=1)
    ax2.set_ylabel('Carrington \n Longitude (deg)', color='r')

    pad_ax = axs[1].pcolormesh(epochpad, pa, np.log10(np.array(pad)).T, cmap='jet')
    axs[1].set_ylabel('Pitch Angle \n (deg)')
    pad_axpos = axs[1].get_position()
    pad_caxpos = matplotlib.transforms.Bbox.from_extents(
        pad_axpos.x1 + 0.01, pad_axpos.y0, pad_axpos.x1 + 0.01 + 0.01, pad_axpos.y1)
    cax = axs[1].figure.add_axes(pad_caxpos)
    cbar = plt.colorbar(pad_ax, cax=cax, label='E~' + ene)

    axs[2].plot(cases[i]['epochmag'], cases[i]['Babs'], 'm-', linewidth=1, label='Btot')
    axs[2].plot(cases[i]['epochmag'], cases[i]['Br'], 'k-', linewidth=1, label='Br')
    # axs[2].set_ylim([-10, 14])
    axs[2].set_ylabel('$B_{r,tot}$\n (nT)')
    axs[2].legend(loc='upper left')
    ax2 = axs[2].twinx()
    # cases[i]['vp_r'][cases[i]['vp_r']<=0.] =np.nan
    ax2.plot(cases[i]['epochpmom'], cases[i]['vp_r'], 'r-', linewidth=1)
    ax2.set_ylabel('$V_{pr}$\n$(km/s)$', color='r')

    axs[3].plot(cases[i]['epochmag'], cases[i]['Bt'], 'r-', linewidth=1, label='Bt')
    axs[3].plot(cases[i]['epochmag'], cases[i]['Bn'], 'b-', linewidth=1, label='Bn')
    axs[3].legend(loc=2, bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
    # axs[3].set_ylim([-13, 12])
    axs[3].set_ylabel('$B_{tn}$\n (nT)')

    # cases[i]['densp'][cases[i]['densp'] <= 0.] = np.nan
    # cases[i]['Tp'][cases[i]['Tp'] <= 0.] = np.nan

    axs[4].plot(cases[i]['epochpmom'], cases[i]['densp'], 'k-', linewidth=1, label='Np')
    axs[4].set_ylabel('$N_p$\n$(cm^{-3})$')
    axs[4].set_xlabel('Time')
    ax2 = axs[4].twinx()
    ax2.plot(cases[i]['epochpmom'], cases[i]['Tp'], 'r-', linewidth=1, label='Tp')
    ax2.set_ylabel('$T_p$\n$(eV)$', color='r')
    plt.suptitle('PSP Overview')

    plt.show()

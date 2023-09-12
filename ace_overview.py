from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pyspedas
from spacepy import pycdf
from plot_body_positions import get_rlonlat_earth_carr
import os
import matplotlib.transforms

os.environ['SOLO_DATA_DIR'] = '/Users/ephe/SolO_Data_Analysis'
Rs = 696300
AU = 1.5e8


def read_mag_data_ace(beg_time, end_time):
    beg_time_str = beg_time.strftime('%Y-%m-%d/%H:%M:%S')
    end_time_str = end_time.strftime('%Y-%m-%d/%H:%M:%S')
    filelist = pyspedas.ace.mfi(trange=[beg_time_str, end_time_str], datatype='h0', downloadonly=True)
    mag_ace = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    epochmag = mag_ace['Epoch']
    timebinmag = (epochmag > beg_time) & (epochmag < end_time)
    epochmag = epochmag[timebinmag]
    Br = -mag_ace['BGSEc'][timebinmag, 0]
    Bt = -mag_ace['BGSEc'][timebinmag, 1]
    Bn = mag_ace['BGSEc'][timebinmag, 2]
    Babs = np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2)
    return epochmag, Br, Bt, Bn, Babs


def read_pmom_data_ace(beg_time, end_time):
    beg_time_str = beg_time.strftime('%Y-%m-%d/%H:%M:%S')
    end_time_str = end_time.strftime('%Y-%m-%d/%H:%M:%S')
    filelist = pyspedas.ace.swe(trange=[beg_time_str, end_time_str], datatype='k0', downloadonly=True)
    pmom_ace = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    epochpmom = pmom_ace['Epoch']
    timebinpmom = (epochpmom > beg_time) & (epochpmom < end_time)
    epochpmom = epochpmom[timebinpmom]
    densp = pmom_ace['Np'][timebinpmom]
    vp = pmom_ace['Vp'][timebinpmom]
    T = pmom_ace['Tpr'][timebinpmom] * 8.617e-5
    densp[densp <= 0.] = np.nan
    vp[vp <= 0.] = np.nan
    T[T <= 0.] = np.nan
    return epochpmom, densp, vp, T


if __name__ == '__main__':
    n_Cases = 1
    cases = list({} for i in range(n_Cases))
    i = 0
    cases[i]['beg_time'] = datetime(2022, 3, 2)
    cases[i]['end_time'] = datetime(2022, 3, 7)
    cases[i]['mid_time'] = datetime(2022, 3, 3, 14)
    cases[i]['observer'] = 'wind'

    cases[i]['epochmag'], cases[i]['Br'], cases[i]['Bt'], cases[i]['Bn'], cases[i]['Babs'] \
        = read_mag_data_ace(cases[i]['beg_time'], cases[i]['end_time'])
    cases[i]['epochpmom'], cases[i]['densp'], cases[i]['vp_r'], cases[i]['Tp'] \
        = read_pmom_data_ace(cases[i]['beg_time'], cases[i]['end_time'])
    cases[i]['r_sc_carr_mag'], cases[i]['lon_sc_carr_mag'], cases[i]['lat_sc_carr_mag'] \
        = get_rlonlat_earth_carr(cases[i]['epochmag'], for_psi=False)

    # %%
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(7, 7))
    plt.subplots_adjust(top=0.95, bottom=0.1, right=0.85, left=0.15, hspace=0, wspace=0)

    axs[0].plot(cases[i]['epochmag'], cases[i]['r_sc_carr_mag'], 'k-', linewidth=1)
    axs[0].set_ylabel('Radial \n Distance (Rs)')
    ax2 = axs[0].twinx()
    ax2.plot(cases[i]['epochmag'], np.rad2deg(cases[i]['lon_sc_carr_mag']), 'r-', linewidth=1)
    ax2.set_ylabel('Carrington \n Longitude (deg)', color='r')

    axs[1].text(0.5, 0.5, 'NO DATA')

    axs[2].plot(cases[i]['epochmag'], cases[i]['Babs'], 'm-', linewidth=1, label='Btot')
    axs[2].plot(cases[i]['epochmag'], cases[i]['Br'], 'k-', linewidth=1, label='Br')
    axs[2].set_ylim([-10, 14])
    axs[2].set_ylabel('$B_{r,tot}$\n (nT)')
    axs[2].legend(loc='upper left')
    ax2 = axs[2].twinx()
    # cases[i]['vp_r'][cases[i]['vp_r']<=0.] =np.nan
    ax2.plot(cases[i]['epochpmom'], cases[i]['vp_r'], 'r-', linewidth=1)
    ax2.set_ylabel('$V_{pr}$\n$(km/s)$', color='r')

    axs[3].plot(cases[i]['epochmag'], cases[i]['Bt'], 'r-', linewidth=1, label='Bt')
    axs[3].plot(cases[i]['epochmag'], cases[i]['Bn'], 'b-', linewidth=1, label='Bn')
    axs[3].legend(loc=2, bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
    axs[3].set_ylim([-13, 12])
    axs[3].set_ylabel('$B_{tn}$\n (nT)')

    # cases[i]['densp'][cases[i]['densp'] <= 0.] = np.nan
    # cases[i]['Tp'][cases[i]['Tp'] <= 0.] = np.nan

    axs[4].plot(cases[i]['epochpmom'], cases[i]['densp'], 'k-', linewidth=1, label='Np')
    axs[4].set_ylabel('$N_p$\n$(cm^{-3})$')
    axs[4].set_xlabel('Time')
    ax2 = axs[4].twinx()
    ax2.plot(cases[i]['epochpmom'], cases[i]['Tp'], 'r-', linewidth=1, label='Tp')
    ax2.set_ylabel('$T_p$\n$(eV)$', color='r')
    plt.suptitle('ACE Overview')

    plt.show()

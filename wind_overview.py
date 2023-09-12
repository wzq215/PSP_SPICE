from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pyspedas.wind
from spacepy import pycdf
from plot_body_positions import get_rlonlat_earth_carr
import os
import matplotlib.transforms

os.environ['SOLO_DATA_DIR'] = '/Users/ephe/SolO_Data_Analysis'
Rs = 696300
AU = 1.5e8


def read_mag_data_wind(beg_time, end_time, mag_type=''):
    beg_time_str = beg_time.strftime('%Y-%m-%d/%H:%M:%S')
    end_time_str = end_time.strftime('%Y-%m-%d/%H:%M:%S')
    filelist = pyspedas.wind.mfi(trange=[beg_time_str, end_time_str], datatype='h3-rtn', downloadonly=True)
    mag_wind = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    epochmag = mag_wind['Epoch']
    timebinmag = (epochmag > beg_time) & (epochmag < end_time)
    epochmag = epochmag[timebinmag]
    Br = mag_wind['BRTN'][timebinmag, 0]

    Bt = mag_wind['BRTN'][timebinmag, 1]
    Bn = mag_wind['BRTN'][timebinmag, 2]
    Babs = np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2)
    Br[Br < -20.] = np.nan
    Bt[Bt < -20.] = np.nan
    Bn[Bn < -20.] = np.nan
    return epochmag, Br, Bt, Bn, Babs


def read_pmom_data_wind(beg_time, end_time):
    beg_time_str = beg_time.strftime('%Y-%m-%d/%H:%M:%S')
    end_time_str = end_time.strftime('%Y-%m-%d/%H:%M:%S')
    filelist = pyspedas.wind.threedp(trange=[beg_time_str, end_time_str], datatype='3dp_plsp', downloadonly=True)
    pmom_wind = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    epochpmom = pmom_wind['Epoch']
    timebinpmom = (epochpmom > beg_time) & (epochpmom < end_time)
    epochpmom = epochpmom[timebinpmom]
    densp = pmom_wind['MOM.P.DENSITY'][timebinpmom]
    vr = -pmom_wind['MOM.P.VELOCITY'][timebinpmom, 0]
    vt = -pmom_wind['MOM.P.VELOCITY'][timebinpmom, 1]
    vn = pmom_wind['MOM.P.VELOCITY'][timebinpmom, 2]
    T = pmom_wind['MOM.P.AVGTEMP'][timebinpmom]
    return epochpmom, densp, vr, vt, vn, T


def read_epad_data_wind(beg_time, end_time):
    beg_time_str = beg_time.strftime('%Y-%m-%d/%H:%M:%S')
    end_time_str = end_time.strftime('%Y-%m-%d/%H:%M:%S')
    filelist = pyspedas.wind.swe(trange=[beg_time_str, end_time_str], datatype='h3', downloadonly=True)
    pad_wind = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    epochpad = pad_wind['Epoch']
    timebinpad = (epochpad > beg_time) & (epochpad < end_time)
    epochpad = epochpad[timebinpad]
    pad_e6 = pad_wind['f_pitch_E06'][timebinpad, :]
    ene_e6 = '193.4 eV'
    pa = np.linspace(0, 180, 30)

    return epochpad, ene_e6, pa, pad_e6


if __name__ == '__main__':
    n_Cases = 1
    cases = list({} for i in range(n_Cases))
    i = 0
    cases[i]['beg_time'] = datetime(2022, 3, 2)
    cases[i]['end_time'] = datetime(2022, 3, 7)
    cases[i]['mid_time'] = datetime(2022, 3, 3, 14)
    cases[i]['observer'] = 'wind'

    cases[i]['epochmag'], cases[i]['Br'], cases[i]['Bt'], cases[i]['Bn'], cases[i]['Babs'] \
        = read_mag_data_wind(cases[i]['beg_time'], cases[i]['end_time'])
    cases[i]['epochpmom'], cases[i]['densp'], cases[i]['vp_r'], cases[i]['vp_t'], cases[i]['vp_n'], cases[i]['Tp'] \
        = read_pmom_data_wind(cases[i]['beg_time'], cases[i]['end_time'])
    cases[i]['r_sc_carr_mag'], cases[i]['lon_sc_carr_mag'], cases[i]['lat_sc_carr_mag'] \
        = get_rlonlat_earth_carr(cases[i]['epochmag'], for_psi=False)
    epochpad, ene, pa, pad = read_epad_data_wind(cases[i]['beg_time'], cases[i]['end_time'])
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

    # cases[i]['Br'][cases[i]['Br']<-20.]=np.nan
    axs[2].plot(cases[i]['epochmag'], cases[i]['Babs'], 'm-', linewidth=1, label='Btot')
    axs[2].plot(cases[i]['epochmag'], cases[i]['Br'], 'k-', linewidth=1, label='Br')
    axs[2].set_ylim([-12, 14])
    axs[2].set_ylabel('$B_{r,tot}$\n (nT)')
    axs[2].legend(loc='upper left')
    ax2 = axs[2].twinx()
    ax2.plot(cases[i]['epochpmom'], cases[i]['vp_r'], 'r-', linewidth=1)
    ax2.set_ylabel('$V_{pr}$\n$(km/s)$', color='r')

    # cases[i]['Bn'][cases[i]['Bn']<-20.]=np.nan
    # cases[i]['Bt'][cases[i]['Bt'] < -20.] = np.nan
    axs[3].plot(cases[i]['epochmag'], cases[i]['Bt'], 'r-', linewidth=1, label='Bt')
    axs[3].plot(cases[i]['epochmag'], cases[i]['Bn'], 'b-', linewidth=1, label='Bn')
    axs[3].set_ylim([-12, 12])
    axs[3].legend(loc=2, bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
    axs[3].set_ylabel('$B_{tn}$\n (nT)')

    axs[4].plot(cases[i]['epochpmom'], cases[i]['densp'], 'k-', linewidth=1, label='Np')
    axs[4].set_ylabel('$N_p$\n$(cm^{-3})$')
    axs[4].set_xlabel('Time')
    ax2 = axs[4].twinx()
    ax2.plot(cases[i]['epochpmom'], cases[i]['Tp'], 'r-', linewidth=1, label='Tp')
    ax2.set_ylabel('$T_p$\n$(eV)$', color='r')
    plt.suptitle('WIND Overview')

    plt.show()

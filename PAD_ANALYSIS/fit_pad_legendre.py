from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# MY MODULES
from load_psp_data import load_spe_data

######### READ DATA ###########

# INPUT #
beg_time = datetime(2021, 4, 28, 15)
end_time = datetime(2021, 4, 30, 5)

# Read data. Cut into subepoch
spe_pad = load_spe_data(beg_time.strftime('%Y%m%d'), end_time.strftime('%Y%m%d'))
epochpade = spe_pad['Epoch']
timebinpade = (epochpade > beg_time) & (epochpade < end_time)

epochpade = epochpade[timebinpade]
EfluxVsPAE = spe_pad['EFLUX_VS_PA_E'][timebinpade, :, :]
PitchAngle = spe_pad['PITCHANGLE'][timebinpade, :]
Energy_val = spe_pad['ENERGY_VALS'][timebinpade, :]
EfluxVsE = spe_pad['EFLUX_VS_ENERGY']

# Calculate normalized PAD
norm_EfluxVsPAE = np.zeros_like(EfluxVsPAE)
for i in range(12):
    norm_EfluxVsPAE[:, i, :] = EfluxVsPAE[:, i, :] / np.nansum(EfluxVsPAE, 1)

# INPUT # CHOOSE ENERGY CHANNEL
i_energy = 8
print('PAD Energy:', Energy_val[0, i_energy])
enestr = '%.2f' % Energy_val[0, i_energy]

# INPUT # CHOOSE CLIM. zmin/max1 for PAD; zmin/max2 for norm_PAD
zmin1 = 9.
zmax1 = 10.5
zmin2 = -1.5
zmax2 = -0.

# INPUT # Preview PAD and modify energy channel & colorbar
preview_PAD = False
if preview_PAD:
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('PAD')
    plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_energy])).T, cmap='jet')
    plt.colorbar()
    plt.clim([zmin1, zmax1])
    plt.subplot(2, 1, 2)
    plt.title('Norm PAD')
    plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(norm_EfluxVsPAE[:, :, i_energy])).T, cmap='jet')
    plt.colorbar()
    plt.clim([zmin2, zmax2])
    plt.suptitle(enestr)
    plt.show()

# %%
from scipy.optimize import curve_fit
from scipy.special import legendre
from tqdm import tqdm


def legendre_fit_func(x_rad_, a0_, a1_, a2_, a3_, a4_, a5_, norm_):
    return norm_ * 1e9 * (a0_ * legendre(0)(np.cos(x_rad_))
                          + a1_ * legendre(1)(np.cos(x_rad_))
                          + a2_ * legendre(2)(np.cos(x_rad_))
                          + a3_ * legendre(3)(np.cos(x_rad_))
                          + a4_ * legendre(4)(np.cos(x_rad_))
                          + a5_ * legendre(5)(np.cos(x_rad_)))


legendre_parameters = np.zeros((len(epochpade), 6))
for pad_ind in tqdm(range(len(epochpade))):
    # for pad_ind in [6000]:

    pad_dt = epochpade[pad_ind]
    pad_1d = np.array(EfluxVsPAE[pad_ind, :, i_energy])
    pa_rad = np.deg2rad(PitchAngle[pad_ind][:])

    nanbin = np.isnan(pad_1d)
    pad_1d = pad_1d[~nanbin]
    pa_rad = pa_rad[~nanbin]

    popt, pcov = curve_fit(legendre_fit_func, pa_rad, pad_1d)
    legendre_parameters[pad_ind] = popt[:-1] * popt[-1]

    # ### PLOT ###
    # pa_rad_fit = np.linspace(0, np.pi, 100)
    # pad_1d_fit = legendre_fit_func(pa_rad_fit, *popt)
    # plt.figure()
    # plt.subplot(3,1,1)
    # plt.pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_energy])).T, cmap='jet')
    # plt.clim([zmin1, zmax1])
    # plt.plot([epochpade[pad_ind],epochpade[pad_ind]],[0,180],'r--')
    # plt.subplot(3,1,2)
    # plt.scatter(np.cos(pa_rad),pad_1d,label='raw')
    # plt.plot(np.cos(pa_rad_fit),pad_1d_fit,label='fit')
    # plt.xlabel('cos(θ)')
    # plt.ylabel('PAD')
    # plt.legend()
    # plt.subplot(3,1,3)
    # plt.plot(popt[:-1]*popt[-1])
    # plt.xlabel('Terms')
    # plt.ylabel('Ai (*1e9)')
    # plt.suptitle(pad_dt.strftime())
    # plt.show()

# %%
from matplotlib.transforms import Bbox

fig, axs = plt.subplots(2, 1, sharex=True)
pos = axs[0].pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(EfluxVsPAE[:, :, i_energy])).T, cmap='jet',
                        vmax=zmax1, vmin=zmin1)
axpos = axs[0].get_position()
pad = 0.01
width = 0.01
caxpos = Bbox.from_extents(
    axpos.x1 + pad,
    axpos.y0,
    axpos.x1 + pad + width,
    axpos.y1
)
cax = axs[1].figure.add_axes(caxpos)
cbar = plt.colorbar(pos, cax=cax)
axs[0].set_ylabel('Pitch Angle [deg]')
axs[0].set_title('e-PAD E=' + enestr + 'eV')

pos = axs[1].pcolormesh(epochpade, np.arange(6), legendre_parameters.T, cmap='seismic', vmax=5, vmin=-5)
axpos = axs[1].get_position()
pad = 0.01
width = 0.01
caxpos = Bbox.from_extents(
    axpos.x1 + pad,
    axpos.y0,
    axpos.x1 + pad + width,
    axpos.y1
)
cax = axs[1].figure.add_axes(caxpos)
cbar = plt.colorbar(pos, cax=cax)
axs[1].set_ylabel('i')
axs[1].set_title('Polynomial Coefficients')
plt.show()

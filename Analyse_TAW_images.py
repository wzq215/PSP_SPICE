import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# plot_psp_sun_carrington('20210426T000000', '20210427T000000')

def plot_images_d(time_xx, tt, timemap, d):
    maxind = []
    centerline = []

    for i in range(time_xx.shape[1]):
        # print(np.nansum(timemap[i-1:i+1,430:470]**2*np.arange(430,470))/np.nansum(timemap[i-1:i+1,430:470]**2))
        maxind_tmp = int(
            np.nansum(timemap[i, 430:470] ** 2 * np.arange(430, 470)) / np.nansum(timemap[i, 430:470] ** 2))
        # print(i,maxind_tmp)
        maxind.append(maxind_tmp)
        centerline.append(np.nanmin(timemap[i, maxind_tmp - 7:maxind_tmp + 7]))
    displacement = tt[maxind, 0] * d

    plt.pcolormesh(time_xx, tt * d, timemap.T)
    plt.plot(time_xx[0, :], tt[maxind, 0] * d, 'white')
    plt.colorbar()
    plt.clim([9e-15, 9e-13])
    plt.ylabel('d * Carrington Latitude (Rs)')
    plt.xlabel('Observation Time')
    plt.xlim([time_xx[0, 0], time_xx[0, nn]])
    plt.ylim([-6 * d, 6 * d])
    plt.title('HEEQ Map (R=' + str(d) + 'Rs' + ')')
    return displacement, centerline

def func(x,a,b,c,d):
    return a*np.cos(b*x+c)+d


def fit_displacement(time_xx, tt, displacement, d, nn):
    popt, pcov = curve_fit(func, np.arange(nn), displacement[0:nn], p0=[-5, 0.25, -4, -15])
    plt.plot(displacement[0:nn])
    plt.plot(np.arange(nn), func(np.arange(nn), *popt))
    plt.xlabel('Observation Time/15min')
    plt.ylabel('Y [Rs]')
    plt.title('Displacement d=' + str(d) + 'Rs')
    plt.ylim([-30, 5])
    print(popt)
    return popt


def fit_centerline(time_xx, tt, centerline, d, nn):
    popt, pcov = curve_fit(func, np.arange(nn), centerline[0:nn], p0=[-10, 0.25, -4, -15])
    plt.plot(centerline[0:nn])
    plt.plot(np.arange(nn), func(np.arange(nn), *popt))
    # plt.ylim([-30,5])
    print(popt)
    return popt


def fft_displacement(time_xx, tt, displacement, d, nn):
    sp = np.fft.fft(displacement, n=nn)
    # print(sp[1:nn//2])
    freq = np.fft.fftfreq(nn)
    # print(freq[1:nn//2])
    timestep = np.mean(np.diff(time_xx[0, 0:nn]))
    # print(timestep)
    plt.semilogy(freq[1:nn // 2] / (15 / 60), abs(sp[1:nn // 2]) * 2)
    # print(abs(sp))
    plt.title('PSD d=' + str(d) + 'Rs')
    plt.xlabel('f [1/hr]')
    return freq, sp


plt.figure()
nn = 40
d = 10.5
data = np.load('Timemap_d=' + str(d) + '_0426.npz', allow_pickle=True)
time_xx = data['arr_0']
tt = data['arr_1']
timemap = data['arr_2']

plt.subplot(3, 3, 1)
# plt.figure()
displacement, centerline = plot_images_d(time_xx, tt, timemap, d)
# displacement = tt[maxind,0]*d
# print(maxind)
# print(centerline)
plt.clim([9e-15, 9e-13])
# plt.show()
plt.subplot(3, 3, 2)
popt = fit_displacement(time_xx, tt, displacement, d, nn)
# popt = fit_centerline(time_xx,tt,centerline,d,nn)

plt.subplot(3, 3, 3)
freq, sp = fft_displacement(time_xx, tt, displacement, d, nn)

d = 11.5
data = np.load('Timemap_d=' + str(d) + '_0426.npz', allow_pickle=True)
time_xx = data['arr_0']
tt = data['arr_1']
timemap = data['arr_2']

plt.subplot(3, 3, 4)
displacement, centerline = plot_images_d(time_xx, tt, timemap, d)
plt.clim([7e-15, 7e-13])

plt.subplot(3, 3, 5)
popt = fit_displacement(time_xx, tt, displacement, d, nn)
# popt = fit_centerline(time_xx,tt,centerline,d,nn)

plt.subplot(3, 3, 6)
fft_displacement(time_xx, tt, displacement, d, nn)

d = 12.5
data = np.load('Timemap_d=' + str(d) + '_0426.npz', allow_pickle=True)
time_xx = data['arr_0']
tt = data['arr_1']
timemap = data['arr_2']

plt.subplot(3, 3, 7)
displacement, centerline = plot_images_d(time_xx, tt, timemap, d)
# maxind[11]=447
# maxind[6]=451
# maxind[7]=450
plt.clim([5e-15, 5e-13])

plt.subplot(3, 3, 8)
popt = fit_displacement(time_xx, tt, displacement, d, nn)
# popt = fit_centerline(time_xx,tt,centerline,d,nn)

plt.subplot(3, 3, 9)
fft_displacement(time_xx, tt, displacement, d, nn)

plt.show()

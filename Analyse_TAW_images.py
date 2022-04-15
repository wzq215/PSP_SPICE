import numpy as np
import matplotlib.pyplot as plt
from plot_body_positions import plot_psp_sun_carrington
# plot_psp_sun_carrington('20210426T000000', '20210427T000000')



plt.figure()

plt.subplot(3,2,1)

d=10
data10 = np.load('Timemap_d='+str(d)+'_0426.npz',allow_pickle=True)
time_xx = data10['arr_0']
tt = data10['arr_1']
timemap = data10['arr_2']

maxind=[]
centerline = []
nn= 50
for i in range(time_xx.shape[1]):
    maxind_tmp = int(np.nansum(timemap[i,430:460]**2*np.arange(430,460))/np.nansum(timemap[i,430:460]**2))
    print(i,maxind_tmp)
    maxind.append(maxind_tmp)
    centerline.append(np.nanmin(timemap[i,maxind_tmp-5:maxind_tmp+5]))
# indexs = [8,9,10,11,12,16,19,49]
# subs = [439,435,436,437,438,440,441,430]
# # maxind[8,9,10,11,12,16,19,49] = [439,435,436,437,438,440,441,430]
# for i in range(len(indexs)):
#     maxind[indexs[i]] = subs[i]
plt.pcolormesh(time_xx, tt, timemap.T)
plt.plot(time_xx[0,:],tt[maxind,0],'white')
plt.colorbar()
plt.clim([9e-15, 9e-13])
plt.ylabel('Carrington Latitude (deg)')
plt.xlabel('Observation Time')
plt.xlim([time_xx[0,0],time_xx[0,nn]])
plt.ylim([-20,20])
plt.title('HEEQ Map (R=' + str(d) + 'Rs' + ')')

from scipy.optimize import curve_fit
def func(x,a,b,c,d):
    return a*np.cos(b*x+c)+d
popt,pcov = curve_fit(func,np.arange(nn),maxind[0:nn])

plt.subplot(3,2,2)
# plt.semilogy(time_xx[0,:],centerline)
# plt.plot(time_xx[0,:],maxind)
plt.plot(maxind[0:nn])
plt.plot(np.arange(nn),func(np.arange(nn),*popt))

# plt.plot(time_xx[0,:],tt[maxind,0])
# plt.xlim([time_xx[0,0],time_xx[0,50]])
# plt.ylim([2e-13,2e-12])

plt.show()

sp=np.fft.fft(maxind,n=nn)
print(sp[1:nn//2])
freq = np.fft.fftfreq(nn)
print(freq[1:nn//2])
timestep = np.mean(np.diff(time_xx[0,0:nn]))
print(timestep)
# freq =
plt.semilogy(freq[1:nn//2]/(15*60),abs(sp[1:nn//2])*2)
print(abs(sp))
plt.title('PSD d=10Rs')
plt.xlabel('f [Hz]')
plt.show()



plt.subplot(3,2,3)
d=11
data11 = np.load('Timemap_d='+str(d)+'_0426.npz',allow_pickle=True)
time_xx = data11['arr_0']
tt = data11['arr_1']
timemap = data11['arr_2']
maxind=[]

centerline = []
for i in range(time_xx.shape[1]):
    maxind_tmp = int(np.nansum(timemap[i,430:460]**2*np.arange(430,460))/np.nansum(timemap[i,430:460]**2))
    print(i,maxind_tmp)
    maxind.append(maxind_tmp)
    centerline.append(np.nanmin(timemap[i,maxind_tmp-5:maxind_tmp+5]))

plt.pcolormesh(time_xx, tt, timemap.T)
plt.plot(time_xx[0,:],tt[maxind,0],'white')
plt.colorbar()
plt.clim([8e-15, 8e-13])
plt.ylabel('Carrington Latitude (deg)')
plt.xlabel('Observation Time')
plt.xlim([time_xx[0,0],time_xx[0,50]])
plt.ylim([-20,20])
# plt.ylim([-20,20])
plt.title('HEEQ Map (R=' + str(d) + 'Rs' + ')')

# plt.subplot(3,2,4)
# # plt.semilogy(time_xx[0,:],centerline)
# plt.plot(time_xx[0,:],maxind)
# plt.xlim([time_xx[0,0],time_xx[0,50]])
# # plt.ylim([1e-13,1e-12])

popt,pcov = curve_fit(func,np.arange(nn),maxind[0:nn])

plt.subplot(3,2,4)
# plt.semilogy(time_xx[0,:],centerline)
# plt.plot(time_xx[0,:],maxind)
plt.plot(maxind[0:nn])
plt.plot(np.arange(nn),func(np.arange(nn),*popt))

# plt.plot(time_xx[0,:],tt[maxind,0])
# plt.xlim([time_xx[0,0],time_xx[0,50]])
# plt.ylim([2e-13,2e-12])

plt.show()

sp=np.fft.fft(maxind,n=nn)
print(sp[1:nn//2])
freq = np.fft.fftfreq(nn)
print(freq[1:nn//2])
timestep = np.mean(np.diff(time_xx[0,0:nn]))
print(timestep)
# freq =
plt.semilogy(freq[1:nn//2]/(15*60),abs(sp[1:nn//2])*2)
plt.title('PSD d=11Rs')
plt.xlabel('f [Hz]')
print(abs(sp))
plt.show()

plt.subplot(3,2,5)
d=12
data12 = np.load('Timemap_d='+str(d)+'_0426.npz',allow_pickle=True)
time_xx = data12['arr_0']
tt = data12['arr_1']
timemap = data12['arr_2']
maxind=[]
centerline = []
for i in range(time_xx.shape[1]):
    maxind_tmp = int(np.nansum(timemap[i,430:460]**2*np.arange(430,460))/np.nansum(timemap[i,430:460]**2))
    print(i,maxind_tmp)
    maxind.append(maxind_tmp)
    centerline.append(np.nanmin(timemap[i,maxind_tmp-5:maxind_tmp+5]))

plt.pcolormesh(time_xx, tt, timemap.T)
plt.plot(time_xx[0,:],tt[maxind,0],'white')
plt.colorbar()
plt.clim([6e-15, 6e-13])
plt.ylabel('Carrington Latitude (deg)')
plt.xlabel('Observation Time')
plt.xlim([time_xx[0,0],time_xx[0,50]])
plt.ylim([-20,20])
# plt.ylim([-20,20])
plt.title('HEEQ Map (R=' + str(d) + 'Rs' + ')')

# plt.subplot(3,2,6)
# # plt.semilogy(time_xx[0,:],centerline)
# plt.plot(time_xx[0,:],maxind)
# plt.xlim([time_xx[0,0],time_xx[0,50]])
# # plt.ylim([1e-13,1e-12])

popt,pcov = curve_fit(func,np.arange(nn),maxind[0:nn])

plt.subplot(3,2,6)
# plt.semilogy(time_xx[0,:],centerline)
# plt.plot(time_xx[0,:],maxind)
plt.plot(maxind[0:nn])
plt.plot(np.arange(nn),func(np.arange(nn),*popt))

# plt.plot(time_xx[0,:],tt[maxind,0])
# plt.xlim([time_xx[0,0],time_xx[0,50]])
# plt.ylim([2e-13,2e-12])

plt.show()

sp=np.fft.fft(maxind,n=nn)
print(sp[1:nn//2])
freq = np.fft.fftfreq(nn)
print(freq[1:nn//2])
timestep = np.mean(np.diff(time_xx[0,0:nn]))
print(timestep)
# freq =
plt.semilogy(freq[1:nn//2]/(15*60),abs(sp[1:nn//2])*2)
print(abs(sp))
plt.title('PSD d=12Rs')
plt.xlabel('f [Hz]')
plt.show()

plt.show()
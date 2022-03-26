import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
import furnsh_kernels
from datetime import datetime
from wispr_insitu import plot_frames
from scipy.optimize import leastsq
from PIL import Image

AU = 1.49e8  # km
'''
1 - 20210116T010019 - (34,228)
2 - 20210116T013019 - (49,227)
3 - 20210116T020019 - (61,224)
4 - 20210116T023019 - (79,223)
5 - 20210116T030019 - (96,221)
6 - 20210116T033019 - (107,219)
7 - 20210116T040019 - (127,216)
'''
'''
1 - 190019 - 67,224
2 - 193019 - 
3 - 200019 - 78,223
4 - 203019 - 88,220
5 - 210019 - 100,219
6 - 213019 - 111,218
7 - 220019 - 
8 - 223019 - 127,216
9 - 230028 - 141,214
'''

et1 = spice.datetime2et(datetime.strptime('20210116T010019','%Y%m%dT%H%M%S'))
et2 = spice.datetime2et(datetime.strptime('20210116T040019','%Y%m%dT%H%M%S'))
ets = np.linspace(et1,et2,7)
locs=np.array([(43,228),(65,225),(83,222),(107,220),(122,219),(148,215),(167,211)])

# et1 = spice.datetime2et(datetime.strptime('20210116T190019','%Y%m%dT%H%M%S'))
# et2 = spice.datetime2et(datetime.strptime('20210116T230019','%Y%m%dT%H%M%S'))
# ets = np.linspace(et1,et2,9)
# ets = ets[[2,3,4,5,7]]
# locs=np.array([(109,217),(135,215),(150,213),(163,211),(179,212)])#,(127,216),(141,214)])

print(locs)
boresight = np.array([480,512])//2
im = Image.open('psp/wispr/images/inner/psp_L3_wispr_20210116T190019_V1_1221.png')
plt.imshow(im)
plt.scatter(locs.T[0],locs.T[1],c=ets,cmap='jet',s=10)
plt.title('Trace Markers')
plt.show()
# plot_frames(et1)

'''Step 1: pixel position to LOS in PSP frame'''
los_locs=[]
for i in range(7):
    loc = np.array(locs[i])
    et = ets[i]
    los_in_wispr_inner = (loc-boresight)*40/480 #deg
    x_tmp = AU*np.sin(np.deg2rad(los_in_wispr_inner[0]))
    y_tmp = AU*np.sin(np.deg2rad(los_in_wispr_inner[1]))
    z_tmp = np.sqrt(AU**2-x_tmp**2-y_tmp**2)
    # trg_in_wispr_inner = [x_t]
    print(los_in_wispr_inner)
    los_in_psp_frame,_ = spice.spkcpt([x_tmp,y_tmp,z_tmp],'SPP','SPP_WISPR_INNER',et,'SPP_SPACECRAFT','OBSERVER','NONE','SPP')
    lat_in_psp_frame = np.rad2deg(spice.reclat(los_in_psp_frame[[2, 0, 1]])[1:3])
    los_locs.append(lat_in_psp_frame)

los_locs = np.array(los_locs)
print(los_locs)
plt.scatter(los_locs[:,0],los_locs[:,1],c=ets,cmap='jet')
plt.xlabel(r'$\gamma$ (Longitude/deg)')
plt.ylabel(r'$\beta$ (Latitude/deg)')
plt.xlim(0,30)
plt.ylim(-15,15)
plt.gca().invert_yaxis()
plt.title('Trace marked in the PSP SPACECRAFT Frame')
plt.show()

'''Step 2: fit trajectory'''
G = 6.6726e-11 #m^3sec^-2kg^-1
M = 1.988e30 #kg
mu = G*M*1e-9 # km^3sec^-2
t=np.linspace(0,6,7)*30*60 # sec
gamma = np.deg2rad(los_locs[:,0]) #rad
beta = np.deg2rad(los_locs[:,1])
print('gamma (rad): ',gamma)
print('beta (rad): ',beta)
psp_states,_ = spice.spkezr('SPP',ets,'SPP_HCI','NONE','SUN') #km
# print(psp_states[i])
psp_states = np.array(psp_states)
phi1=[]
r1=[]
incs = []
lnodes = []
for i in range(7):
    elt = spice.oscelt(psp_states[i],ets[i],mu)
    inc = elt[3]
    incs.append(inc)
    lnode=elt[4]
    lnodes.append(lnode)
    rlonlat = spice.reclat(psp_states[i][0:3])
    r1.append(rlonlat[0]) #km
    phi1.append(np.arccos(np.cos(rlonlat[2])*np.cos(rlonlat[1]-lnode))) #rad
    # phi1.append(np.arccos(np.cos(rlonlat[2])*np.cos(rlonlat[1])))
incs = np.array(incs)
print('inclination (deg): ',180-np.rad2deg(incs))

delta2 = -0.075#-0.055
print(delta2)
# print(np.rad2deg(lnodes))
print(np.rad2deg(phi1))
def func(p):
    r20=p[0]
    v=p[1]
    phi2=p[2]
    a = p[3]
    # a=0
    delta2=p[4]
    r2 = r20 + v*t+a*t**2/2
    phi2 = phi2-lnodes
    return (r1-r2*np.cos(delta2)*np.cos(phi2-phi1))/(r2*np.cos(delta2)*np.sin(phi2-phi1))-1/np.tan(gamma)
Rs = 696300 #km
p0=[5*Rs,300,np.deg2rad(80),0,0]
plsq = leastsq(func,p0)
print(plsq)
print(plsq[0][0]/Rs)
print(plsq[0][1])
print(np.rad2deg(plsq[0][2]))
print(plsq[0][3])
print(plsq[0][4])


'''Visualization'''
# UTC2ET
start_time = '2021-01-14'
stop_time = '2021-01-17'
start_dt = datetime.strptime(start_time, '%Y-%m-%d')
stop_dt = datetime.strptime(stop_time, '%Y-%m-%d')
utc = [start_dt.strftime('%b %d, %Y'), stop_dt.strftime('%b %d, %Y')]
etOne = spice.str2et(utc[0])
etTwo = spice.str2et(utc[1])
etOne = spice.datetime2et(datetime.strptime('20210115T000000','%Y%m%dT%H%M%S'))
etTwo = spice.datetime2et(datetime.strptime('20210117T140000','%Y%m%dT%H%M%S'))

# Epochs
step = 100
times = [x * (etTwo - etOne) / step + etOne for x in range(step)]

## Plot PSP&SUN orbit for specified time range
psp_positions, psp_LightTimes = spice.spkpos('SPP', times, 'SPP_HCI', 'NONE', 'SUN')
sun_positions, sun_LightTimes = spice.spkpos('SUN', times, 'SPP_HCI', 'NONE', 'SUN')
psp_positions = psp_positions.T  # psp_positions is shaped (4000, 3), let's transpose to (3, 4000) for easier indexing
psp_positions = psp_positions / AU
sun_positions = sun_positions.T
sun_positions = sun_positions / AU

r2 = (plsq[0][0]+plsq[0][1]*t+plsq[0][3]*t**2/2)
phi2 = plsq[0][2]
structure_pos = []
fit_lats=[]
for i in range(7):
    pos_tmp = [r2[i]*np.cos(delta2)*np.cos(phi2),r2[i]*np.cos(delta2)*np.sin(phi2),r2[i]*np.sin(delta2)]
    structure_pos.append(pos_tmp)
    fit_pos_inner, _ = spice.spkcpt(pos_tmp, 'SUN', 'SPP_HCI', ets[i], 'SPP_SPACECRAFT',
                                          'OBSERVER', 'NONE', 'SPP')
    fit_lat = spice.reclat(fit_pos_inner[[2, 0, 1]])
    fit_lats.append(fit_lat[1:3])

print(structure_pos)
structure_pos = np.array(structure_pos).T/AU
fit_lats = np.array(fit_lats)

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot(psp_positions[0],psp_positions[1],psp_positions[2],c='gray')
ax.scatter(psp_states.T[0]/AU, psp_states.T[1]/AU, psp_states.T[2]/AU, c=ets, cmap='jet')
ax.scatter(structure_pos[0],structure_pos[1],structure_pos[2],c=ets,cmap='jet')
ax.scatter(0,0,0, c='r')
ax.plot([0,0.01],[0,0],[0,0],c='k')
ax.plot([0,0],[0,0.03],[0,0],c='k')
ax.plot([0,0],[0,0],[0,0.09],c='k')
# ax.plot([0,0.1*np.cos(np.mean(lnodes))],[0,0.1*np.sin(np.mean(lnodes))],[0,0],c='b')
# ax.scatter(1, 0, 0, c='red')
ax.set_xlabel('X (AU)')
ax.set_ylabel('Y (AU)')
ax.set_zlabel('Z (AU)')
ax.set_xlim([0,0.1])
ax.set_ylim([0,0.1])
ax.set_zlim([-0.05,0.05])
# plt.title('PSP' + '(' + start_time + '_' + stop_time + ')')
plt.title('Fitting Result (in PSP_HCI frame)')
plt.show()
plt.subplot(121)
plt.plot(np.linspace(1,7,7),np.rad2deg(gamma),'k.-',label='data')
plt.plot(np.linspace(1,7,7),np.rad2deg(fit_lats.T[0]),'b-',label='fit')
plt.legend()
plt.title(r'$\gamma$ (deg)')
plt.subplot(122)
plt.plot(np.linspace(1,7,7),np.rad2deg(beta),'k.-',label='data')
plt.plot(np.linspace(1,7,7),np.rad2deg(fit_lats.T[1]),'b-',label='fit')
plt.legend()
plt.title(r'$\beta$ (deg)')
plt.show()






etOne = spice.datetime2et(datetime.strptime('20210116T000000','%Y%m%dT%H%M%S'))
etTwo = spice.datetime2et(datetime.strptime('20210118T000000','%Y%m%dT%H%M%S'))

# Epochs
step = 100
times = np.array([x * (etTwo - etOne) / step + etOne for x in range(step)])
r2 = (plsq[0][0]+plsq[0][1]*(times-et1)+plsq[0][3]*(times-et1)**2/2)
phi2 = plsq[0][2]
structure_pos = []
# fit_lats=[]
for i in range(step):
    pos_tmp = [r2[i]*np.cos(delta2)*np.cos(phi2),r2[i]*np.cos(delta2)*np.sin(phi2),r2[i]*np.sin(delta2)]
    structure_pos.append(pos_tmp)
structure_pos = np.array(structure_pos).T/AU

## Plot PSP&SUN orbit for specified time range
psp_positions, psp_LightTimes = spice.spkpos('SPP', times, 'SPP_HCI', 'NONE', 'SUN')
sun_positions, sun_LightTimes = spice.spkpos('SUN', times, 'SPP_HCI', 'NONE', 'SUN')
psp_positions = psp_positions.T  # psp_positions is shaped (4000, 3), let's transpose to (3, 4000) for easier indexing
psp_positions = psp_positions / AU
sun_positions = sun_positions.T
sun_positions = sun_positions / AU
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(psp_positions[0],psp_positions[1],psp_positions[2],c=times,cmap='jet')
# ax.scatter(psp_states.T[0]/AU, psp_states.T[1]/AU, psp_states.T[2]/AU, c=ets, cmap='jet')
ax.scatter(structure_pos[0],structure_pos[1],structure_pos[2],c=times,cmap='jet')
ax.scatter(0,0,0, c='r')
ax.plot([0,0.01],[0,0],[0,0],c='k')
ax.plot([0,0],[0,0.03],[0,0],c='k')
ax.plot([0,0],[0,0],[0,0.09],c='k')
# ax.plot([0,0.1*np.cos(np.mean(lnodes))],[0,0.1*np.sin(np.mean(lnodes))],[0,0],c='b')
# ax.scatter(1, 0, 0, c='red')
ax.set_xlabel('X (AU)')
ax.set_ylabel('Y (AU)')
ax.set_zlabel('Z (AU)')
ax.set_xlim([0,0.1])
ax.set_ylim([0,0.1])
ax.set_zlim([-0.05,0.05])
# plt.title('PSP' + '(' + start_time + '_' + stop_time + ')')
plt.title('Fitting Result (in PSP_HCI frame)')
plt.show()

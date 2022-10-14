from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice

Rs = 393600
AU = 1.49e8  # km

# start_time = datetime(2021,4,27)
# stop_time = datetime(2021,5,15)
#
# start_et = spice.datetime2et(start_time)
# stop_et = spice.datetime2et(stop_time)
# start_time_str = start_time.strftime('%Y%m%dT%H%M%S')
# stop_time_str = stop_time.strftime('%Y%m%dT%H%M%S')
#
# step = 7*2
# times = [x * (stop_et - start_et) / step + start_et for x in range(step)]


start_time = datetime(2021, 4, 27, 0, 0, 0)
stop_time = datetime(2021, 5, 1, 0, 0, 0)
start_et = spice.datetime2et(start_time)
stop_et = spice.datetime2et(stop_time)
start_time_str = start_time.strftime('%Y%m%dT%H%M%S')
stop_time_str = stop_time.strftime('%Y%m%dT%H%M%S')
from datetime import timedelta

timestep = timedelta(hours=3)
steps = (stop_time - start_time) // timestep + 1
dttimes = np.array([x * timestep + start_time for x in range(steps)])
# print(dttimes[0:-1:24])
times = spice.datetime2et(dttimes)
coord = 'SPP_HG'

psp_pos_carr, _ = spice.spkpos('SPP', times, coord, 'NONE', 'SUN')
psp_pos_carr = psp_pos_carr.T / Rs
sun_pos_carr, _ = spice.spkpos('SUN', times, coord, 'NONE', 'SUN')

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot(psp_pos_carr[0], psp_pos_carr[1], psp_pos_carr[2], c='black')
ax.scatter(0, 0, 0, c='red')
ax.scatter(psp_pos_carr[0], psp_pos_carr[1], psp_pos_carr[2], c=times, cmap='rainbow', s=50)

# ax.plot([0,psp_pos_carr[0][0]],[0,psp_pos_carr[1][0]],[0,psp_pos_carr[2][0]], c='yellow')


wispr_inner_parameter = spice.getfov(-96100, 4)
wispr_outer_parameter = spice.getfov(-96120, 4)
obs_time = times[0]
obs_pos = psp_pos_carr[:, 0]
for i_edge in range(4):
    edge_inner, _ = spice.spkcpt(30 * Rs * wispr_inner_parameter[4][i_edge], 'SPP', 'SPP_WISPR_INNER', obs_time,
                                 coord, 'CENTER', 'NONE', 'SUN')
    edge_outer, _ = spice.spkcpt(30 * Rs * wispr_outer_parameter[4][i_edge], 'SPP', 'SPP_WISPR_OUTER', obs_time,
                                 coord, 'CENTER', 'NONE', 'SUN')
    edge_inner = edge_inner / Rs
    edge_outer = edge_outer / Rs

    edge_inner2, _ = spice.spkcpt(30 * Rs * wispr_inner_parameter[4][i_edge - 1], 'SPP', 'SPP_WISPR_INNER', obs_time,
                                  coord, 'CENTER', 'NONE', 'SUN')
    edge_outer2, _ = spice.spkcpt(30 * Rs * wispr_outer_parameter[4][i_edge - 1], 'SPP', 'SPP_WISPR_OUTER', obs_time,
                                  coord, 'CENTER', 'NONE', 'SUN')
    edge_inner2 = edge_inner2 / Rs
    edge_outer2 = edge_outer2 / Rs

    inedge, = ax.plot([obs_pos[0], edge_inner[0]], [obs_pos[1], edge_inner[1]], [obs_pos[2], edge_inner[2]],
                      c='green')
    outedge, = ax.plot([obs_pos[0], edge_outer[0]], [obs_pos[1], edge_outer[1]], [obs_pos[2], edge_outer[2]],
                       c='blue')

    ax.plot([edge_inner[0], edge_inner2[0]], [edge_inner[1], edge_inner2[1]], [edge_inner[2], edge_inner2[2]],
            c='green')
    ax.plot([edge_outer[0], edge_outer2[0]], [edge_outer[1], edge_outer2[1]], [edge_outer[2], edge_outer2[2]],
            c='blue')

inedge.set_label('WISPR_INNER_FOV [|lon|,|lat|]=' + str(
    abs(np.rad2deg(spice.reclat(wispr_inner_parameter[4][0][[2, 0, 1]])[1:3]))))
outedge.set_label('WISPR_OUTER_FOV [|lon|,|lat|]=' + str(
    abs(np.rad2deg(spice.reclat(wispr_outer_parameter[4][0][[2, 0, 1]])[1:3]))))

ax.set_xlim([-10, 50])
ax.set_ylim([-20, 40])
ax.set_zlim([-30, 30])
# ax.set_box_aspect((1,1,1))

plt.title('PSP orbit (20210427-20210503), HCI')
ax.set_xlabel('x [Rs]')
ax.set_ylabel('y [Rs]')
ax.set_zlabel('z [Rs]')

plt.show()

from wispr_insitu_TS import find_PSP_in_WISPR

wispr_time = start_et
insitu_time_range = times[1:]
inner_lst, outer_lst = find_PSP_in_WISPR(insitu_time_range, wispr_time, coord=coord)
print(inner_lst)

import sunpy.io.fits

data, header = sunpy.io.fits.read('data/orbit08/20210427/psp_L3_wispr_20210427T002720_V1_1211.fits')[0]
plt.figure()
plt.imshow(data)
plt.colorbar
plt.set_cmap('gray')
plt.clim([1e-14, 1e-12])
plt.xlabel('x [pixel]')
plt.ylabel('y [pixel]')

plt.title('20210427T001220')
size_inner = 40
boresight_inner = (960 // 2, 1024 // 2)
if inner_lst.size > 0:
    imlon = inner_lst[:, 2] * 960 / size_inner + boresight_inner[0]
    imlat = -inner_lst[:, 3] * 960 / size_inner + boresight_inner[1]
    im = plt.scatter(imlon, imlat, c=inner_lst[:, 0], cmap='rainbow', vmin=times[0], vmax=times[-1], s=50,
                     marker='.', alpha=1)
    cticks_dt = [datetime(2021, 4, 27), datetime(2021, 4, 28), datetime(2021, 4, 29), datetime(2021, 4, 30),
                 datetime(2021, 5, 1),
                 # datetime(2021,5,2),datetime(2021,5,3)
                 ]
    cbar = plt.colorbar(ticks=spice.datetime2et(cticks_dt))
    print(cbar.get_ticks)
    cticks = cbar.get_ticks()
    # cticks = np.floor(cticks)
    print(cticks)
    # cticklabels = [spice.et2datetime(et).strftime('%Y%m%d') for et in cticks]
    # cbar.ax.set_yticks(cticks)
    # cbar.ax.set_yticklabels(spice.et2datetime(cticks))
    # cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar
    # plt.colorbar()
plt.gca().invert_yaxis()
plt.show()

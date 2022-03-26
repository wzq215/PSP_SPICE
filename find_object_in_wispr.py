import imageio
import numpy as np
import matplotlib.patches
import furnsh_kernels
from plot_body_positions import get_aster_pos, get_body_pos
import spiceypy as spice
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

AU = 1.49e8  # km
start_time = '2021-10-01'
stop_time = '2021-12-31'
# obs_time = '20210115T1230'
start_dt = datetime.strptime(start_time, '%Y-%m-%d')
stop_dt = datetime.strptime(stop_time, '%Y-%m-%d')
# obs_dt = datetime.strptime(obs_time,'%Y%m%dT%H%M')
utc = [start_dt.strftime('%b %d, %Y'), stop_dt.strftime('%b %d, %Y')]
etOne = spice.str2et(utc[0])
etTwo = spice.str2et(utc[1])
step = 500
times = [x * (etTwo - etOne) / step + etOne for x in range(step)]

# obs1 = '20211201T120000'
# obs2 = '20211221T000000'
# obs_step = 39*4
obs1 = '20211201T120000'
obs2 = '20211221T000000'
obs_step = 500#45*4

obs_one = spice.datetime2et(datetime.strptime(obs1, '%Y%m%dT%H%M%S'))
obs_two = spice.datetime2et(datetime.strptime(obs2, '%Y%m%dT%H%M%S'))
obs_times = [x * (obs_two - obs_one) / obs_step + obs_one for x in range(obs_step)]

id = 1003751
aster_positions = get_aster_pos(id, start_time, stop_time, observer='SUN', frame='SPP_HCI')
'''found! 3712675, 3791243'''

psp_positions, _ = spice.spkpos('SPP', times, 'SPP_HCI', 'NONE', 'SUN')
sun_positions, _ = spice.spkpos('SUN', times, 'SPP_HCI', 'NONE', 'SUN')
ast_positions, _ = spice.spkpos(str(id), times, 'SPP_HCI', 'NONE', 'SUN')
ear_positions, _ = spice.spkpos('EARTH', times, 'SPP_HCI', 'NONE', 'SUN')
psp_positions = psp_positions.T / AU
sun_positions = sun_positions.T / AU
ast_positions = ast_positions.T / AU
ear_positions = ear_positions.T / AU
frames = []
frames2 = []
lat_lst = []
for obs_et in obs_times:
    obs_time_str = spice.et2datetime(obs_et).strftime('%Y-%m-%d %H:%M:%S')
    # print(obs_time_str)

    isvisibe_outer = spice.fovtrg('SPP_WISPR_OUTER', str(id), 'POINT', '', 'None', 'SPP', obs_et)
    isvisibe_inner = spice.fovtrg('SPP_WISPR_INNER', str(id), 'POINT', '', 'None', 'SPP', obs_et)
    if isvisibe_outer:
        print(str(id) + ' VISIBLE in WISPR_OUTER at ' + obs_time_str)
    if isvisibe_inner:
        print(str(id) + ' VISIBLE in WISPR_INNER at ' + obs_time_str)

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(psp_positions[0], psp_positions[1], psp_positions[2], c='gray',alpha=0.8)
    ax.plot(ast_positions[0], ast_positions[1], ast_positions[2], c='gray',alpha=0.8)
    ax.scatter(sun_positions[0], sun_positions[1], sun_positions[2], c='r')
    ax.plot(ear_positions[0], ear_positions[1], ear_positions[2], c='gray',alpha=0.8)

    psp_pos, _ = spice.spkpos('SPP', obs_et, 'SPP_HCI', 'NONE', 'SUN')
    comet_pos, _ = spice.spkpos(str(id), obs_et, 'SPP_HCI', 'NONE', 'SUN')
    earth_pos, _ = spice.spkpos('EARTH', obs_et, 'SPP_HCI', 'NONE', 'SUN')
    psp_pos = psp_pos.T/AU
    comet_pos = comet_pos.T/AU
    earth_pos = earth_pos.T/AU

    ax.scatter(psp_pos[0], psp_pos[1], psp_pos[2], c='k')
    ax.scatter(comet_pos[0], comet_pos[1], comet_pos[2], c='green')

    ax.plot([psp_pos[0],comet_pos[0]],[psp_pos[1],comet_pos[1]],[psp_pos[2],comet_pos[2]],c='green',linestyle='--',alpha=0.8,label='PSP to Leonard')
    print('distance')
    print(np.linalg.norm(psp_pos-comet_pos,2))
    ax.scatter(earth_pos[0], earth_pos[1], earth_pos[2], c='blue')
    obs_pos = psp_pos

    wispr_inner_parameter = spice.getfov(-96100, 4)
    wispr_outer_parameter = spice.getfov(-96120, 4)
    # print(wispr_outer_parameter)
    # Plot INNER/OUTER FOV
    for i_edge in range(4):
        edge_inner, _ = spice.spkcpt(1 * AU * wispr_inner_parameter[4][i_edge], 'SPP', 'SPP_WISPR_INNER', obs_et,
                                     'SPP_HCI', 'CENTER', 'NONE', 'SUN')
        edge_inner2, _ = spice.spkcpt(1 * AU * wispr_inner_parameter[4][i_edge-1], 'SPP', 'SPP_WISPR_INNER', obs_et,
                                     'SPP_HCI', 'CENTER', 'NONE', 'SUN')
        edge_outer, _ = spice.spkcpt(1 * AU * wispr_outer_parameter[4][i_edge], 'SPP', 'SPP_WISPR_OUTER', obs_et,
                                     'SPP_HCI', 'CENTER', 'NONE', 'SUN')
        edge_outer2, _ = spice.spkcpt(1 * AU * wispr_outer_parameter[4][i_edge-1], 'SPP', 'SPP_WISPR_OUTER', obs_et,
                                     'SPP_HCI', 'CENTER', 'NONE', 'SUN')
        edge_inner = edge_inner / AU
        edge_outer = edge_outer / AU
        edge_inner2 = edge_inner2 / AU
        edge_outer2 = edge_outer2 / AU
        if isvisibe_inner:
            inedge, = ax.plot([obs_pos[0], edge_inner[0]], [obs_pos[1], edge_inner[1]], [obs_pos[2], edge_inner[2]],
                          c='orange',linewidth=1.0)
            ax.plot([edge_inner[0], edge_inner2[0]], [edge_inner[1], edge_inner2[1]], [edge_inner[2], edge_inner2[2]],
                        c='orange',linewidth=1.0)

        if isvisibe_outer:
            outedge, = ax.plot([obs_pos[0], edge_outer[0]], [obs_pos[1], edge_outer[1]], [obs_pos[2], edge_outer[2]],
                           c='gold',linewidth=1.0)
            ax.plot([edge_outer[0], edge_outer2[0]], [edge_outer[1], edge_outer2[1]], [edge_outer[2], edge_outer2[2]],
                    c='gold',linewidth=1.0)

    if isvisibe_inner:
        inedge.set_label('WISPR_INNER_FOV')
    if isvisibe_outer:
        outedge.set_label('WISPR_OUTER_FOV')

    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.title('PSP & Leonard '+obs_time_str)
    plt.legend(loc='lower right')
    fig.savefig('figures/comet/3d('+obs_time_str+').png')
    plt.close(fig)
    # plt.show()
    frames.append(imageio.imread('figures/comet/3d('+obs_time_str+').png'))



    delta_bore = 43.9884470948
    if isvisibe_inner:
        innerpos, _ = spice.spkcpt(comet_pos * AU, 'SUN', 'SPP_HCI', obs_et, 'SPP_WISPR_INNER',
                                       'OBSERVER', 'NONE', 'SPP')
        innerlat = spice.reclat(innerpos[[2, 0, 1]])
        print('inner')
        print([np.rad2deg(innerlat[1]),np.rad2deg(innerlat[2])])
        lat_lst.append([np.rad2deg(innerlat[1]),np.rad2deg(innerlat[2])])
    if isvisibe_outer:
        outerpos, _ = spice.spkcpt(comet_pos * AU, 'SUN', 'SPP_HCI', obs_et, 'SPP_WISPR_OUTER',
                                   'OBSERVER', 'NONE', 'SPP')
        outerlat = spice.reclat(outerpos[[2, 0, 1]])
        print('outer')
        print([np.rad2deg(outerlat[1]),np.rad2deg(outerlat[2])])
        lat_lst.append([np.rad2deg(outerlat[1])+delta_bore,np.rad2deg(outerlat[2])])
    # outeredgepos, _ = spice.spkcpt(edge_outer[0:3] * AU, 'SUN', 'SPP_HCI', obs_et, 'SPP_WISPR_OUTER',
    #                            'OBSERVER', 'NONE', 'SPP')
    # outeredgelat = spice.reclat(outeredgepos[[2, 0, 1]])
    # inneredgepos, _ = spice.spkcpt(edge_inner[0:3] * AU, 'SUN', 'SPP_HCI', obs_et, 'SPP_WISPR_INNER',
    #                                'OBSERVER', 'NONE', 'SPP')
    # inneredgelat = spice.reclat(inneredgepos[[2, 0, 1]])

    plt.figure()
    lat_tmp = np.array(lat_lst)
    plt.scatter(0,0,c='orange',marker='x',label='Boresight of Inner')
    plt.scatter(delta_bore,0,c='gold',marker='x',label='Boresight of Outer')

    # plt.scatter(np.rad2deg(inneredgelat[1]),np.rad2deg(inneredgelat[2]),c='k')
    # plt.scatter(-np.rad2deg(inneredgelat[1]),np.rad2deg(inneredgelat[2]),c='k')
    # plt.scatter(-np.rad2deg(inneredgelat[1]),-np.rad2deg(inneredgelat[2]),c='k')
    # plt.scatter(np.rad2deg(inneredgelat[1]),-np.rad2deg(inneredgelat[2]),c='k')
    #
    # plt.scatter(np.rad2deg(outeredgelat[1])+delta_bore,np.rad2deg(outeredgelat[2]),c='k')
    # plt.scatter(-np.rad2deg(outeredgelat[1])+delta_bore,np.rad2deg(outeredgelat[2]),c='k')
    # plt.scatter(-np.rad2deg(outeredgelat[1])+delta_bore,-np.rad2deg(outeredgelat[2]),c='k')
    # plt.scatter(np.rad2deg(outeredgelat[1])+delta_bore,-np.rad2deg(outeredgelat[2]),c='k')
    #
    # print(np.rad2deg(outeredgelat[1]),np.rad2deg(outeredgelat[2]))
    # print(np.rad2deg(inneredgelat[1]),np.rad2deg(inneredgelat[2]))
    plt.gca().add_patch(plt.Rectangle((-19.2,-19.4),38.4,38.8,ec='orange',fill=False,linewidth=2.0))
    plt.gca().add_patch(plt.Rectangle((16.29,-26.7),55.4,53.4,ec='gold',fill=False,linewidth=2.0))
    plt.text(-19,22.5,'INNER FOV (40°x40°)')
    plt.text(17,30,'OUTER FOV (58°x58°)')
    plt.scatter(lat_tmp[:,0],lat_tmp[:,1],c='green',s=20,marker='.')
    plt.xlim([-22,75])
    plt.ylim([-31,31])
    plt.xticks([-20,-10,0,20,30,40,50,60,70],['20°E','10°E','0°','20°W','30°W','40°W','50°W','60°W','70°W'])
    # plt.xticklabels(['20°E','10°E','0°','20°W','30°W','40°W','50°W','60°W','70°W'])
    plt.yticks([-30,-20,-10,0,10,20,30],['30°N','20°N','10°N','0°','10°S','20°S','30°S'])
    # plt.yticklabels([])
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.gca().set_aspect(1)
    # plt.show()
    plt.title('Leonard in FOV'+obs_time_str)
    plt.savefig('figures/comet/2d('+obs_time_str+').png')
    plt.close()
    # plt.show()
    frames2.append(imageio.imread('figures/comet/2d('+obs_time_str+').png'))

imageio.mimsave('movies/comet/3d('+obs1+'-'+obs2+'.mp4', frames, fps=12)
imageio.mimsave('movies/comet/2d('+obs1+'-'+obs2+'.mp4', frames2, fps=12)


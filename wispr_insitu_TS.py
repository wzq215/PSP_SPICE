import re

import os

import itertools

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import spiceypy as spice
from datetime import datetime
import furnsh_kernels
import pandas as pd
import sunpy.map
from matplotlib import pyplot as plt
from datetime import timedelta

from plot_body_positions import xyz2rtp_in_Carrington
from plot_body_positions import plot_psp_sun_carrington

AU = 1.49e8  # km
Rs = 6.96e5  # solar radii km

ps = np.arange(0, 360, 0.2)
ts = np.arange(-90, 90, 0.2)

def datetimestr2et(datetime, type='%Y-%m-%d'):
    dt = datetime.strptime(datetime, type)
    # utc = dt.strftime('%b %d, %Y')
    et = spice.datetime2et(dt)
    return et


def plot_frames(obs_time):
    '''Plot PSP frame & WISPR_INNER/OUTER frame/FOV at obs_time'''
    obs_time_str = spice.et2datetime(obs_time).strftime('%Y-%m-%d %H:%M:%S')
    wispr_pos, _ = spice.spkpos('SPP', obs_time, 'SPP_HCI', 'NONE', 'SUN')  # km
    sun_pos, _ = spice.spkpos('SUN', obs_time, 'SPP_HCI', 'NONE', 'SUN')  # km
    obs_pos = wispr_pos.T / AU

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(obs_pos[0], obs_pos[1], obs_pos[2], c='black')
    wispr_inner_parameter = spice.getfov(-96100, 4)
    wispr_outer_parameter = spice.getfov(-96120, 4)

    print('inner_edge in deg')
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][0][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][1][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][2][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][3][[2, 0, 1]])[1:3]))
    print('outer_edge in deg')
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][0][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][1][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][2][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][3][[2, 0, 1]])[1:3]))

    # Plot PSP FRAME
    x_spp, _ = spice.spkcpt([0.3 * AU, 0, 0], 'SPP', 'SPP_SPACECRAFT', obs_time, 'SPP_HCI', 'CENTER', 'NONE', 'SUN')
    x_spp = x_spp / AU
    y_spp, _ = spice.spkcpt([0, 0.6 * AU, 0], 'SPP', 'SPP_SPACECRAFT', obs_time, 'SPP_HCI', 'CENTER', 'NONE', 'SUN')
    y_spp = y_spp / AU
    z_spp, _ = spice.spkcpt([0, 0, 0.9 * AU], 'SPP', 'SPP_SPACECRAFT', obs_time, 'SPP_HCI', 'CENTER', 'NONE', 'SUN')
    z_spp = z_spp / AU

    ax.plot([obs_pos[0], x_spp[0]], [obs_pos[1], x_spp[1]], [obs_pos[2], x_spp[2]], c='silver', linewidth=5.0,
            alpha=0.8)
    ax.plot([obs_pos[0], y_spp[0]], [obs_pos[1], y_spp[1]], [obs_pos[2], y_spp[2]], c='silver', linewidth=5.0,
            alpha=0.8)
    ax.plot([obs_pos[0], z_spp[0]], [obs_pos[1], z_spp[1]], [obs_pos[2], z_spp[2]], c='silver', linewidth=5.0,
            alpha=0.8, label='PSP_FRAME (longest for z, shortest for x)')

    # Plot INNER FRAME
    x_inner, _ = spice.spkcpt([0.3 * AU, 0, 0], 'SPP', 'SPP_WISPR_INNER', obs_time, 'SPP_HCI', 'CENTER', 'NONE', 'SUN')
    x_inner = x_inner / AU
    y_inner, _ = spice.spkcpt([0, 0.6 * AU, 0], 'SPP', 'SPP_WISPR_INNER', obs_time, 'SPP_HCI', 'CENTER', 'NONE', 'SUN')
    y_inner = y_inner / AU
    z_inner, _ = spice.spkcpt([0, 0, 0.9 * AU], 'SPP', 'SPP_WISPR_INNER', obs_time, 'SPP_HCI', 'CENTER', 'NONE', 'SUN')
    z_inner = z_inner / AU

    ax.plot([obs_pos[0], x_inner[0]], [obs_pos[1], x_inner[1]], [obs_pos[2], x_inner[2]], c='lime', linewidth=5.0,
            alpha=0.8)
    ax.plot([obs_pos[0], y_inner[0]], [obs_pos[1], y_inner[1]], [obs_pos[2], y_inner[2]], c='lime', linewidth=5.0,
            alpha=0.8)
    ax.plot([obs_pos[0], z_inner[0]], [obs_pos[1], z_inner[1]], [obs_pos[2], z_inner[2]], c='lime', linewidth=5.0,
            alpha=0.8, label='WISPR_INNER_FRAME (longest for z, shortest for x)')

    # Plot OUTER FRAME
    x_outer, _ = spice.spkcpt([0.3 * AU, 0, 0], 'SPP', 'SPP_WISPR_OUTER', obs_time, 'SPP_HCI', 'CENTER', 'NONE', 'SUN');
    x_outer = x_outer / AU
    y_outer, _ = spice.spkcpt([0, 0.6 * AU, 0], 'SPP', 'SPP_WISPR_OUTER', obs_time, 'SPP_HCI', 'CENTER', 'NONE', 'SUN');
    y_outer = y_outer / AU
    z_outer, _ = spice.spkcpt([0, 0, 0.9 * AU], 'SPP', 'SPP_WISPR_OUTER', obs_time, 'SPP_HCI', 'CENTER', 'NONE', 'SUN');
    z_outer = z_outer / AU

    ax.plot([obs_pos[0], x_outer[0]], [obs_pos[1], x_outer[1]], [obs_pos[2], x_outer[2]], c='aqua', linewidth=5.0,
            alpha=0.8)
    ax.plot([obs_pos[0], y_outer[0]], [obs_pos[1], y_outer[1]], [obs_pos[2], y_outer[2]], c='aqua', linewidth=5.0,
            alpha=0.8)
    ax.plot([obs_pos[0], z_outer[0]], [obs_pos[1], z_outer[1]], [obs_pos[2], z_outer[2]], c='aqua', linewidth=5.0,
            alpha=0.8, label='WISPR_OUTER_FRAME (longest for z, shortest for x)')

    # Plot INNER/OUTER FOV
    for i_edge in range(4):
        edge_inner, _ = spice.spkcpt(1 * AU * wispr_inner_parameter[4][i_edge], 'SPP', 'SPP_WISPR_INNER', obs_time,
                                     'SPP_HCI', 'CENTER', 'NONE', 'SUN')
        edge_outer, _ = spice.spkcpt(1 * AU * wispr_outer_parameter[4][i_edge], 'SPP', 'SPP_WISPR_OUTER', obs_time,
                                     'SPP_HCI', 'CENTER', 'NONE', 'SUN')
        edge_inner = edge_inner / AU
        edge_outer = edge_outer / AU
        inedge, = ax.plot([obs_pos[0], edge_inner[0]], [obs_pos[1], edge_inner[1]], [obs_pos[2], edge_inner[2]],
                          c='green')
        outedge, = ax.plot([obs_pos[0], edge_outer[0]], [obs_pos[1], edge_outer[1]], [obs_pos[2], edge_outer[2]],
                           c='blue')
    inedge.set_label('WISPR_INNER_FOV [|lon|,|lat|]=' + str(
        abs(np.rad2deg(spice.reclat(wispr_inner_parameter[4][0][[2, 0, 1]])[1:3]))))
    outedge.set_label('WISPR_OUTER_FOV [|lon|,|lat|]=' + str(
        abs(np.rad2deg(spice.reclat(wispr_outer_parameter[4][0][[2, 0, 1]])[1:3]))))

    ax.scatter(sun_pos[0], sun_pos[1], sun_pos[2], c='r')
    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    plt.title('PSP Frames' + '(' + obs_time_str + ')')
    ax.set_xlim([-0.7, 0.7])
    ax.set_ylim([-0.7, 0.7])
    ax.set_zlim([-0.7, 0.7])
    ax.legend(loc='lower right')
    plt.show()
    return


def find_PSP_in_WISPR(insitu_time_range, wispr_time):
    ''' find future PSP positions during insitu_time_range in WISPR image imaged at wispr_time'''
    wispr_pos, _ = spice.spkpos('SPP', wispr_time, 'SPP_HCI', 'NONE', 'SUN')
    insitu_pos, _ = spice.spkpos('SPP', insitu_time_range, 'SPP_HCI', 'NONE', 'SUN')
    sun_pos, _ = spice.spkpos('SUN', insitu_time_range, 'SPP_HCI', 'NONE', 'SUN')
    obs_pos = wispr_pos.T / AU
    insitu_pos = insitu_pos.T / AU

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(obs_pos[0], obs_pos[1], obs_pos[2], c='black')
    ax.plot(insitu_pos[0], insitu_pos[1], insitu_pos[2])
    wispr_inner_parameter = spice.getfov(-96100, 4)
    wispr_outer_parameter = spice.getfov(-96120, 4)
    print('inner_edge in deg')
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][0][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][1][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][2][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][3][[2, 0, 1]])[1:3]))
    print('outer_edge in deg')
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][0][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][1][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][2][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][3][[2, 0, 1]])[1:3]))

    for i_edge in range(4):
        edge_inner, _ = spice.spkcpt(1 * AU * wispr_inner_parameter[4][i_edge], 'SPP', 'SPP_WISPR_INNER', wispr_time,
                                     'SPP_HCI', 'CENTER', 'NONE', 'SUN')
        edge_outer, _ = spice.spkcpt(1 * AU * wispr_outer_parameter[4][i_edge], 'SPP', 'SPP_WISPR_OUTER', wispr_time,
                                     'SPP_HCI', 'CENTER', 'NONE', 'SUN')
        edge_inner = edge_inner / AU
        edge_outer = edge_outer / AU
        ax.plot([obs_pos[0], edge_inner[0]], [obs_pos[1], edge_inner[1]], [obs_pos[2], edge_inner[2]], c='green')
        ax.plot([obs_pos[0], edge_outer[0]], [obs_pos[1], edge_outer[1]], [obs_pos[2], edge_outer[2]], c='blue')

    inner_lst = []
    outer_lst = []
    print(spice.et2datetime(wispr_time).strftime('%Y-%m-%d %H:%M:%S'))
    for i in range(len(insitu_time_range)):
        print(i)
        insitu_time_tmp = insitu_time_range[i]
        insitu_utc_tmp = spice.et2datetime(insitu_time_tmp).strftime('%Y-%m-%d %H:%M:%S')

        # get pos of future PSP in the SPP_WISPR_INNER frame centered at SPP at wispr_time
        insitu_innerpos_tmp, _ = spice.spkcpt(insitu_pos[:, i] * AU, 'SUN', 'SPP_HCI', wispr_time, 'SPP_WISPR_INNER',
                                              'OBSERVER', 'NONE', 'SPP')
        insitu_outerpos_tmp, _ = spice.spkcpt(insitu_pos[:, i] * AU, 'SUN', 'SPP_HCI', wispr_time, 'SPP_WISPR_OUTER',
                                              'OBSERVER', 'NONE', 'SPP')

        # convert pos of future PSP to (r,lon,lat)
        insitu_innerpos_lat = spice.reclat(insitu_innerpos_tmp[[2, 0, 1]])
        insitu_outerpos_lat = spice.reclat(insitu_outerpos_tmp[[2, 0, 1]])

        print(np.rad2deg(insitu_innerpos_lat[1:3]))
        print(np.rad2deg(insitu_outerpos_lat[1:3]))

        '''
        <Equivalent to codes below, reserved for future debug.>
        if spice.fovray('SPP_WISPR_INNER',insitu_innerpos_tmp[0:3],'SPP_WISPR_INNER','NONE','SPP',wispr_time):
            print('PSP at '+insitu_utc_tmp+' Found in WISPR INNER FOV')
            inner_lst.append([i,insitu_innerpos_lat[0],insitu_innerpos_lat[1],insitu_innerpos_lat[2]])
            ax.scatter(insitu_pos[0,i],insitu_pos[1,i],insitu_pos[2,i],c='lime',alpha=0.5,s=10)

        if spice.fovray('SPP_WISPR_OUTER',insitu_outerpos_tmp[0:3],'SPP_WISPR_OUTER','NONE','SPP',wispr_time):
            print('PSP at'+insitu_utc_tmp+'Found in WISPR OUTER VOF')
            outer_lst.append([i,insitu_outerpos_lat[0],insitu_outerpos_lat[1],insitu_outerpos_lat[2]])
            ax.scatter(insitu_pos[0,i],insitu_pos[1,i],insitu_pos[2,i],c='aqua',alpha=0.5,s=10)
        '''
        # check whether PSP is in WISPR's FOV
        if spice.fovray('SPP_WISPR_INNER', insitu_pos[:, i] - obs_pos, 'SPP_HCI', 'NONE', 'SPP', wispr_time):
            print('PSP at ' + insitu_utc_tmp + ' Found in WISPR INNER FOV')
            inner_lst.append([insitu_time_tmp, insitu_innerpos_lat[0] / AU, np.rad2deg(insitu_innerpos_lat[1]),
                              np.rad2deg(insitu_innerpos_lat[2])])
            ax.scatter(insitu_pos[0, i], insitu_pos[1, i], insitu_pos[2, i], c='lime', alpha=0.5, s=20)

        if spice.fovray('SPP_WISPR_OUTER', insitu_pos[:, i] - obs_pos, 'SPP_HCI', 'NONE', 'SPP', wispr_time):
            print('PSP at' + insitu_utc_tmp + 'Found in WISPR OUTER VOF')
            outer_lst.append([insitu_time_tmp, insitu_outerpos_lat[0] / AU, np.rad2deg(insitu_outerpos_lat[1]),
                              np.rad2deg(insitu_outerpos_lat[2])])
            ax.scatter(insitu_pos[0, i], insitu_pos[1, i], insitu_pos[2, i], c='aqua', alpha=0.5, s=20)

    inner_lst = np.array(inner_lst)
    outer_lst = np.array(outer_lst)

    ax.scatter(sun_pos[0], sun_pos[1], sun_pos[2], c='r')
    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    plt.title('PSP (' + spice.et2datetime(insitu_time_range[0]).strftime('%Y-%m-%d') + '-' + spice.et2datetime(
        insitu_time_range[-1]).strftime('%Y-%m-%d') + ')\n WISPR image at ' + spice.et2datetime(wispr_time).strftime(
        '%Y-%m-%d'))
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    plt.close()
    # plt.show()

    return inner_lst, outer_lst


def find_WISPR_for_PSP(insitu_time, wispr_time_range):
    '''For the present PSP, find the past WISPR images.'''
    wispr_pos, _ = spice.spkpos('SPP', wispr_time_range, 'SPP_HCI', 'NONE', 'SUN')
    insitu_pos, _ = spice.spkpos('SPP', insitu_time, 'SPP_HCI', 'NONE', 'SUN')
    sun_pos, _ = spice.spkpos('SUN', insitu_time, 'SPP_HCI', 'NONE', 'SUN')
    obs_pos = wispr_pos.T / AU
    insitu_pos = insitu_pos.T / AU

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(insitu_pos[0], insitu_pos[1], insitu_pos[2], c='black')
    ax.plot(obs_pos[0], obs_pos[1], obs_pos[2])
    wispr_inner_parameter = spice.getfov(-96100, 4)
    wispr_outer_parameter = spice.getfov(-96120, 4)

    print('inner_edge in deg')
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][0][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][1][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][2][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_inner_parameter[4][3][[2, 0, 1]])[1:3]))
    print('outer_edge in deg')
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][0][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][1][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][2][[2, 0, 1]])[1:3]))
    print(np.rad2deg(spice.reclat(wispr_outer_parameter[4][3][[2, 0, 1]])[1:3]))

    inner_lst = []
    outer_lst = []
    for i in range(len(wispr_time_range)):
        print(i)
        wispr_time_tmp = wispr_time_range[i]
        wispr_utc_tmp = spice.et2datetime(wispr_time_tmp).strftime('%Y-%m-%d %H:%M:%S')

        insitu_innerpos_tmp, _ = spice.spkcpt(insitu_pos * AU, 'SUN', 'SPP_HCI', wispr_time_tmp, 'SPP_WISPR_INNER',
                                              'OBSERVER', 'NONE', 'SPP')
        insitu_outerpos_tmp, _ = spice.spkcpt(insitu_pos * AU, 'SUN', 'SPP_HCI', wispr_time_tmp, 'SPP_WISPR_OUTER',
                                              'OBSERVER', 'NONE', 'SPP')

        insitu_innerpos_lat = spice.reclat(insitu_innerpos_tmp[[2, 0, 1]])
        insitu_outerpos_lat = spice.reclat(insitu_outerpos_tmp[[2, 0, 1]])

        print(np.rad2deg(insitu_innerpos_lat[1:3]))
        print(np.rad2deg(insitu_outerpos_lat[1:3]))

        if spice.fovray('SPP_WISPR_INNER', insitu_pos - obs_pos[:, i], 'SPP_HCI', 'NONE', 'SPP', wispr_time_tmp):
            print('PSP Found in WISPR INNER VOF at ' + wispr_utc_tmp)
            inner_lst.append([wispr_time_tmp, insitu_innerpos_lat[0] / AU, np.rad2deg(insitu_innerpos_lat[1]),
                              np.rad2deg(insitu_innerpos_lat[2])])
            ax.scatter(obs_pos[0, i], obs_pos[1, i], obs_pos[2, i], c='lime', alpha=0.5, s=20)
            boresight_inner, _ = spice.spkcpt([0, 0, 0.5 * AU], 'SPP', 'SPP_WISPR_INNER', wispr_time_tmp, 'SPP_HCI',
                                              'CENTER', 'NONE', 'SUN')
            boresight_inner = boresight_inner / AU
            ax.plot([obs_pos[0, i], boresight_inner[0]], [obs_pos[1, i], boresight_inner[1]],
                    [obs_pos[2, i], boresight_inner[2]], c='green')
            ax.plot([obs_pos[0, i], insitu_pos[0]], [obs_pos[1, i], insitu_pos[1]], [obs_pos[2, i], insitu_pos[2]],
                    'k--')

        if spice.fovray('SPP_WISPR_OUTER', insitu_pos - obs_pos[:, i], 'SPP_HCI', 'NONE', 'SPP', wispr_time_tmp):
            print('PSP Found in WISPR OUTER VOF at ' + wispr_utc_tmp)
            outer_lst.append([wispr_time_tmp, insitu_outerpos_lat[0] / AU, np.rad2deg(insitu_outerpos_lat[1]),
                              np.rad2deg(insitu_outerpos_lat[2])])
            ax.scatter(obs_pos[0, i], obs_pos[1, i], obs_pos[2, i], c='aqua', alpha=0.5, s=20)
            boresight_outer, _ = spice.spkcpt([0, 0, 0.5 * AU], 'SPP', 'SPP_WISPR_OUTER', wispr_time_tmp, 'SPP_HCI',
                                              'CENTER', 'NONE', 'SUN')
            boresight_outer = boresight_outer / AU
            ax.plot([obs_pos[0, i], boresight_outer[0]], [obs_pos[1, i], boresight_outer[1]],
                    [obs_pos[2, i], boresight_outer[2]], c='blue')
            ax.plot([obs_pos[0, i], insitu_pos[0]], [obs_pos[1, i], insitu_pos[1]], [obs_pos[2, i], insitu_pos[2]],
                    'k--')

    inner_lst = np.array(inner_lst)
    outer_lst = np.array(outer_lst)

    ax.scatter(sun_pos[0], sun_pos[1], sun_pos[2], c='r')
    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    plt.title('PSP (' + spice.et2datetime(wispr_time_range[0]).strftime('%Y-%m-%d') + '-' + spice.et2datetime(
        wispr_time_range[-1]).strftime('%Y-%m-%d') + ')\n INSITU observation made at ' + spice.et2datetime(
        insitu_time).strftime('%Y-%m-%d'))
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    plt.show()
    return inner_lst, outer_lst


def WISPR_to_Carrington(R, lon_angle, lat_angle, wispr_time):
    xyz_wisprinner = [R * np.cos(np.deg2rad(lat_angle)) * np.sin(np.deg2rad(lon_angle)),
                      - R * np.sin(np.deg2rad(lat_angle)),
                      R * np.cos(np.deg2rad(lat_angle) * np.cos(np.deg2rad(lon_angle)))]  # km
    xyz_carrington, _ = spice.spkcpt(xyz_wisprinner, 'SPP', 'SPP_WISPR_INNER', wispr_time,
                                     'SPP_HG', 'CENTER', 'NONE', 'SUN')
    xyz_carrington /= Rs
    r_carrington, p_carrington, t_carrington = xyz2rtp_in_Carrington(xyz_carrington)
    # r_carrington = np.linalg.norm(xyz_carrington[0:3], 2)
    # if xyz_carrington[1] > 0:
    #     p_carrington = np.arccos(xyz_carrington[0] / r_carrington)
    # elif xyz_carrington[1] <= 0:
    #     p_carrington = 2 * np.pi - np.arccos(xyz_carrington[0] / r_carrington)
    #
    # t_carrington =np.pi/2- np.arccos(xyz_carrington[2] / r_carrington)
    return [r_carrington, p_carrington, t_carrington]


def Plot_Carrington_map_PSP(wispr_time_str, d, data_path):
    wispr_time = spice.datetime2et(datetime.strptime(wispr_time_str, '%Y%m%dT%H%M%S'))
    wispr_date = wispr_time_str[0:8]

    print('time:', wispr_time_str)
    print('distance from the Sun:', d)

    wispr_pos, _ = spice.spkpos('SPP', wispr_time, 'SPP_HCI', 'NONE', 'SUN')
    R = np.linalg.norm(wispr_pos, 2) / Rs  # Rs
    l = np.sqrt(R ** 2 - d ** 2)
    print('distance between PSP & Sun:', R)
    print('distance between TS point & PSP', l)

    lats_spp = np.deg2rad(np.linspace(-20, 20, 960))
    boresight = (960 / 2, 1024 / 2)
    # ps = np.arange(0, 360, 0.5)
    # ts = np.arange(-90, 90, 0.5)
    carrmap = np.zeros((len(ps), len(ts))) * np.nan

    path = data_path + wispr_date + '/'
    filelist = os.listdir(path)

    fnamelist = []
    for filename in filelist:
        if re.match(r'^psp_L3_wispr_' + wispr_time_str + '_V1_12', filename) != None:
            fnamelist.append(filename)
    data, header = sunpy.io.fits.read(path + fnamelist[0])[0]
    data = data.T

    t_indexs = []
    p_indexs = []
    msb = []
    x_indexs = []
    y_indexs = []
    for lat_spp in lats_spp:
        if l / R < np.cos(lat_spp):
            lon_spp = np.arccos(l / (R * np.cos(lat_spp)))

            TM_arr = spice.sxform('SPP_SPACECRAFT', 'SPP_WISPR_INNER', wispr_time)
            ray_spp = [np.cos(lat_spp) * np.sin(lon_spp),
                       -np.sin(lat_spp),
                       np.cos(lat_spp) * np.cos(lon_spp),
                       ]
            ray_wispr = np.dot(TM_arr[0:3, 0:3], ray_spp)

            lon_inner = np.rad2deg(np.arctan(ray_wispr[0] / ray_wispr[2]))
            lat_inner = np.rad2deg(-np.arcsin(ray_wispr[1]))

            r, p, t = WISPR_to_Carrington(l * Rs, lon_inner, lat_inner, wispr_time)

            x_inner = int(lon_inner * 960 / 40 + boresight[0])
            y_inner = int(lat_inner * 960 / 40 + boresight[1])
            x_indexs.append(x_inner)
            y_indexs.append(y_inner)

            if x_inner < 960 and y_inner < 960 and y_inner > 0 and x_inner > 0:
                # print('lonlat in spp', np.rad2deg(lon_spp), np.rad2deg(lat_spp))
                # print('lonlat in inner', lon_inner, lat_inner)
                # print('xy in inner', x_inner,y_inner)
                # print('pos',x_inner,y_inner)
                # print('lonlat in Carrington',np.rad2deg(p),np.rad2deg(t))
                # print('------')

                tmp_msb = data[x_inner, y_inner]
                data[x_inner, y_inner] = np.nan
                t_index = np.abs(ts - np.rad2deg(t)).argmin()
                p_index = np.abs(ps - np.rad2deg(p)).argmin()

                carrmap[p_index, t_index] = tmp_msb
                p_indexs.append(p_index)
                t_indexs.append(t_index)
                msb.append(tmp_msb)
    bin_msb = (p_indexs, t_indexs)
    value_msb = np.array(msb) * np.arccos(l / R)  # /(R**2)

    # plt.imshow(data.T)
    # plt.gca().invert_yaxis()
    # plt.clim([1e-14, 1e-12])
    # plt.title('PSP_WISPR_INNER ('+wispr_time_str+')')
    # plt.xlabel('x (pixels)')
    # plt.ylabel('z (pixels)')
    # plt.show()
    # exit()
    return bin_msb, value_msb


def Plot_Jmap_PSP(wispr_time_str, pre_wispr_time_str, data_path):
    wispr_time = spice.datetime2et(datetime.strptime(wispr_time_str, '%Y%m%dT%H%M%S'))
    wispr_date = wispr_time_str[0:8]
    pre_time = spice.datetime2et(datetime.strptime(pre_wispr_time_str, '%Y%m%dT%H%M%S'))
    pre_date = pre_wispr_time_str[0:8]

    # Load Fits and get Running Difference Data
    path = data_path + wispr_date + '/'

    filelist = os.listdir(path)
    fnamelist = []
    for filename in filelist:
        if re.match(r'^psp_L3_wispr_' + wispr_time_str + '_V1_12', filename) != None:
            fnamelist.append(filename)
    data, header = sunpy.io.fits.read(path + fnamelist[0])[0]
    data = data.T

    pre_path = data_path + pre_date + '/'
    pre_filelist = os.listdir(pre_path)

    pre_fnamelist = []
    for pre_filename in pre_filelist:
        if re.match(r'^psp_L3_wispr_' + pre_wispr_time_str + '_V1_12', pre_filename) != None:
            pre_fnamelist.append(pre_filename)
    pre_data, pre_header = sunpy.io.fits.read(pre_path + pre_fnamelist[0])[0]
    pre_data = pre_data.T

    rd_data = data - pre_data

    print('time:', wispr_time_str)
    # print('distance from the Sun:', d)

    boresight = (960 / 2, 1024 / 2)

    wispr_state, _ = spice.spkezr('SPP', wispr_time, 'SPP_HCI', 'NONE', 'SUN')
    sun_pos, _ = spice.spkpos('SUN', wispr_time, 'SPP_HCI', 'NONE', 'SUN')
    wispr_pos = wispr_state[0:3]
    wispr_velocity = wispr_state[3:6]

    R = np.linalg.norm(wispr_pos, 2) / Rs  # Rs

    sun_ray_spp = sun_pos - wispr_pos
    velocity_ray_spp = wispr_velocity

    TM_arr = spice.sxform('SPP_HCI', 'SPP_WISPR_INNER', wispr_time)
    sun_ray_wispr = np.dot(TM_arr[0:3, 0:3], sun_ray_spp)
    sun_ray_wispr = sun_ray_wispr / np.linalg.norm(sun_ray_wispr, 2)
    velocity_ray_wispr = np.dot(TM_arr[0:3, 0:3], velocity_ray_spp)
    velocity_ray_wispr = velocity_ray_wispr / np.linalg.norm(velocity_ray_wispr, 2)
    print('sun ray', sun_ray_wispr)
    sun_lon_inner = np.rad2deg(np.arctan(sun_ray_wispr[0] / sun_ray_wispr[2]))
    sun_lat_inner = np.rad2deg(-np.arcsin(sun_ray_wispr[1]))
    print('sun lon lat', sun_lon_inner, sun_lat_inner)

    sun_x_inner = int(sun_lon_inner * 960 / 40 + boresight[0])
    sun_y_inner = int(sun_lat_inner * 960 / 40 + boresight[1])
    print('sun xy', sun_x_inner, sun_y_inner)

    velocity_lon_inner = np.rad2deg(np.arctan(velocity_ray_wispr[0] / velocity_ray_wispr[2]))
    velocity_lat_inner = np.rad2deg(-np.arcsin(velocity_ray_wispr[1]))
    print('velocity lon lat', velocity_lon_inner, velocity_lat_inner)

    velocity_x_inner = int(velocity_lon_inner * 960 / 40 + boresight[0])
    velocity_y_inner = int(velocity_lat_inner * 960 / 40 + boresight[1])
    print('velocity xy', velocity_x_inner, velocity_y_inner)

    k = (velocity_y_inner - sun_y_inner) / (velocity_x_inner - sun_x_inner)

    x_indexs = np.linspace(0, 959, 960)
    # print(x_indexs)
    y_indexs = k * (x_indexs - sun_x_inner) + sun_y_inner
    x_indexs = x_indexs[y_indexs < 1024]
    y_indexs = y_indexs[y_indexs < 1024]

    value_msb = [rd_data[int(x_indexs[i]), int(y_indexs[i])] for i in range(len(x_indexs))]
    elon_msb = (x_indexs - sun_x_inner) * np.sqrt(1 + k ** 2) * 40 / 960

    return elon_msb, value_msb


def Plot_HEEQ_map_PSP(wispr_time_str, d):
    wispr_time = spice.datetime2et(datetime.strptime(wispr_time_str, '%Y%m%dT%H%M%S'))
    wispr_date = wispr_time_str[0:8]

    print('time:', wispr_time_str)
    print('distance from the Sun:', d)

    wispr_pos, _ = spice.spkpos('SPP', wispr_time, 'SPP_HCI', 'NONE', 'SUN')
    R = np.linalg.norm(wispr_pos, 2) / Rs  # Rs
    l = np.sqrt(R ** 2 - d ** 2)
    print('distance between PSP & Sun:', R)
    print('distance between TS point & PSP', l)

    lats_spp = np.deg2rad(np.arange(-30, 30, 0.05))
    boresight = (960 / 2, 1024 / 2)
    # ps = np.arange(0, 360, 0.5)
    # ts = np.arange(-90, 90, 0.5)
    carrmap = np.zeros((len(ps), len(ts))) * np.nan

    path = 'data/WISPR_ENC07_L3_FITS/' + wispr_date + '/'
    filelist = os.listdir(path)

    fnamelist = []
    for filename in filelist:
        if re.match(r'^psp_L3_wispr_' + wispr_time_str + '_V1_12', filename) != None:
            fnamelist.append(filename)
    data, header = sunpy.io.fits.read(path + fnamelist[0])[0]
    data = data.T

    t_indexs = []
    p_indexs = []
    msb = []
    x_indexs = []
    y_indexs = []
    for lat_spp in lats_spp:
        if l / R < np.cos(lat_spp):
            lon_spp = np.arccos(l / (R * np.cos(lat_spp)))
            # lon_spp = np.arccos(l/R)

            TM_arr = spice.sxform('SPP_SPACECRAFT', 'SPP_WISPR_INNER', wispr_time)
            ray_spp = [np.cos(lat_spp) * np.sin(lon_spp),
                       -np.sin(lat_spp),
                       np.cos(lat_spp) * np.cos(lon_spp),
                       ]
            ray_wispr = np.dot(TM_arr[0:3, 0:3], ray_spp)

            lon_inner = np.rad2deg(np.arctan(ray_wispr[0] / ray_wispr[2]))
            lat_inner = np.rad2deg(-np.arcsin(ray_wispr[1]))

            r, p, t = WISPR_to_Carrington(l * Rs, lon_inner, lat_inner, wispr_time)

            x_inner = int(lon_inner * 960 / 40 + boresight[0])
            y_inner = int(lat_inner * 960 / 40 + boresight[1])
            x_indexs.append(x_inner)
            y_indexs.append(y_inner)

            if x_inner < 960 and y_inner < 960 and y_inner > 0 and x_inner > 0:
                print('lonlat in spp', np.rad2deg(lon_spp), np.rad2deg(lat_spp))
                print('lonlat in inner', lon_inner, lat_inner)
                print('xy in inner', x_inner, y_inner)
                print('pos', x_inner, y_inner)
                print('lonlat in Carrington', np.rad2deg(p), np.rad2deg(t))
                print('------')

                tmp_msb = data[x_inner, y_inner]
                data[x_inner, y_inner] = np.nan
                t_index = np.abs(ts - np.rad2deg(t)).argmin()
                p_index = np.abs(ps - np.rad2deg(p)).argmin()

                carrmap[p_index, t_index] = tmp_msb
                p_indexs.append(p_index)
                t_indexs.append(t_index)
                msb.append(tmp_msb)
    bin_msb = (p_indexs, t_indexs)
    value_msb = msb * np.arccos(l / R)  # /l**2

    # plt.imshow(data.T)
    # plt.gca().invert_yaxis()
    # plt.clim([1e-14, 1e-12])
    # plt.title('PSP_WISPR_INNER ('+wispr_time_str+')')
    # plt.xlabel('x (pixels)')
    # plt.ylabel('z (pixels)')
    # plt.show()
    # exit()
    return bin_msb, value_msb


def load_wispr_inner_fits(path, datestr_list):
    '''filelist = os.listdir('data/orbit08/20210415')
    for filename in filelist:
        if re.match(r'^psp_L3_wispr_20210415T......_V1_12', filename) != None:
            fnamelist.append(filename)
    filelist = os.listdir('data/orbit08/20210416')
    for filename in filelist:
        if re.match(r'^psp_L3_wispr_20210416T......_V1_12', filename) != None:
            fnamelist.append(filename)'''
    fnamelist = []
    for datestr in datestr_list:
        filelist = os.listdir(path + datestr)
        for filename in filelist:
            if re.match(r'^psp_L3_wispr_' + datestr + 'T......_V1_12', filename) != None:
                fnamelist.append(filename)
    return fnamelist


if __name__ == "__main__":
    # plot_frames(et)
    # 'data/WISPR_ENC07_L3_FITS/'+wispr_date+'/psp_L3_wispr_'+wispr_time_str+'_V1_1211.fits'

    # Overview of PSP Orbit
    plot_psp_sun_carrington('20210426T000000', '20210429T000000')

    # Define files need reading
    fnamelist = []
    path = 'data/orbit08/'
    datetime_beg = datetime(2021, 4, 26)
    datetime_end = datetime(2021, 4, 27)

    datetime_length = int((datetime_end - datetime_beg) / timedelta(days=1))
    datetime_list = [datetime_beg + timedelta(days=i) for i in range(datetime_length)]
    datestr_list = [dt.strftime('%Y%m%d') for dt in datetime_list]
    fnamelist = load_wispr_inner_fits(path, datestr_list)
    fnamelist.sort(key=lambda x: datetime.strptime(x[13:28], '%Y%m%dT%H%M%S'))
    subepoch_beg = datetime(2021,4,26,3,30,0)
    subepoch_end = datetime(2021,4,27,0,0,0)
    # fnamelist = fnamelist[x for x in fnamelist if ((datetime.strptime(x[13:28],'%Y%m%dT%H%M%S')>subepoch_beg) & (datetime.strptime(x[13:28],'%Y%m%dT%H%M%S')<subepoch_end))]
    fnamelist = list(filter(lambda x: (datetime.strptime(x[13:28],'%Y%m%dT%H%M%S')>subepoch_beg) & (datetime.strptime(x[13:28],'%Y%m%dT%H%M%S')<subepoch_end),
                            fnamelist))

    # Set Resolutions
    # ps = np.arange(0, 360, 0.5)
    # ts = np.arange(-90, 90, 0.5)
    elons = np.linspace(13, 55, 960)

    plot_Jmap = False
    plot_Carrmap = True
    # plot_Timemap = True
    # Plot Jmaps
    if plot_Jmap:
        wispr_times = []
        jmap = np.zeros((len(fnamelist), len(elons))) * np.nan
        for i, fname in enumerate(fnamelist):
            print(i)
            print(fnamelist[i])
            print(fnamelist[i - 1])
            wispr_time_str = fname[13:28]

            if i == 0:
                print('i==0')
                pre_wispr_time_str = wispr_time_str
            else:
                pre_wispr_time_str = fnamelist[i - 1][13:28]
            wispr_times.append(datetime.strptime(wispr_time_str, '%Y%m%dT%H%M%S'))
            print('now', wispr_time_str)
            print('previous', pre_wispr_time_str)
            elon_msb, msb = Plot_Jmap_PSP(wispr_time_str, pre_wispr_time_str, path)
            # print(elon_msb)
            msb_interp = np.interp(elons, elon_msb, msb)
            jmap[i, :] = msb_interp

        plt.figure()
        TIME, ELON = np.meshgrid(wispr_times, elons)
        plt.pcolormesh(wispr_times, elons, jmap.T, cmap='gray')
        plt.clim([-2e-13, 2e-13])
        plt.colorbar()
        plt.show()

    if plot_Carrmap:
        wispr_times = []
        timemap = np.zeros((len(fnamelist), len(ts))) * np.nan
        carrmap = np.zeros((len(ps), len(ts))) * np.nan
        d = 10.5
        for i, fname in enumerate(fnamelist):
            wispr_time_str = fname[13:28]
            wispr_times.append(datetime.strptime(wispr_time_str, '%Y%m%dT%H%M%S'))
            bin_msb, msb = Plot_Carrington_map_PSP(wispr_time_str, d, path)
            # print(bin_msb)
            carrmap[bin_msb] = msb
            # print(bin_msb[1])
            timemap[i, bin_msb[1]] = msb

        pp, tt = np.meshgrid(ps, ts)
        print(carrmap.shape)
        print(pp.shape)
        # plt.ion()
        plt.figure()
        plt.pcolormesh(pp, tt, carrmap.T)
        plt.colorbar()
        plt.ylim([-60, 60])
        plt.xlim([100, 250])
        plt.gca().set_aspect(1)
        # plt.clim([1e-18, 1e-16])
        # plt.clim([1e-15, 1e-13])
        plt.clim([1e-14,1e-13])
        plt.xlabel('Carrington Longitude (deg)')
        plt.ylabel('Carrington Latitude (deg)')
        plt.title('Carrington Map (R=' + str(d) + 'Rs' + '_210111-210123)')
        # plt.text('Generated by wispr_insitu_TS.py')

        plt.show()

        plt.figure()
        time_xx, tt = np.meshgrid(wispr_times, ts)
        print(time_xx.shape,tt.shape,timemap.shape)
        plt.pcolormesh(time_xx, tt, timemap.T)
        plt.colorbar()
        plt.clim([1e-14, 1e-12])
        plt.ylabel('Carrington Latitude (deg)')
        plt.xlabel('Observation Time')
        plt.title('HEEQ Map (R=' + str(d) + 'Rs' + ')')
        plt.show()

    np.savez('Timemap_d=10.5_0426.npz',time_xx,tt,timemap)


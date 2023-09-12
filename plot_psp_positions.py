from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
import pandas as pd
from matplotlib import gridspec

from horizons_spk import get_spk_from_horizons
from scipy import interpolate
import furnsh_kernels

from datetime import timedelta

spice.furnsh('kernels/spp_nom_20180812_20250831_v039_RO6.bsp')
AU = 1.49e8  # km
Rs = 6.96e5  # km


def rtp2xyz_in_Carrington(rtp_carrington, for_psi=False):
    if for_psi:
        rtp_carrington[2] = np.pi / 2 - rtp_carrington[2]

    z_carrington = rtp_carrington[0] * np.cos(np.pi / 2 - rtp_carrington[2])
    y_carrington = rtp_carrington[0] * np.sin(np.pi / 2 - rtp_carrington[2]) * np.sin(rtp_carrington[1])
    x_carrington = rtp_carrington[0] * np.sin(np.pi / 2 - rtp_carrington[2]) * np.cos(rtp_carrington[1])
    return x_carrington, y_carrington, z_carrington


def xyz2rtp_in_Carrington(xyz_carrington, for_psi=False):
    """
    Convert (x,y,z) to (r,t,p) in Carrington Coordination System.
        (x,y,z) follows the definition of SPP_HG in SPICE kernel.
        (r,lon,lat) is (x,y,z) converted to heliographic lon/lat, where lon \in [0,2pi], lat \in [-pi/2,pi/2] .
    :param xyz_carrington:
    :return:
    """
    r_carrington = np.linalg.norm(xyz_carrington[0:3], 2)

    lon_carrington = np.arcsin(xyz_carrington[1] / np.sqrt(xyz_carrington[0] ** 2 + xyz_carrington[1] ** 2))
    if xyz_carrington[0] < 0:
        lon_carrington = np.pi - lon_carrington
    if lon_carrington < 0:
        lon_carrington += 2 * np.pi

    lat_carrington = np.pi / 2 - np.arccos(xyz_carrington[2] / r_carrington)
    if for_psi:
        lat_carrington = np.pi / 2 - lat_carrington
    return r_carrington, lon_carrington, lat_carrington


def plot_psp_sun_carrington(start_time_str, stop_time_str):
    print(datetime.strptime(start_time_str, '%Y%m%dT%H%M%S'))
    print(datetime.strptime(stop_time_str, '%Y%m%dT%H%M%S'))

    start_time = datetime.strptime(start_time_str, '%Y%m%dT%H%M%S')
    stop_time = datetime.strptime(stop_time_str, '%Y%m%dT%H%M%S')

    timestep = timedelta(hours=1)
    steps = (stop_time - start_time) // timestep + 1
    dttimes = np.array([x * timestep + start_time for x in range(steps)])
    dttimes_str = [dt.strftime('%m%d') for dt in dttimes[0:-1:24]]
    print(dttimes[0:-1:24])
    times = spice.datetime2et(dttimes)

    psp_pos_carr, _ = spice.spkpos('SPP', times, 'SPP_HG', 'NONE', 'SUN')
    psp_pos_carr = psp_pos_carr.T / Rs
    sun_pos_carr, _ = spice.spkpos('SUN', times, 'SPP_HG', 'NONE', 'SUN')

    psp_pos_carr_rtp = np.array([xyz2rtp_in_Carrington(psp_pos_carr[:, i]) for i in range(len(psp_pos_carr.T))])
    psp_pos_carr_rtp = psp_pos_carr_rtp.T
    print('Start Point:', psp_pos_carr_rtp[0][0], np.rad2deg(psp_pos_carr_rtp[1][0]),
          np.rad2deg(psp_pos_carr_rtp[2][0]))
    print('End Point:', psp_pos_carr_rtp[0][-1], np.rad2deg(psp_pos_carr_rtp[1][-1]),
          np.rad2deg(psp_pos_carr_rtp[2][-1]))

    # fig = plt.figure(figsize=(9, 9))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(psp_pos_carr[0], psp_pos_carr[1], psp_pos_carr[2], c='blue')
    # ax.scatter(0, 0, 0, c='red')
    # ax.scatter(psp_pos_carr[0, -1], psp_pos_carr[1, -1], psp_pos_carr[2, -1], c='blue')
    #
    # ax.plot([0, psp_pos_carr[0][0]], [0, psp_pos_carr[1][0]], [0, psp_pos_carr[2][0]], c='yellow')
    #
    # ax.set_xlim([-75, 75])
    # ax.set_ylim([-75, 75])
    # ax.set_zlim([-75, 75])
    # plt.title('PSP orbit (' + start_time_str + '-' + stop_time_str + '), Carrington')
    # ax.set_xlabel('x in Carrington (Rs)')
    # ax.set_ylabel('y in Carrington (Rs)')
    # ax.set_zlabel('z in Carrington (Rs)')
    # plt.show()

    # plt.figure()
    # plt.scatter(np.rad2deg(psp_pos_carr_rtp[1]), np.rad2deg(psp_pos_carr_rtp[2]), c=psp_pos_carr_rtp[0])
    # plt.colorbar(label='Radial Distance')
    # plt.clim([20, 60])
    # plt.scatter(np.rad2deg(psp_pos_carr_rtp[1, -1]), np.rad2deg(psp_pos_carr_rtp[2, -1]), c='black')
    # plt.xlabel('Carrington Longitude')
    # plt.ylabel('Carrington Latitude')
    # plt.title('PSP Orbit in HG')
    # plt.show()

    plt.figure()

    ax = plt.subplot(121, projection='polar')
    ax.plot(psp_pos_carr_rtp[1], psp_pos_carr_rtp[0], 'k-')
    ax.scatter(psp_pos_carr_rtp[1, 0:-1:24], psp_pos_carr_rtp[0, 0:-1:24], marker='x', c='black')
    for i in range(len(dttimes_str)):
        print(dttimes_str[i])
        ax.annotate(dttimes_str[i], (psp_pos_carr_rtp[1, 0:-1:24][i], psp_pos_carr_rtp[0, 0:-1:24][i]))
    ax.scatter(0, 0, c='red', s=100)
    ax.set_axisbelow('True')
    ax.set_thetagrids(np.arange(0.0, 360.0, 15.0))
    # ax.set_thetamin(30.0)  # 设置极坐标图开始角度为0°
    # ax.set_thetamax(210.0)  # 设置极坐标结束角度为180°
    # ax.set_rmax(2)
    # ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    ax.set_xlabel('Radius [Rs]')
    ax.set_ylabel('Carrington Longitude [deg]')
    ax.set_rlabel_position(0)  # Move radial labels away from plotted line
    ax.grid(True)
    # plt.show()

    # plt.figure()
    plt.subplot(2, 2, 2)
    # psp_pos_carr_rtp[1][psp_pos_carr_rtp[1]>np.deg2rad(300)] -= 2*np.pi
    plt.plot(spice.et2datetime(times), np.rad2deg(psp_pos_carr_rtp[1]))
    plt.xlabel('Observation Time')
    plt.ylabel('Carrington Longitude [deg]')
    plt.subplot(2, 2, 4)
    plt.plot(spice.et2datetime(times), psp_pos_carr_rtp[0])
    plt.xlabel('Observation Time')
    plt.ylabel('Solar Radius [Rs]')
    plt.savefig('SAVED.png')
    plt.show()


def plot_psp_sun_hci(start_time_str, stop_time_str, step=100):
    print(datetime.strptime(start_time_str, '%Y%m%dT%H%M%S'))
    print(datetime.strptime(stop_time_str, '%Y%m%dT%H%M%S'))
    start_et = spice.datetime2et(datetime.strptime(start_time_str, '%Y%m%dT%H%M%S'))
    stop_et = spice.datetime2et(datetime.strptime(stop_time_str, '%Y%m%dT%H%M%S'))

    # step = 100
    times = [x * (stop_et - start_et) / step + start_et for x in range(step)]

    psp_pos_hci, _ = spice.spkpos('PSP', times, 'SPP_HCI', 'NONE', 'SUN')
    psp_pos_hci = psp_pos_hci.T / Rs
    sun_pos_hci, _ = spice.spkpos('SUN', times, 'SPP_HCI', 'NONE', 'SUN')

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(psp_pos_hci[0], psp_pos_hci[1], psp_pos_hci[2], '-.', c='blue')
    ax.scatter(0, 0, 0, c='red')
    ax.scatter(psp_pos_hci[0, -1], psp_pos_hci[1, -1], psp_pos_hci[2, -1], c='blue')

    ax.plot([0, psp_pos_hci[0][0]], [0, psp_pos_hci[1][0]], [0, psp_pos_hci[2][0]], c='yellow')

    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])
    ax.set_zlim([-200, 200])
    plt.title('PSP orbit (' + start_time_str + '-' + stop_time_str + '), HCI')
    ax.set_xlabel('x in HCI (Rs)')
    ax.set_ylabel('y in HCI (Rs)')
    ax.set_zlabel('z in HCI (Rs)')
    plt.show()

    plt.figure()
    plt.plot(psp_pos_hci[0], psp_pos_hci[1], 'k-')
    plt.scatter(sun_pos_hci[0], sun_pos_hci[1], c='r')
    plt.xlim([-200, 200])
    plt.ylim([-200, 200])
    plt.xlabel('x(Rs) in HCI')
    plt.ylabel('y(Rs) in HCI')
    plt.show()


def get_rlonlat_psp_carr(dt_times, for_psi=False):
    r_psp_carr = []
    lon_psp_carr = []
    lat_psp_carr = []
    for dt in dt_times:
        et = spice.datetime2et(dt)
        psp_pos, _ = spice.spkpos('SPP', et, 'IAU_SUN', 'NONE', 'SUN')  # km
        psp_pos = psp_pos.T / Rs
        r_psp = np.linalg.norm(psp_pos[0:3], 2)
        if psp_pos[1] > 0:
            p_psp = np.arccos(psp_pos[0] / r_psp)
        elif psp_pos[1] <= 0:
            p_psp = 2 * np.pi - np.arccos(psp_pos[0] / r_psp)
        t_psp = np.arccos(psp_pos[2] / r_psp)
        # print('rtp',r_psp,t_psp,p_psp)
        r_psp, p_psp, t_psp = xyz2rtp_in_Carrington(psp_pos, for_psi=for_psi)
        # t_psp = np.pi/2-t_psp
        # print('xyz2rtp',r_psp,t_psp,p_psp)
        r_psp_carr.append(r_psp)
        lon_psp_carr.append(p_psp)
        lat_psp_carr.append(t_psp)
    return np.array(r_psp_carr), np.array(lon_psp_carr), np.array(lat_psp_carr)


def write_trajectory_file_for_SWMF(dt_beg, dt_end, dt_resolution, target='PSP', coord='HGR', norm_unit='Rs'):
    dts = [dt_beg + n * dt_resolution for n in range((dt_end - dt_beg) // dt_resolution)]
    ets = spice.datetime2et(dts)
    if coord == 'HGR' or coord == 'HG':
        coord_spice = 'SPP_HG'
    elif coord == 'HCI':
        coord_spice = 'SPP_HCI'

    print('> Using ' + coord + ' for SWMF, aka ' + coord_spice + ' for SPICE... <')

    if target == 'PSP':
        target = 'SPP'
        print('> Target is ' + target + '... <')
    psp_pos, _ = spice.spkezr(target, ets, coord_spice, 'NONE', 'SUN')
    psp_pos = np.array(psp_pos)
    if norm_unit == 'Rs':
        psp_pos = psp_pos.T / Rs
    elif norm_unit == 'km':
        psp_pos = psp_pos.T / Rs
    print('> Normalize to ' + norm_unit + ' and ' + norm_unit + '/s ... <')

    dt_beg_str = dt_beg.strftime('%Y%m%dT%H%M%S')
    dt_end_str = dt_end.strftime('%Y%m%dT%H%M%S')

    filepath = '/Users/ephe/THL8/traj_data/'
    filename = target + '_(' + coord + ')_(' + dt_beg_str + '-' + dt_end_str + ').dat'

    with open(filepath + filename, 'w') as f:
        print('> Writing the trajectory of ' + target + ' from ' + dt_beg_str + ' to ' + dt_end_str + ' to File: <')
        print('> ' + filepath + filename + ' <')
        f.write('#COOR\n')
        f.write(coord + '\n')
        f.write('\n')
        f.write('#START\n')
        for i in range(len(dts)):
            dt_tmp = dts[i]
            f.write(str(dt_tmp.year) + '\t' + str(dt_tmp.month) + '\t' + str(dt_tmp.day) + '\t'
                    + str(dt_tmp.hour) + '\t' + str(dt_tmp.minute) + '\t' + str(dt_tmp.second) + '\t' + str(
                dt_tmp.microsecond)
                    + '\t' + str(psp_pos[0][i]) + '\t' + str(psp_pos[1][i]) + '\t' + str(psp_pos[2][i])
                    + '\t' + str(psp_pos[3][i]) + '\t' + str(psp_pos[4][i]) + '\t' + str(psp_pos[5][i]) + '\n')


if __name__ == '__main__':
    # plot_psp_sun_carrington('20210801T000000','20210824T000000')
    dt_beg = datetime(2020, 1, 1)
    dt_end = datetime(2020, 5, 1)
    dt_resolution = timedelta(hours=6)

    write_trajectory_file_for_SWMF(dt_beg, dt_end, dt_resolution, target='PSP', coord='HCI', norm_unit='km')

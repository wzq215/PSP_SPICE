"""
Target:
    Get and plot trajectories of asteroids, planets, PSP and other objects in various coordination systems.
Functions:
    xyz2rtp_in_Carrington(xyz_carrington)
    get_aster_pos(spkid, start_time, stop_time, observer='Earth BARYCENTER', frame='SPP_GSE', plot=True)
    get_body_pos(body, start_time, stop_time, observer='SUN BARYCENTER', frame='SPP_GSE', plot=True)
    plot_psp_sun_carrington(start_time_str, stop_time_str)
Versions:
    21/03/22 file rearranged by Ziqu Wu.
"""

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
import pandas as pd
from matplotlib import gridspec

from horizons_spk import get_spk_from_horizons
from scipy import interpolate
import furnsh_kernels
# from load_psp_data import load_spi_data
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

    lon_carrington = np.arcsin(xyz_carrington[1]/np.sqrt(xyz_carrington[0]**2+xyz_carrington[1]**2))
    if xyz_carrington[0] < 0:
        lon_carrington = np.pi - lon_carrington
    if lon_carrington < 0:
        lon_carrington += 2*np.pi

    lat_carrington = np.pi / 2 - np.arccos(xyz_carrington[2] / r_carrington)
    if for_psi:
        lat_carrington = np.pi/2-lat_carrington
    return r_carrington, lon_carrington, lat_carrington


def get_aster_pos(spkid, start_time, stop_time, observer='Earth BARYCENTER', frame='SPP_GSE', plot=True):
    # Needed for Asteroids
    spkfileid = get_spk_from_horizons(spkid, start_time, stop_time)
    spice.furnsh('kernels/' + str(spkfileid) + '(' + start_time + '_' + stop_time + ')' + '.bsp')

    # UTC2ET
    start_dt = datetime.strptime(start_time, '%Y-%m-%d')
    stop_dt = datetime.strptime(stop_time, '%Y-%m-%d')
    utc = [start_dt.strftime('%b %d, %Y'), stop_dt.strftime('%b %d, %Y')]
    etOne = spice.str2et(utc[0])
    etTwo = spice.str2et(utc[1])

    # Epochs
    step = 365
    times = [x * (etTwo - etOne) / step + etOne for x in range(step)]

    # Get positions
    positions, LightTimes = spice.spkpos(str(spkfileid), times, frame, 'NONE', observer)

    positions = positions.T  # positions is shaped (4000, 3), let's transpose to (3, 4000) for easier indexing
    positions = positions / AU
    if plot:
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(positions[0], positions[1], positions[2])
        ax.scatter(1, 0, 0, c='red')
        ax.set_xlabel('X (AU)')
        ax.set_ylabel('Y (AU)')
        ax.set_zlabel('Z (AU)')
        plt.title('SPKID=' + str(spkid) + '(' + start_time + '_' + stop_time + ')')
        plt.show()
    return positions


def get_body_pos(body, start_time, stop_time, observer='SUN BARYCENTER', frame='SPP_GSE', plot=True):
    # UTC2ET
    start_dt = datetime.strptime(start_time, '%Y-%m-%d')
    stop_dt = datetime.strptime(stop_time, '%Y-%m-%d')
    utc = [start_dt.strftime('%b %d, %Y'), stop_dt.strftime('%b %d, %Y')]
    etOne = spice.str2et(utc[0])
    etTwo = spice.str2et(utc[1])

    # Epochs
    step = 365
    times = [x * (etTwo - etOne) / step + etOne for x in range(step)]

    # Get positions
    positions, LightTimes = spice.spkpos(body, times, frame, 'NONE', observer)

    AU = 1.49e8  # distance from sun to earth
    positions = positions.T  # positions is shaped (4000, 3), let's transpose to (3, 4000) for easier indexing
    positions = positions / AU
    if plot:
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(positions[0], positions[1], positions[2])
        ax.scatter(1, 0, 0, c='red')
        ax.set_xlabel('X (AU)')
        ax.set_ylabel('Y (AU)')
        ax.set_zlabel('Z (AU)')
        plt.title('Body_Name=' + body + '(' + start_time + '_' + stop_time + ')')
        plt.show()
    return positions


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




def psp_backmap(epoch, r_source_surface_rs=2.5):
    pmom_SPI = load_spi_data(epoch[0].strftime('%Y%m%d'), epoch[-1].strftime('%Y%m%d'))
    epochpmom = pmom_SPI['Epoch']
    timebinpmom = (epochpmom > epoch[0]) & (epochpmom < epoch[-1])
    epochpmom = epochpmom[timebinpmom]
    densp = pmom_SPI['DENS'][timebinpmom]
    vp_r = pmom_SPI['VEL_RTN_SUN'][timebinpmom, 0]
    vp_t = pmom_SPI['VEL_RTN_SUN'][timebinpmom, 1]
    vp_n = pmom_SPI['VEL_RTN_SUN'][timebinpmom, 2]
    # r_source_surface_rs = 2.
    dir_data_PFSS = '/Users/ephe/PFSS_data/'

    # Vsw_r_interp = np.interp(np.array((epoch-epochpmom[0])/timedelta(days=1),dtype='float64'),np.array((epochpmom-epochpmom[0])/timedelta(days=1),dtype='float64'),vp_r)

    f = interpolate.interp1d(np.array((epochpmom - epochpmom[0]) / timedelta(days=1), dtype='float64'), vp_r,
                             kind='previous', fill_value='extrapolate')

    Vsw_r_interp = f(np.array((epoch - epochpmom[0]) / timedelta(days=1),
                              dtype='float64'))  # use interpolation function returned by `interp1d`
    plt.plot(np.array((epochpmom - epochpmom[0]) / timedelta(days=1), dtype='float64'))
    plt.plot(np.array((epoch - epochpmom[0]) / timedelta(days=1), dtype='float64'))
    plt.show()
    plt.plot(epochpmom, vp_r)
    plt.plot(epoch, Vsw_r_interp)
    plt.show()

    ets = spice.datetime2et(epoch)
    psp_pos_carr, _ = spice.spkpos('SPP', ets, 'SPP_HG', 'NONE', 'SUN')
    psp_pos_carr = psp_pos_carr.T / Rs
    psp_pos_carr_rtp = np.array([xyz2rtp_in_Carrington(psp_pos_carr[:, i]) for i in range(len(psp_pos_carr.T))])
    psp_pos_carr_rtp = psp_pos_carr_rtp.T

    r_footpoint_on_SourceSurface_rs = np.arange(len(epoch)) * np.nan
    lon_footpoint_on_SourceSurface_deg = np.arange(len(epoch)) * np.nan
    lat_footpoint_on_SourceSurface_deg = np.arange(len(epoch)) * np.nan
    MFL_photosphere_lon_deg = np.arange(len(epoch)) * np.nan
    MFL_photosphere_lat_deg = np.arange(len(epoch)) * np.nan
    field_lines = []

    from two_step_ballistic_backmapping_method import two_step_backmapping
    for i, datetime_trace in enumerate(epoch):
        r_beg_au = psp_pos_carr_rtp[0, i] * Rs / AU
        lat_beg_deg = np.rad2deg(psp_pos_carr_rtp[2, i])
        lon_beg_deg = np.rad2deg(psp_pos_carr_rtp[1, i])
        Vsw_r = Vsw_r_interp[i]
        print('Vsw_r', Vsw_r)
        r_footpoint_on_SourceSurface_rs[i], lon_footpoint_on_SourceSurface_deg[i], lat_footpoint_on_SourceSurface_deg[
            i], \
        MFL_photosphere_lon_deg[i], MFL_photosphere_lat_deg[i], field_line \
            = two_step_backmapping(datetime_trace, r_beg_au, lat_beg_deg, lon_beg_deg, Vsw_r, r_source_surface_rs,
                                   dir_data_PFSS)
        field_lines.append(field_line)

    plt.figure()
    plt.scatter(MFL_photosphere_lon_deg, MFL_photosphere_lat_deg, c=np.arange(len(epoch)))
    plt.colorbar()
    plt.show()
    return r_footpoint_on_SourceSurface_rs, lon_footpoint_on_SourceSurface_deg, lat_footpoint_on_SourceSurface_deg, \
           MFL_photosphere_lon_deg, MFL_photosphere_lat_deg, field_lines


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


def get_rlonlat_solo_carr(dt_times, for_psi=False):
    r_solo_carr = []
    lon_solo_carr = []
    lat_solo_carr = []
    for dt in dt_times:
        et = spice.datetime2et(dt)
        solo_pos, _ = spice.spkpos('SOLO', et, 'IAU_SUN', 'NONE', 'SUN')  # km
        solo_pos = solo_pos.T / Rs

        r_solo, p_solo, t_solo = xyz2rtp_in_Carrington(solo_pos, for_psi=for_psi)
        r_solo_carr.append(r_solo)
        lon_solo_carr.append(p_solo)
        lat_solo_carr.append(t_solo)
    return np.array(r_solo_carr), np.array(lon_solo_carr), np.array(lat_solo_carr)


def get_rlonlat_earth_carr(dt_times, for_psi=False):
    r_earth_carr = []
    lon_earth_carr = []
    lat_earth_carr = []
    for dt in dt_times:
        et = spice.datetime2et(dt)
        earth_pos, _ = spice.spkpos('EARTH', et, 'IAU_SUN', 'NONE', 'SUN')  # km
        earth_pos = earth_pos.T / Rs

        r_earth, p_earth, t_earth = xyz2rtp_in_Carrington(earth_pos, for_psi=for_psi)
        r_earth_carr.append(r_earth)
        lon_earth_carr.append(p_earth)
        lat_earth_carr.append(t_earth)
    return np.array(r_earth_carr), np.array(lon_earth_carr), np.array(lat_earth_carr)


def write_trajectory_file_for_SWMF(dt_beg, dt_end, dt_resolution, target='PSP', coord='HGR', norm_unit='Rs'):
    dts = [dt_beg + n * dt_resolution for n in range((dt_end - dt_beg) // dt_resolution)]
    ets = spice.datetime2et(dts)
    if coord == 'HGR':
        coord_spice = 'SPP_HG'
        print('Using ' + coord + ' for SWMF, aka ' + coord_spice + ' for SPICE...')
    if target == 'PSP':
        target = 'SPP'
        print('Target is ' + target + '...')
    psp_pos, _ = spice.spkpos(target, ets, coord_spice, 'NONE', 'SUN')
    if norm_unit == 'Rs':
        print('Normalize to ' + norm_unit + '...')
    psp_pos = psp_pos.T / Rs
    dt_beg_str = dt_beg.strftime('%Y%m%dT%H%M%S')
    dt_end_str = dt_end.strftime('%Y%m%dT%H%M%S')
    filepath = '/Users/ephe/THL8/traj_data/'
    filename = target + '_(' + coord + ')_(' + dt_beg_str + '-' + dt_end_str + ').dat'
    with open(filepath + filename, 'w') as f:
        print('Writing the trajectory of ' + target + ' from ' + dt_beg_str + ' to ' + dt_end_str + ' to File:')
        print(filepath + filename)
        f.write('#COOR\n')
        f.write(coord + '\n')
        f.write('\n')
        f.write('#START\n')
        for i in range(len(dts)):
            dt_tmp = dts[i]
            f.write(str(dt_tmp.year) + '\t' + str(dt_tmp.month) + '\t' + str(dt_tmp.day) + '\t'
                    + str(dt_tmp.hour) + '\t' + str(dt_tmp.minute) + '\t' + str(dt_tmp.second) + '\t' + str(
                dt_tmp.microsecond)
                    + '\t' + str(psp_pos[0][i]) + '\t' + str(psp_pos[1][i]) + '\t' + str(psp_pos[2][i]) + '\n')


if __name__ == '__main__':
    # %%
    #     get_rlonlat_solo_carr(np.array([datetime(2022,2,28),datetime(2022,3,1),datetime(2022,3,2)]))
    # %%
    # dt_beg = datetime(2023,3,15,0)
    # dt_end = datetime(2023,3,21,0)
    # dt_res = timedelta(minutes=30)
    # write_trajectory_file_for_SWMF(dt_beg,dt_end,dt_res)
    # plot_psp_sun_carrington('20211110T000000','20211130T000000')
    plot_psp_sun_carrington('20220216T000000', '20220312T000000')
    # plot_psp_sun_carrington('20210801T000000','20210824T000000')
    # quit()
    #
    # start_time_str = '20220217T120000'
    # stop_time_str = '20220311T120000'
    #
    # start_time = datetime.strptime(start_time_str, '%Y%m%dT%H%M%S')
    # stop_time = datetime.strptime(stop_time_str, '%Y%m%dT%H%M%S')
    # from load_psp_data import load_RTN_4sa_data,load_RTN_1min_data
    #
    #
    # timestep = timedelta(hours=1)
    # steps = (stop_time - start_time) // timestep + 1
    # dttimes = np.array([x * timestep + start_time for x in range(steps)])
    # print(dttimes[0:-1:24])
    # times = spice.datetime2et(dttimes)
    #
    # psp_pos_carr, _ = spice.spkpos('SPP', times, 'SPP_HG', 'NONE', 'SUN')
    # psp_pos_carr = psp_pos_carr.T / Rs
    # sun_pos_carr, _ = spice.spkpos('SUN', times, 'SPP_HG', 'NONE', 'SUN')
    #
    # psp_pos_carr_rtp = np.array([xyz2rtp_in_Carrington(psp_pos_carr[:, i]) for i in range(len(psp_pos_carr.T))])
    # psp_pos_carr_rtp = psp_pos_carr_rtp.T
    # print('Start Point:', psp_pos_carr_rtp[0][0], np.rad2deg(psp_pos_carr_rtp[1][0]),
    #       np.rad2deg(psp_pos_carr_rtp[2][0]))
    # print('End Point:', psp_pos_carr_rtp[0][-1], np.rad2deg(psp_pos_carr_rtp[1][-1]),
    #       np.rad2deg(psp_pos_carr_rtp[2][-1]))
    #
    # fig = plt.figure(figsize=(9, 9))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(psp_pos_carr[0], psp_pos_carr[1], psp_pos_carr[2], c='blue')
    # ax.scatter(0, 0, 0, c='red')
    # ax.scatter(psp_pos_carr[0, -1], psp_pos_carr[1, -1], psp_pos_carr[2, -1], c='blue')
    #
    # ax.plot([0, psp_pos_carr[0][0]], [0, psp_pos_carr[1][0]], [0, psp_pos_carr[2][0]], c='yellow')
    #
    # start = [datetime(2022,2,17,16,45,0),datetime(2022,2,25,12,25,0),datetime(2022,3,11,6,15,0)]
    # stop = [datetime(2022,2,17,23,30,0),datetime(2022,2,25,12,40,0),datetime(2022,3,11,12,0,0)]
    # for i in range(3):
    #     mag_RTN = load_RTN_1min_data(start[i].strftime('%Y%m%d'), stop[i].strftime('%Y%m%d'))
    #
    #     epochmag = mag_RTN['epoch_mag_RTN_1min']
    #     timebinmag = (epochmag > start[i]) & (epochmag < stop[i])
    #     epochmag = epochmag[timebinmag]
    #
    #     mag_rtn = mag_RTN['psp_fld_l2_mag_RTN_1min'][timebinmag, :]
    #     Babs = np.sqrt(mag_rtn[:,0] ** 2 + mag_rtn[:,1] ** 2 + mag_rtn[:,2] ** 2)
    #     etmag = spice.datetime2et(epochmag)
    #     TM_arr = spice.sxform('SPP_RTN', 'SPP_HG', etmag)
    #     print(TM_arr.shape)
    #     mag_hg = np.array([np.dot(TM_arr[j,0:3, 0:3], mag_rtn[j,:]) for j in range(len(etmag))])
    #     print(mag_hg.shape)
    #     print(psp_pos_carr)
    #     # for j in range(len(etmag)):
    #     #     mag_hg = np.dot(TM_arr[j,0:3, 0:3], mag_rtn[j,:])
    #     mag_pos_carr, _ = spice.spkpos('SPP', etmag, 'SPP_HG', 'NONE', 'SUN')
    #     mag_pos_carr = mag_pos_carr.T / Rs
    #     mag_pos_carr_rtp = np.array([xyz2rtp_in_Carrington(mag_pos_carr[:, i]) for i in range(len(mag_pos_carr.T))])
    #     mag_pos_carr_rtp = mag_pos_carr_rtp.T
    #
    #     ax.quiver(mag_pos_carr[0][mag_rtn[:,0]>0], mag_pos_carr[1][mag_rtn[:,0]>0], mag_pos_carr[2][mag_rtn[:,0]>0],mag_hg[:,0][mag_rtn[:,0]>0]/Babs[mag_rtn[:,0]>0]*10,mag_hg[:,1][mag_rtn[:,0]>0]/Babs[mag_rtn[:,0]>0]*10,mag_hg[:,2][mag_rtn[:,0]>0]/Babs[mag_rtn[:,0]>0]*10,linewidth=0.2,color='red')
    #     ax.quiver(mag_pos_carr[0][mag_rtn[:,0]<0], mag_pos_carr[1][mag_rtn[:,0]<0], mag_pos_carr[2][mag_rtn[:,0]<0],mag_hg[:,0][mag_rtn[:,0]<0]/Babs[mag_rtn[:,0]<0]*10,mag_hg[:,1][mag_rtn[:,0]<0]/Babs[mag_rtn[:,0]<0]*10,mag_hg[:,2][mag_rtn[:,0]<0]/Babs[mag_rtn[:,0]<0]*10,linewidth=0.2,color='blue')
    #     # plt.colorbar()
    #     # plt.set_clim([-1,1])
    #     import pandas as pd
    #     df = pd.DataFrame()
    #     df['epoch'] = epochmag
    #     df['pos_x'] = mag_pos_carr[0]
    #     df['pos_y'] = mag_pos_carr[1]
    #     df['pos_z'] = mag_pos_carr[2]
    #     df['pos_r'] = mag_pos_carr_rtp[0]
    #     df['pos_lon'] = np.rad2deg(mag_pos_carr_rtp[1])
    #     df['pos_lat'] = np.rad2deg(mag_pos_carr_rtp[2])
    #     df['mag_r'] = mag_rtn[:,0]
    #     df['mag_t'] = mag_rtn[:,1]
    #     df['mag_n'] = mag_rtn[:,2]
    #     df['mag_x'] = mag_hg[:,0]
    #     df['mag_y'] = mag_hg[:,1]
    #     df['mag_z'] = mag_hg[:,2]
    #     df['mag_tot'] = Babs
    #     df.to_csv('mag'+str(i)+'.csv')
    #
    # ax.set_xlim([-5, -3])
    # ax.set_ylim([12, 14])
    # ax.set_zlim([-1, 1])
    # plt.title('PSP orbit (' + start_time_str + '-' + stop_time_str + '), Carrington')
    # ax.set_xlabel('x in Carrington (Rs)')
    # ax.set_ylabel('y in Carrington (Rs)')
    # ax.set_zlabel('z in Carrington (Rs)')
    # plt.show()
    # exit()

    datetime_beg = datetime(2022, 3, 11, 6, 15, 0)
    datetime_end = datetime(2022, 3, 11, 12, 0, 0)
    # datetime_beg = datetime(2022,2,17,16,45,0)
    # datetime_end = datetime(2022,2,17,23,30,0)
    # datetime_beg = datetime(2022,2,25,12,25,0)
    # datetime_end = datetime(2022,2,25,12,40,0)
    timestep = timedelta(minutes=15)

    timestr_beg = datetime_beg.strftime('%Y%m%dT%H%M%S')
    timestr_end = datetime_end.strftime('%Y%m%dT%H%M%S')

    steps = (datetime_end - datetime_beg) // timestep + 1
    epoch = np.array([x * timestep + datetime_beg for x in range(steps)])
    r_footpoint_on_SourceSurface_rs, lon_footpoint_on_SourceSurface_deg, lat_footpoint_on_SourceSurface_deg, MFL_photosphere_lon_deg, MFL_photosphere_lat_deg, field_lines = psp_backmap(
        epoch, r_source_surface_rs=2.5)

    df = pd.DataFrame()
    df['Epoch'] = epoch
    df['r_footpoint_on_SourceSurface_rs'] = r_footpoint_on_SourceSurface_rs
    df['lon_footpoint_on_SourceSurface_deg'] = lon_footpoint_on_SourceSurface_deg
    df['lat_footpoint_on_SourceSurface_deg'] = lat_footpoint_on_SourceSurface_deg
    df['MFL_photosphere_lon_deg'] = MFL_photosphere_lon_deg
    df['MFL_photosphere_lat_deg'] = MFL_photosphere_lat_deg

    fl_coords = []
    fl_expansions = []
    for fl in field_lines:
        try:
            fl_coords.append(fl.coords)
            fl_expansions.append(fl.expansion_factor)
        except:
            fl_coords.append(np.nan)
            fl_expansions.append(np.nan)
    df['Expansion_Factor'] = fl_expansions
    df.to_csv('export/plot_body_positions/pfss_trace/trace_fps_(' + timestr_beg + '-' + timestr_end + '-' + str(
        timestep // timedelta(minutes=1)) + 'min).csv')
    np.save('export/plot_body_positions/pfss_trace/trace_fls_(' + timestr_beg + '-' + timestr_end + '-' + str(
        timestep // timedelta(minutes=1)) + 'min).npy', fl_coords)

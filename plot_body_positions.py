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

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice

from horizons_spk import get_spk_from_horizons

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
    # start_et = spice.datetime2et(datetime.strptime(start_time_str, '%Y%m%dT%H%M%S'))
    # stop_et = spice.datetime2et(datetime.strptime(stop_time_str, '%Y%m%dT%H%M%S'))
    #
    # step = 100
    # times = [x * (stop_et - start_et) / step + start_et for x in range(step)]

    start_time = datetime.strptime(start_time_str, '%Y%m%dT%H%M%S')
    stop_time = datetime.strptime(stop_time_str, '%Y%m%dT%H%M%S')
    # start_et = spice.datetime2et(start_time)
    # stop_et = spice.datetime2et(stop_time)
    # start_time_str = start_time.strftime('%Y%m%dT%H%M%S')
    # stop_time_str = stop_time.strftime('%Y%m%dT%H%M%S')
    from datetime import timedelta
    timestep = timedelta(hours=1)
    steps = (stop_time - start_time) // timestep + 1
    dttimes = np.array([x * timestep + start_time for x in range(steps)])
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

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(psp_pos_carr[0], psp_pos_carr[1], psp_pos_carr[2], c='blue')
    ax.scatter(0, 0, 0, c='red')
    ax.scatter(psp_pos_carr[0, -1], psp_pos_carr[1, -1], psp_pos_carr[2, -1], c='blue')

    ax.plot([0, psp_pos_carr[0][0]], [0, psp_pos_carr[1][0]], [0, psp_pos_carr[2][0]], c='yellow')

    ax.set_xlim([-75, 75])
    ax.set_ylim([-75, 75])
    ax.set_zlim([-75, 75])
    plt.title('PSP orbit (' + start_time_str + '-' + stop_time_str + '), Carrington')
    ax.set_xlabel('x in Carrington (Rs)')
    ax.set_ylabel('y in Carrington (Rs)')
    ax.set_zlabel('z in Carrington (Rs)')
    plt.show()

    plt.figure()
    plt.scatter(np.rad2deg(psp_pos_carr_rtp[1]), np.rad2deg(psp_pos_carr_rtp[2]), c=psp_pos_carr_rtp[0])
    plt.colorbar(label='Radial Distance')
    plt.clim([20, 60])
    plt.scatter(np.rad2deg(psp_pos_carr_rtp[1, -1]), np.rad2deg(psp_pos_carr_rtp[2, -1]), c='black')
    plt.xlabel('Carrington Longitude')
    plt.ylabel('Carrington Latitude')
    plt.title('PSP Orbit in HG')
    plt.show()

    plt.figure()
    ax = plt.subplot(111, projection='polar')
    ax.plot(psp_pos_carr_rtp[1], psp_pos_carr_rtp[0], 'k-')
    ax.scatter(psp_pos_carr_rtp[1, 0:-1:24], psp_pos_carr_rtp[0, 0:-1:24], marker='x', c='black')
    ax.scatter(0, 0, c='red', s=100)
    ax.set_axisbelow('True')
    ax.set_thetagrids(np.arange(0.0, 360.0, 15.0))
    ax.set_thetamin(0.0)  # 设置极坐标图开始角度为0°
    ax.set_thetamax(180.0)  # 设置极坐标结束角度为180°
    # ax.set_rmax(2)
    # ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    ax.set_xlabel('Radius [Rs]')
    ax.set_ylabel('Carrington Longitude [deg]')
    ax.set_rlabel_position(0)  # Move radial labels away from plotted line
    ax.grid(True)
    plt.show()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(spice.et2datetime(times), np.rad2deg(psp_pos_carr_rtp[1]))
    # plt.title('Carrington Longitude of PSP')
    plt.xlabel('Observation Time')
    plt.ylabel('Carrington Longitude [deg]')
    plt.subplot(2, 1, 2)
    plt.plot(spice.et2datetime(times), psp_pos_carr_rtp[0])
    # plt.title('Solar Radius of PSP')
    plt.xlabel('Observation Time')
    plt.ylabel('Solar Radius [Rs]')

    plt.show()


def psp_backmap(epoch, r_source_surface_rs=2.5):
    from load_psp_data import load_spi_data
    from datetime import timedelta
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

    from scipy import interpolate

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

    psp_pos_hci, _ = spice.spkpos('SPP', times, 'SPP_HCI', 'NONE', 'SUN')
    psp_pos_hci = psp_pos_hci.T / Rs
    sun_pos_hci, _ = spice.spkpos('SUN', times, 'SPP_HCI', 'NONE', 'SUN')

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(psp_pos_hci[0], psp_pos_hci[1], psp_pos_hci[2], '-.', c='blue')
    ax.scatter(0, 0, 0, c='red')
    ax.scatter(psp_pos_hci[0, -1], psp_pos_hci[1, -1], psp_pos_hci[2, -1], c='blue')

    ax.plot([0, psp_pos_hci[0][0]], [0, psp_pos_hci[1][0]], [0, psp_pos_hci[2][0]], c='yellow')

    ax.set_xlim([-75, 75])
    ax.set_ylim([-75, 75])
    ax.set_zlim([-75, 75])
    plt.title('PSP orbit (' + start_time_str + '-' + stop_time_str + '), Carrington')
    ax.set_xlabel('x in Carrington (Rs)')
    ax.set_ylabel('y in Carrington (Rs)')
    ax.set_zlabel('z in Carrington (Rs)')
    plt.show()

    plt.figure()
    plt.plot(psp_pos_hci[0], psp_pos_hci[1], 'k-')
    plt.scatter(sun_pos_hci[0], sun_pos_hci[1], c='r')
    plt.xlim([-75, 75])
    plt.ylim([-75, 75])
    plt.xlabel('x(Rs) in HCI')
    plt.ylabel('y(Rs) in HCI')
    plt.show()


def get_rlonlat_psp_carr(dt_times, for_psi=True):
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


if __name__ == '__main__':
    xyz_carrington = np.array([1, 0, -1])
    r, lon, lat = xyz2rtp_in_Carrington(xyz_carrington)
    print('input XYZ', xyz_carrington)
    print('radius in Carr', r)
    print('Longitude in Carr', np.rad2deg(lon))
    print('Latitude in Carr', np.rad2deg(lat))
    # plot_psp_sun_carrington('20210422T000000','20210505T000000')
    plot_psp_sun_hci('20210422T000000', '20210505T000000')
    exit()
    datetime_beg = datetime(2021, 4, 28, 15, 30, 0)
    datetime_end = datetime(2021, 4, 30, 4, 30, 0)
    from datetime import timedelta

    timestep = timedelta(minutes=30)
    steps = (datetime_end - datetime_beg) // timestep + 1
    epoch = np.array([x * timestep + datetime_beg for x in range(steps)])
    r_footpoint_on_SourceSurface_rs, lon_footpoint_on_SourceSurface_deg, lat_footpoint_on_SourceSurface_deg, MFL_photosphere_lon_deg, MFL_photosphere_lat_deg, field_lines = psp_backmap(
        epoch, r_source_surface_rs=2.7)
    import pandas as pd

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
        # print(fl.coords）
        print(fl)

        try:
            fl_coords.append(fl.coords)
            fl_expansions.append(fl.expansion_factor)
        except:
            fl_coords.append(np.nan)
            fl_expansions.append(np.nan)
    df['Expansion_Factor'] = fl_expansions
    df.to_csv('E8trace0429_ss27.csv')
    np.save('save_field_lines_0429_ss27.npy', fl_coords)

    # import json
    # with open('field_lines.json','w') as save_file:
    #     for fl in field_lines:
    #         json.dump(fl.__dict__,save_file)

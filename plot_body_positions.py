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
import numpy as np
import matplotlib.pyplot as plt

import spiceypy as spice

from horizons_spk import get_spk_from_horizons
import furnsh_kernels



AU = 1.49e8  # km
Rs = 6.96e5  # km


def xyz2rtp_in_Carrington(xyz_carrington,for_psi = False):
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
    print(datetime.strptime(start_time_str, '%Y%m%d'))
    print(datetime.strptime(stop_time_str, '%Y%m%d'))
    start_et = spice.datetime2et(datetime.strptime(start_time_str, '%Y%m%d'))
    stop_et = spice.datetime2et(datetime.strptime(stop_time_str, '%Y%m%d'))

    step = 100
    times = [x * (stop_et - start_et) / step + start_et for x in range(step)]

    psp_pos_carr, _ = spice.spkpos('SPP', times, 'SPP_HG', 'NONE', 'SUN')
    psp_pos_carr = psp_pos_carr.T / Rs
    sun_pos_carr, _ = spice.spkpos('SUN', times, 'SPP_HG', 'NONE', 'SUN')

    psp_pos_carr_rtp = np.array([xyz2rtp_in_Carrington(psp_pos_carr[:, i]) for i in range(len(psp_pos_carr.T))])
    psp_pos_carr_rtp = psp_pos_carr_rtp.T

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(psp_pos_carr[0], psp_pos_carr[1], psp_pos_carr[2], c='blue')
    ax.scatter(0, 0, 0, c='red')
    ax.scatter(psp_pos_carr[0, -1], psp_pos_carr[1, -1], psp_pos_carr[2, -1], c='blue')
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
    plt.plot(np.rad2deg(psp_pos_carr_rtp[1]))
    plt.show()

if __name__ == '__main__':
    # xyz_carrington=np.array([1,0,-1])
    # r,lon,lat = xyz2rtp_in_Carrington(xyz_carrington)
    # print('input XYZ', xyz_carrington)
    # print('radius in Carr', r)
    # print('Longitude in Carr', np.rad2deg(lon))
    # print('Latitude in Carr', np.rad2deg(lat))
    plot_psp_sun_carrington('20210426','20210429')

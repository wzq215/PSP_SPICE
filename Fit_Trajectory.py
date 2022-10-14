from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spiceypy as spice
import sunpy.map
from scipy.optimize import leastsq

from plot_body_positions import xyz2rtp_in_Carrington

AU = 1.49e8  # km
Rs = 696300  # km
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
'''
045715-044215 (86,594) (94,590)
051215-045715 (95,594) (103,592)
052715-051215 (106,597) (114,595)
054215-052715 (111,602) (126,595)
055715-054215 (122,602) (136,597)
061215-055715 (131,603) (144,597)
062715-061215 (139,603) (147,600)
064215-062715 (148,607) (157,603)
065715-064115 (158,614) (171,607)
'''
'''et1 = spice.datetime2et(datetime.strptime('20210426T044215', '%Y%m%dT%H%M%S'))
et2 = spice.datetime2et(datetime.strptime('20210426T065715', '%Y%m%dT%H%M%S'))
n_markers = 10
delta_markers = 15
ets = np.linspace(et1, et2, n_markers)
locs = np.array(
    [(86, 594), (95, 594), (106, 597), (111, 602), (122, 602), (131, 603), (139, 603), (148, 607), (158, 614)])
locs = np.array(
    [(95, 590), (101, 592), (111, 593), (123, 596), (131, 597), (141, 599), (148, 602), (152, 606), (160, 608)])
locs = np.array(
    [(86, 588), (94, 590), (103, 592), (114, 595), (126, 595), (136, 597), (144, 597), (147, 600), (157, 603),
     (171, 607)])


et1 = spice.datetime2et(datetime.strptime('20210426T044215', '%Y%m%dT%H%M%S'))
et2 = spice.datetime2et(datetime.strptime('20210426T064215', '%Y%m%dT%H%M%S'))
n_markers = 9
delta_markers = 15
ets = np.linspace(et1, et2, n_markers)
locs = np.array([(104,578),(111,583),(119,585),(127,587),(138,588),(144,589),(153,592),(161,594),(170,597)])'''


def fit_trajectory(ets, n_markers, delta_markers, locs, p0, delta2, id_string=''):
    print(locs)
    n_pixelx = 960
    n_pixely = 1024
    boresight = np.array([n_pixelx, n_pixely]) // 2

    '''Step 1: pixel position to LOS in PSP frame'''
    los_locs = []
    for i in range(n_markers):
        loc = np.array(locs[i])
        et = ets[i]
        TM_arr = spice.sxform('SPP_WISPR_INNER', 'SPP_SPACECRAFT', et)
        los_in_wispr_inner = (loc - boresight) * 40 / n_pixelx  # deg
        print('----marker ', i, '----')
        print('LOS in WISPR_INNER (deg): ', los_in_wispr_inner)
        x_tmp = np.sin(np.deg2rad(los_in_wispr_inner[0])) * np.cos(np.deg2rad(los_in_wispr_inner[1]))
        y_tmp = -np.sin(np.deg2rad(los_in_wispr_inner[1]))
        z_tmp = np.cos(np.deg2rad(los_in_wispr_inner[0])) * np.cos(np.deg2rad(los_in_wispr_inner[1]))
        print('XYZ_tmp in WISPR_INNER: ', x_tmp, y_tmp, z_tmp)
        ray_spp = np.dot(TM_arr[0:3, 0:3], [x_tmp, y_tmp, z_tmp])
        lon_spp = np.rad2deg(np.arctan(ray_spp[0] / ray_spp[2]))
        lat_spp = np.rad2deg(-np.arcsin(ray_spp[1]))
        print('LOS in SPP Frame: ', [lon_spp, lat_spp])
        los_locs.append([lon_spp, lat_spp])

    los_locs = np.array(los_locs)
    plt.scatter(los_locs[:, 0], los_locs[:, 1], c=ets, cmap='jet')
    plt.xlabel(r'$\gamma$ (Longitude/deg)')
    plt.ylabel(r'$\beta$ (Latitude/deg)')
    plt.xlim(10, 40)
    plt.ylim(-15, 15)
    plt.title('Trace marked in the PSP SPACECRAFT Frame')
    plt.show()

    '''Step 2: fit trajectory'''
    t = np.linspace(0, n_markers - 1, n_markers) * delta_markers * 60  # sec
    gamma = np.deg2rad(los_locs[:, 0])  # rad
    beta = np.deg2rad(los_locs[:, 1])  # rad
    print(r'======== Markers in (\gamma, \beta) ========')
    print('gamma (deg): ', np.rad2deg(gamma))
    print('beta (deg): ', np.rad2deg(beta))
    df_markers = pd.DataFrame()
    df_markers['ets'] = ets
    df_markers['x_pixel'] = locs[:, 0]
    df_markers['y_pixel'] = locs[:, 1]
    df_markers['gamma_deg'] = los_locs[:, 0]
    df_markers['beta_deg'] = los_locs[:, 1]

    psp_pos, _ = spice.spkpos('SPP', ets, 'SPP_HCI', 'NONE', 'SUN')  # km
    earth_pos, _ = spice.spkpos('EARTH', ets, 'SPP_HCI', 'NONE', 'SUN')
    psp_pos = np.array(psp_pos)
    print('========= PSP positions in HCI =========')
    print('xyz (Rs): ', psp_pos.T / Rs)

    phi1 = []
    r1 = []
    psp_states, _ = spice.spkezr('SPP', ets, 'SPP_HCI', 'NONE', 'SUN')  # km
    G = 6.6726e-11  # m^3sec^-2kg^-1
    M = 1.988e30  # kg
    mu = G * M * 1e-9  # km^3sec^-2
    for i in range(n_markers):
        elt = spice.oscelt(psp_states[i], ets[i], mu)
        inc = elt[3]
        print('inc', 180 - np.rad2deg(inc))
        rlonlat = spice.reclat(psp_pos[i][0:3])
        r1.append(rlonlat[0])  # km
        phi1.append(np.arccos(np.cos(rlonlat[2]) * np.cos(rlonlat[1])))  # rad (-lnode?)
    r1 = np.array(r1)
    phi1 = -np.array(phi1)
    print('======== PSP positions in HCI (r, phi1) ========')
    print('r1 (Rs): ', r1 / Rs)
    print('phi1 (deg): ', np.rad2deg(phi1))

    def func(p, x, y):
        r20 = p[0]
        v = p[1]
        phi_2 = p[2]
        # a = p[3]
        a = 0
        delta_2 = p[4]
        # delta_2 = delta2
        r_2 = r20 + v * t + a * t ** 2 / 2
        # phi2 = phi2  # -lnodes
        return ((x - r_2 * np.cos(delta_2) * np.cos(phi_2 - y)) / (
                r_2 * np.cos(delta_2) * np.sin(phi_2 - y)) - 1 / np.tan(
            gamma))

    plsq = leastsq(func, p0, args=(r1, phi1), ftol=1e-10, xtol=1e-10)
    print('========Fitting parameters========')
    print('radius of starting point [Rs]: ', plsq[0][0] / Rs)
    print('velocity [km/s]: ', plsq[0][1])
    print('longitude [deg]: ', np.rad2deg(plsq[0][2]))
    print('acceleration [km/s^2]: ', plsq[0][3])
    print('latitude [deg]: ', np.rad2deg(plsq[0][4]))

    '''Visualization'''
    # UTC2ET

    etOne = spice.datetime2et(datetime.strptime('20210427T000000', '%Y%m%dT%H%M%S'))
    etTwo = spice.datetime2et(datetime.strptime('20210428T000000', '%Y%m%dT%H%M%S'))

    # Epochs
    step = 100
    times = [x * (etTwo - etOne) / step + etOne for x in range(step)]

    # Plot PSP&SUN orbit for specified time range
    psp_positions, psp_LightTimes = spice.spkpos('SPP', times, 'SPP_HCI', 'NONE', 'SUN')
    sun_positions, sun_LightTimes = spice.spkpos('SUN', times, 'SPP_HCI', 'NONE', 'SUN')
    psp_positions = psp_positions.T  # psp_positions is shaped (4000, 3), let's transpose to (3, 4000) for easier indexing
    psp_positions = psp_positions / AU
    sun_positions = sun_positions.T
    sun_positions = sun_positions / AU

    r2 = (plsq[0][0] + plsq[0][1] * t + plsq[0][3] * t ** 2 / 2)
    phi2 = plsq[0][2]
    structure_pos = []
    fit_lats = []
    fit_rlonlat_carrs = []
    fit_pos_carrs = []

    for i in range(n_markers):
        pos_tmp = [r2[i] * np.cos(delta2) * np.cos(phi2), r2[i] * np.cos(delta2) * np.sin(phi2), r2[i] * np.sin(delta2)]
        structure_pos.append(pos_tmp)
        fit_pos_inner, _ = spice.spkcpt(pos_tmp, 'SUN', 'SPP_HCI', ets[i], 'SPP_SPACECRAFT',
                                        'OBSERVER', 'NONE', 'SPP')
        fit_lat = spice.reclat(fit_pos_inner[[2, 0, 1]])
        fit_lats.append(fit_lat[1:3])
        fit_pos_carr, _ = spice.spkcpt(pos_tmp, 'SUN', 'SPP_HCI', ets[i], 'IAU_SUN', 'OBSERVER', 'NONE', 'SUN')
        fit_pos_carrs.append(fit_pos_carr)
        fit_rlonlat_carr = xyz2rtp_in_Carrington(fit_pos_carr[0:3])
        fit_rlonlat_carrs.append(fit_rlonlat_carr)

    fit_rlonlat_carrs = np.squeeze(np.array(fit_rlonlat_carrs))
    print(fit_rlonlat_carrs[0])
    print('Carr_First:', fit_rlonlat_carrs[0][0] / Rs, np.rad2deg(fit_rlonlat_carrs[0][1]),
          np.rad2deg(fit_rlonlat_carrs[0][2]))
    print('Carr_Last:', fit_rlonlat_carrs[-1][0] / Rs, np.rad2deg(fit_rlonlat_carrs[-1][1]),
          np.rad2deg(fit_rlonlat_carrs[-1][2]))
    structure_pos = np.array(structure_pos).T / AU
    fit_lats = np.array(fit_lats)

    df_markers['fit_gamma_deg'] = np.rad2deg(fit_lats.T[0])
    df_markers['fit_beta_deg'] = np.rad2deg(fit_lats.T[1])
    df_markers['fit_carr_lon_deg'] = np.rad2deg(fit_rlonlat_carrs.T[1])
    df_markers['fit_carr_lat_deg'] = np.rad2deg(fit_rlonlat_carrs.T[2])
    df_markers['fit_carr_r_Rs'] = fit_rlonlat_carrs.T[0] / Rs

    df_markers.to_csv(id_string + 'markers.csv')

    from datetime import timedelta
    t_prd = (spice.et2datetime(times) - spice.et2datetime(ets[0])) / timedelta(seconds=1)
    r2_prd = (plsq[0][0] + plsq[0][1] * t_prd + plsq[0][3] * t_prd ** 2 / 2)
    phi2_prd = plsq[0][2]
    structure_pos_prd = []
    fit_lats_prd = []
    fit_pos_carrs_prd = []
    fit_rlonlat_carrs_prd = []
    for i in range(len(times)):
        pos_tmp_prd = [r2_prd[i] * np.cos(delta2) * np.cos(phi2_prd), r2_prd[i] * np.cos(delta2) * np.sin(phi2_prd),
                       r2_prd[i] * np.sin(delta2)]
        structure_pos_prd.append(pos_tmp_prd)
        fit_pos_inner_prd, _ = spice.spkcpt(pos_tmp_prd, 'SUN', 'SPP_HCI', times[i], 'SPP_SPACECRAFT',
                                            'OBSERVER', 'NONE', 'SPP')
        fit_lat_prd = spice.reclat(fit_pos_inner_prd[[2, 0, 1]])
        fit_lats_prd.append(fit_lat_prd[1:3])
        fit_pos_carr_prd, _ = spice.spkcpt(pos_tmp_prd, 'SUN', 'SPP_HCI', times[i], 'IAU_SUN', 'OBSERVER', 'NONE',
                                           'SUN')
        fit_pos_carrs_prd.append(fit_pos_carr_prd)
        fit_rlonlat_carr_prd = xyz2rtp_in_Carrington(fit_pos_carr_prd[0:3])
        fit_rlonlat_carrs_prd.append(fit_rlonlat_carr_prd)
    fit_lats_prd = np.array(fit_lats_prd)
    fit_rlonlat_carrs_prd = np.squeeze(np.array(fit_rlonlat_carrs_prd))
    df = pd.DataFrame()
    df['ets'] = times
    df['gamma_deg'] = np.rad2deg(fit_lats_prd.T[0])
    df['beta_deg'] = np.rad2deg(fit_lats_prd.T[1])
    df['elongation_deg'] = np.rad2deg(np.arccos(np.cos(fit_lats_prd.T[0]) * np.cos(fit_lats_prd.T[1])))
    df['carr_lon_deg'] = np.rad2deg(fit_rlonlat_carrs_prd.T[1])
    df['carr_lat_deg'] = np.rad2deg(fit_rlonlat_carrs_prd.T[2])
    df['carr_r_Rs'] = fit_rlonlat_carrs_prd.T[0] / Rs

    df.to_csv(id_string + 'predict.csv')

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(psp_positions[0], psp_positions[1], psp_positions[2], c='gray')
    ax.scatter(psp_pos.T[0] / AU, psp_pos.T[1] / AU, psp_pos.T[2] / AU, c=ets, cmap='jet')
    ax.scatter(structure_pos[0], structure_pos[1], structure_pos[2], c=ets, cmap='jet')
    ax.scatter(0, 0, 0, c='r')
    ax.plot([0, 0.01], [0, 0], [0, 0], c='k')
    ax.plot([0, 0], [0, 0.03], [0, 0], c='k')
    ax.plot([0, 0], [0, 0], [0, 0.09], c='k')

    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    ax.set_xlim([0, 0.1])
    ax.set_ylim([0, 0.1])
    ax.set_zlim([-0.05, 0.05])

    plt.title('Fitting Result (in PSP_HCI frame)')
    plt.show()
    plt.subplot(121)
    plt.plot(np.linspace(1, n_markers, n_markers), np.rad2deg(gamma), 'k.-', label='data')
    plt.plot(np.linspace(1, n_markers, n_markers), np.rad2deg(fit_lats.T[0]), 'b-', label='fit')
    plt.legend()
    plt.ylabel(r'$\gamma$ [deg]')
    plt.xlabel(r'N_Markers $(\Delta t=30min)$')
    plt.subplot(122)
    plt.plot(np.linspace(1, n_markers, n_markers), np.rad2deg(beta), 'k.-', label='data')
    plt.plot(np.linspace(1, n_markers, n_markers), np.rad2deg(fit_lats.T[1]), 'b-', label='fit')
    plt.ylim([-1, 1])
    plt.legend()
    plt.ylabel(r'$\beta$ (deg)')
    plt.xlabel(r'N_Markers $(\Delta t=30min)$')
    plt.suptitle(id_string)
    plt.show()

    plt.figure()

    return plsq


def visualize_trajectory(times, plsq, delta2, et1, type, id_string):
    r2 = (plsq[0][0] + plsq[0][1] * (times - et1) + plsq[0][3] * (times - et1) ** 2 / 2)
    print(spice.et2datetime(et1))
    phi2 = plsq[0][2]
    structure_pos = []
    # fit_lats=[]
    structure_pos_carrs = []
    structure_rlonlat_carrs = []
    for i in range(len(times)):
        pos_tmp = [r2[i] * np.cos(delta2) * np.cos(phi2), r2[i] * np.cos(delta2) * np.sin(phi2), r2[i] * np.sin(delta2)]
        structure_pos.append(pos_tmp)
        structure_pos_carr, _ = spice.spkcpt(pos_tmp, 'SUN', 'SPP_HCI', times[i], 'IAU_SUN', 'OBSERVER', 'NONE', 'SUN')
        structure_pos_carrs.append(structure_pos_carr)
        structure_rlonlat_carr = xyz2rtp_in_Carrington(structure_pos_carr[0:3])
        structure_rlonlat_carrs.append(structure_rlonlat_carr)

    structure_pos = np.array(structure_pos).T / Rs
    structure_pos_carrs = np.array(structure_pos_carrs).T / Rs

    ## Plot PSP&SUN orbit for specified time range
    psp_positions, psp_LightTimes = spice.spkpos('SPP', times, 'SPP_HCI', 'NONE', 'SUN')
    earth_positions, earth_LightTimes = spice.spkpos('EARTH', times, 'SPP_HCI', 'NONE', 'SUN')
    psp_positions = psp_positions.T / Rs  # psp_positions is shaped (4000, 3), let's transpose to (3, 4000) for easier indexing
    earth_positions = earth_positions.T / Rs

    dts = spice.et2datetime(times)
    if type == '3D':
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(psp_positions[0], psp_positions[1], psp_positions[2], c=dts, cmap='jet')
        # ax.scatter(psp_states.T[0]/AU, psp_states.T[1]/AU, psp_states.T[2]/AU, c=ets, cmap='jet')
        ax.scatter(structure_pos[0], structure_pos[1], structure_pos[2], c=dts, cmap='jet')
        ax.plot(earth_positions[0], earth_positions[1], earth_positions[2], 'b-')
        ax.scatter(0, 0, 0, c='r')
        ax.plot([0, 3], [0, 0], [0, 0], c='k')
        ax.plot([0, 0], [0, 6], [0, 0], c='k')
        ax.plot([0, 0], [0, 0], [0, 9], c='k')
        # ax.plot([0,0.1*np.cos(np.mean(lnodes))],[0,0.1*np.sin(np.mean(lnodes))],[0,0],c='b')
        # ax.scatter(1, 0, 0, c='red')
        ax.set_xlabel('X (AU)')
        ax.set_ylabel('Y (AU)')
        ax.set_zlabel('Z (AU)')
        ax.set_xlim([-20, 20])
        ax.set_ylim([-20, 20])
        ax.set_zlim([-20, 20])
        # plt.title('PSP' + '(' + start_time + '_' + stop_time + ')')
        plt.title('Fitting Result (in PSP_HCI frame)')
        # plt.show()
    elif type == '2D_HCI':
        # plt.figure()
        plt.plot(psp_positions[0], psp_positions[1])
        plt.plot(structure_pos[0], structure_pos[1])
        plt.ylim([-50, 50])
        plt.xlim([-50, 50])
        # plt.show()
    elif type == '2D_Carr':
        psp_carrs, psp_LightTimes = spice.spkpos('SPP', times, 'IAU_SUN', 'NONE', 'SUN')
        psp_carrs = psp_carrs.T / Rs
        # plt.figure()
        plt.plot(psp_carrs[0], psp_carrs[1], '-', c='black')
        plt.plot(structure_pos_carrs[0], structure_pos_carrs[1], '-', label=id_string)
        plt.scatter(0, 0, c='red')
        plt.scatter(psp_carrs[0], psp_carrs[1], marker='x', c=times, cmap='jet')
        plt.scatter(structure_pos_carrs[0], structure_pos_carrs[1], marker='x', c=times, cmap='jet')
        plt.legend()
        plt.ylim([0, 40])
        plt.xlim([-10, 30])
        plt.gca().set_aspect(1)

        # plt.show()


# def calc_predicted_Jmap(times,plsq,delta2):


if __name__ == '__main__':
    '''et1 = spice.datetime2et(datetime.strptime('20210426T042715', '%Y%m%dT%H%M%S'))
    et2 = spice.datetime2et(datetime.strptime('20210426T155720', '%Y%m%dT%H%M%S'))
    n_markers = 24
    delta_markers = 30
    ets = np.linspace(et1, et2, n_markers)
    xx = [102,122,136,153,174, 190,211,230,247,265, 283,300,315,333,347, 365,382,399,413,425, 441,457,471,485]
    yy = [583,587,590,590,593,595,600,600,603,605,607,609,610,613,617,620,624,626,630,632,635,638,639,642]
    p0 = [15*Rs, 300, np.deg2rad(60), 0, 0]
    locs=np.array([(xx[i],yy[i]) for i in range(len(xx))])'''

    '''et1 = spice.datetime2et(datetime.strptime('20210426T041215', '%Y%m%dT%H%M%S'))
    et2 = spice.datetime2et(datetime.strptime('20210426T154220', '%Y%m%dT%H%M%S'))
    n_markers = 24
    delta_markers = 30
    ets = np.linspace(et1, et2, n_markers)
    xx=[91,113,130,147,161, 182,202,223,240,256, 272,290,304,326,339, 355,370,390,406,422, 439,455,470,481]
    yy=[582,584,589,590,595,596,599,602,603,606,608,612,614,617,618,620,622,625,627,629,631,633,637,640]
    p0 = [18*Rs, 400, np.deg2rad(85), 0, 0]
    locs=np.array([(xx[i],yy[i]) for i in range(len(xx))])'''

    et1 = spice.datetime2et(datetime.strptime('20210427T131220', '%Y%m%dT%H%M%S'))
    et2 = spice.datetime2et(datetime.strptime('20210427T154220', '%Y%m%dT%H%M%S'))
    n_markers = 6
    delta_markers = 30
    ets = np.linspace(et1, et2, n_markers)

    xx = [199, 244, 301, 362, 419, 478]
    yy = [583, 586, 594, 604, 615, 628]

    p0 = [10 * Rs, 300, np.deg2rad(60), 0, 0]
    locs = np.array([(xx[i], yy[i]) for i in range(len(xx))])
    et1_0427_track4 = et1
    delta2_0427_track4 = -0.02
    plsq_0427_track4 = fit_trajectory(ets, n_markers, delta_markers, locs, p0, delta2_0427_track4, 'Track #2')
    print(spice.et2datetime(ets))

    data, header = sunpy.io.fits.read('data/orbit08/20210427/psp_L3_wispr_20210427T141220_V1_1211.fits')[0]
    plt.imshow(data)
    plt.set_cmap('gray')
    plt.clim([1e-14, 5e-12])
    plt.scatter(locs.T[0], locs.T[1], c=ets, cmap='jet', s=10, marker='x')
    plt.title('Trace Markers')
    plt.gca().invert_yaxis()
    plt.show()
    # quit()
    et1 = spice.datetime2et(datetime.strptime('20210427T171220', '%Y%m%dT%H%M%S'))
    et2 = spice.datetime2et(datetime.strptime('20210427T201220', '%Y%m%dT%H%M%S'))
    et1_0427_track3 = et1
    n_markers = 7
    delta_markers = 30
    ets = np.linspace(et1, et2, n_markers)

    xx = [174, 209, 250, 295, 331, 372, 413]
    yy = [598, 603, 611, 616, 622, 630, 640]
    # xx=[140,187,246,302,370,459,573]
    # yy=[584,587,590,594,599,606,631]
    p0 = [8.5 * Rs, 300, np.deg2rad(60), 0, 0]
    locs = np.array([(xx[i], yy[i]) for i in range(len(xx))])
    delta2_0427_track3 = -0.085
    plsq_0427_track3 = fit_trajectory(ets, n_markers, delta_markers, locs, p0, delta2_0427_track3, 'Track #3')
    print(spice.et2datetime(ets))

    data, header = sunpy.io.fits.read('data/orbit08/20210427/psp_L3_wispr_20210427T171220_V1_1211.fits')[0]
    plt.imshow(data)
    plt.set_cmap('gray')
    plt.clim([1e-14, 5e-12])
    plt.scatter(locs.T[0], locs.T[1], c=ets, cmap='jet', s=10, marker='x')
    plt.title('Trace Markers')
    plt.gca().invert_yaxis()
    plt.show()

    #
    #
    #
    #
    #
    # quit()

    et1 = spice.datetime2et(datetime.strptime('20210427T034220', '%Y%m%dT%H%M%S'))
    et2 = spice.datetime2et(datetime.strptime('20210427T091220', '%Y%m%dT%H%M%S'))
    n_markers = 12
    delta_markers = 30
    ets = np.linspace(et1, et2, n_markers)
    xx = [55, 102, 141, 181, 223, 260, 297, 345, 392, 440, 495, 556]
    yy = [585, 594, 600, 610, 620, 628, 635, 643, 647, 657, 668, 681]
    p0 = [8 * Rs, 300, np.deg2rad(45), 0, 0]
    locs = np.array([(xx[i], yy[i]) for i in range(len(xx))])
    et1_0427_track1 = et1
    delta2_0427_track1 = -0.09
    plsq_0427_track1 = fit_trajectory(ets, n_markers, delta_markers, locs, p0, delta2_0427_track1,
                                      id_string='Track #1b')

    data, header = sunpy.io.fits.read('data/orbit08/20210427/psp_L3_wispr_20210427T042720_V1_1211.fits')[0]
    plt.imshow(data)
    plt.set_cmap('gray')
    plt.clim([1e-14, 5e-12])
    plt.scatter(locs.T[0], locs.T[1], c=ets, cmap='jet', s=10, marker='x')
    plt.title('Trace Markers')
    plt.gca().invert_yaxis()
    plt.show()

    xx = [87, 126, 170, 214, 251, 288, 339, 390, 441, 488, 533, 593]
    yy = [570, 579, 584, 588, 594, 598, 607, 612, 620, 629, 642, 655]
    p0 = [9 * Rs, 300, np.deg2rad(45), 0, 0]
    locs = np.array([(xx[i], yy[i]) for i in range(len(xx))])
    et1_0427_track2 = et1
    delta2_0427_track2 = -0.035
    plsq_0427_track2 = fit_trajectory(ets, n_markers, delta_markers, locs, p0, delta2_0427_track2,
                                      id_string='Track #1a')

    data, header = sunpy.io.fits.read('data/orbit08/20210427/psp_L3_wispr_20210427T042720_V1_1211.fits')[0]
    plt.imshow(data)
    plt.set_cmap('gray')
    plt.clim([1e-14, 5e-12])
    plt.scatter(locs.T[0], locs.T[1], c=ets, cmap='jet', s=10, marker='x')
    plt.title('Trace Markers')
    plt.gca().invert_yaxis()
    plt.show()

    # et1 = spice.datetime2et(datetime.strptime('20210428T025716', '%Y%m%dT%H%M%S'))
    # et2 = spice.datetime2et(datetime.strptime('20210428T052716', '%Y%m%dT%H%M%S'))
    # n_markers = 5
    # delta_markers = 30
    # ets = np.linspace(et1, et2, n_markers)
    # print(ets)
    # # xx=[249,309,358,414,461,500]
    # # yy=[615,623,634,645,653,660]
    # xx=[253,309,360,411,466]
    # yy=[617,624,632,638,645]
    # p0 = [10*Rs, 250, np.deg2rad(50), 0, 0]
    # locs=np.array([(xx[i],yy[i]) for i in range(len(xx))])
    # delta2_0428_track1 = 0
    # plsq_0428_track1=fit_trajectory(ets,n_markers,delta_markers,locs,p0,delta2_0428_track1)

    # quit()
    etOne = spice.datetime2et(datetime.strptime('20210427T000000', '%Y%m%dT%H%M%S'))
    etTwo = spice.datetime2et(datetime.strptime('20210430T000000', '%Y%m%dT%H%M%S'))
    print(etTwo)
    step = 72
    times = np.array([x * (etTwo - etOne) / step + etOne for x in range(step)])
    print(spice.et2datetime(times))
    # quit()
    markers = ets
    plt.figure()

    visualize_trajectory(times, plsq_0427_track2, delta2_0427_track2, et1_0427_track2, type='2D_Carr',
                         id_string='0427 Track #1a')
    visualize_trajectory(times, plsq_0427_track1, delta2_0427_track1, et1_0427_track1, type='2D_Carr',
                         id_string='0427 Track #1b')
    visualize_trajectory(times, plsq_0427_track4, delta2_0427_track4, et1_0427_track4, type='2D_Carr',
                         id_string='0427 Track #2')
    visualize_trajectory(times, plsq_0427_track3, delta2_0427_track3, et1_0427_track3, type='2D_Carr',
                         id_string='0427 Track #3')

    # visualize_trajectory(times,plsq_0428_track1,delta2_0428_track1,type='2D_Carr',id_string='0428 trace 3')
    # plt.colorbar()
    dt_ticks = [datetime(2021, 4, 27, 0), datetime(2021, 4, 27, 12), datetime(2021, 4, 28, 0),
                datetime(2021, 4, 28, 12), datetime(2021, 4, 29, 0), datetime(2021, 4, 29, 12),
                datetime(2021, 4, 30, 0)]
    cbar = plt.colorbar(ticks=spice.datetime2et(dt_ticks))
    cticks = cbar.get_ticks()
    cbar.ax.set_yticks(cticks)
    cbar.ax.set_yticklabels(spice.et2datetime(cticks))
    plt.xlabel('x_Carrington [Rs]')
    plt.ylabel('y_Carrington [Rs]')
    plt.title('Fitting Results in Carrington Coordinates')
    plt.show()

import os
from functools import partial

import itertools
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import spiceypy as spice

from ps_read_hdf_3d import ps_read_hdf_3d
from wispr_insitu import plot_frames
import furnsh_kernels

from plot_body_positions import xyz2rtp_in_Carrington
# Read PSI Data
data_rho_corona = ps_read_hdf_3d(2239, 'corona', 'rho002', periodicDim=3)
r_rho_corona = np.array(data_rho_corona['scales1'])  # in Rs, distance from sun
t_rho_corona = np.array(data_rho_corona['scales2'])  # in rad, latitude
p_rho_corona = np.array(data_rho_corona['scales3'])  # in rad, Carrington longitude
rho_corona = np.array(data_rho_corona['datas'])  # 1CU = 10^8 cm^-3
rho_corona = rho_corona * 1e8 * 1e6  # m^-3

data_rho_helio = ps_read_hdf_3d(2239, 'helio', 'rho002', periodicDim=3)
r_rho_helio = np.array(data_rho_helio['scales1'])  # in Rs, distance from sun
t_rho_helio = np.array(data_rho_helio['scales2'])  # in rad, latitude
p_rho_helio = np.array(data_rho_helio['scales3'])  # in rad, Carrington longitude
rho_helio = np.array(data_rho_helio['datas'])  # 1CU = 10^8 cm^-3
rho_helio = rho_helio * 1e8 * 1e6  # m^-3

I_0 = 1361 * 215 ** 2. / np.pi
AU = 1.49e11  # distance from sun to earth m
Rs = 6.96e5  # solar radii km
sigma_e = 7.95e-30

def NGRF_filter(I_matrix,D_matrix,window_width):
    print('min',np.nanmin(D_matrix))
    print('max',np.nanmax(D_matrix))
    n = np.nanmin(D_matrix)
    while n <= np.nanmax(D_matrix):
        bin = (abs(D_matrix-n-window_width/2) <= (window_width/2) )
        I_binned = I_matrix[bin].reshape(1,-1)
        mean_I = np.nanmean(I_binned)
        sigma_I = np.nanstd(I_binned)
        # print(I_matrix.shape)
        # print(I_binned.shape)
        # print(mean_I)
        # print(sigma_I)
        I_matrix[bin] = ((I_matrix[bin]-mean_I)/sigma_I)#*0+n
        n += window_width
        print(n)
    return I_matrix

def maxnorm_filter(I_matrix,D_matrix,window_width):
    print('min',np.nanmin(D_matrix))
    print('max',np.nanmax(D_matrix))
    n = np.nanmin(D_matrix)
    while n <= np.nanmax(D_matrix):
        bin = (abs(D_matrix-n-window_width/2) <= (window_width/2) )
        I_binned = I_matrix[bin].reshape(1,-1)
        # mean_I = np.nanmean(I_binned)
        # sigma_I = np.nanstd(I_binned)
        max_I = np.nanmax(I_binned)
        # print(I_matrix.shape)
        # print(I_binned.shape)
        # print(mean_I)
        # print(sigma_I)
        I_matrix[bin] = ((I_matrix[bin])/max_I)#*0+n
        n += window_width
        print(n)
    return I_matrix

def gaussian_filter(I_matrix,D_matrix):
    maxD = np.nanmax(D_matrix)
    minD = np.nanmin(D_matrix)
    print(maxD)
    print(minD)
    x = np.linspace(minD,maxD,100)
    y = 1.5-(np.exp(-(x-minD)**2/(maxD-minD)**2))
    plt.plot(x,y)
    plt.show()
    I_matrix = I_matrix * (1.5-(np.exp(-(D_matrix-minD)**2/((maxD-minD)**2))))
    return I_matrix

def get_WL_images(time_str, vignetting='quad', type='full',resolution=0.5):
    time = spice.datetime2et(datetime.strptime(time_str, '%Y%m%dT%H%M%S'))
    psp_pos, _ = spice.spkpos('SPP', time, 'IAU_SUN', 'NONE', 'SUN')  # km
    psp_pos = psp_pos.T / Rs
    R = np.linalg.norm(psp_pos, 2)  # Rs
    TM_arr = spice.sxform('SPP_WISPR_INNER', 'SPP_SPACECRAFT', time)[0:3, 0:3]

    def get_K_corona(loc_angle, type='full',return_TEC=False):
        print(loc_angle)

        lon_angle = loc_angle[0]  # LOS经度方向上的方位角
        lat_angle = loc_angle[1]  # LOS纬度方向上的方位角

        ray_wispr = [np.cos(np.deg2rad(lat_angle)) * np.sin(np.deg2rad(lon_angle)),
                     -np.sin(np.deg2rad(lat_angle)),
                     np.cos(np.deg2rad(lat_angle)) * np.cos(np.deg2rad(lon_angle)), ]

        ray_spp = np.dot(TM_arr, ray_wispr)
        lon_spp = np.arctan(ray_spp[0] / ray_spp[2])
        lat_spp = -np.arcsin(ray_spp[1])

        elongation_spp = np.arccos(np.cos(lon_spp) * np.cos(lat_spp))
        d_sun = R*np.tan(elongation_spp)

        deltaz = 0.1
        z_TS = R * np.cos(elongation_spp)
        if type == 'full':
            z_max = 200
            z_min = 0.1
        elif type == 'foreground':
            z_max = z_TS
            z_min = 0.1
        elif type == 'background':
            z_max = 200
            z_min = z_TS

        z = np.arange(z_min, z_max, deltaz)

        rho_los = z * 0
        scatter_angle = z * 0
        poses = np.zeros((len(z), 3))
        rs = z * 0

        for i in range(len(z)):
            tmpz = z[i] * Rs
            tmppos_wisprinner = [tmpz * np.cos(np.deg2rad(lat_angle)) * np.sin(np.deg2rad(lon_angle)),
                                 - tmpz * np.sin(np.deg2rad(lat_angle)),
                                 tmpz * np.cos(np.deg2rad(lat_angle) * np.cos(np.deg2rad(lon_angle)))]
            tmppos_carrington, _ = spice.spkcpt(tmppos_wisprinner, 'SPP', 'SPP_WISPR_INNER', time,
                                                'IAU_SUN', 'CENTER', 'NONE', 'SUN')
            tmppos_carrington /= Rs
            r_tmppos_carrington,p_tmppos_carrington,t_tmppos_carrington = xyz2rtp_in_Carrington(tmppos_carrington,for_psi=True)
            # r_tmppos_carrington = np.linalg.norm(tmppos_carrington[0:3], 2)
            rs[i] = r_tmppos_carrington
            scatter_angle[i] = np.pi - np.arccos(
                (r_tmppos_carrington ** 2 + z[i] ** 2 - R ** 2) / (2 * r_tmppos_carrington * z[i]))
            #
            # if tmppos_carrington[1] > 0:
            #     p_tmppos_carrington = np.arccos(tmppos_carrington[0] / r_tmppos_carrington)
            # elif tmppos_carrington[1] <= 0:
            #     p_tmppos_carrington = 2 * np.pi - np.arccos(tmppos_carrington[0] / r_tmppos_carrington)
            #
            # t_tmppos_carrington = np.arccos(tmppos_carrington[2] / r_tmppos_carrington)

            if r_tmppos_carrington < 30:
                r_rho = r_rho_corona
                p_rho = p_rho_corona
                t_rho = t_rho_corona
                rho = rho_corona
            elif r_tmppos_carrington < 200:
                r_rho = r_rho_helio
                p_rho = p_rho_helio
                t_rho = t_rho_helio
                rho = rho_helio
            else:
                r_rho = r_rho_helio
                p_rho = p_rho_helio
                t_rho = t_rho_helio
                rho = rho_helio * 0

            r_ind = np.argmin(abs(r_tmppos_carrington - r_rho))
            p_ind = np.argmin(abs(p_tmppos_carrington - p_rho))
            t_ind = np.argmin(abs(t_tmppos_carrington - t_rho))

            rho_tmppos_carrington = rho[p_ind, t_ind, r_ind]
            rho_los[i] = rho_tmppos_carrington
            poses[i, :] = tmppos_carrington[0:3]

        # if return_TEC:
        #     TEC = np.nansum(rho_los)
        #     return TEC

        Omega = np.arccos(np.sqrt(rs ** 2 - 1 ** 2) / rs)
        sin_Omega = np.sin(Omega)
        cos_Omega = np.cos(Omega)

        A = cos_Omega * sin_Omega ** 2
        B = -1 / 8 * (1 - 3 * sin_Omega ** 2 - cos_Omega ** 2 / sin_Omega *
                      (1 + 3 * sin_Omega ** 2) * np.log((1 + sin_Omega / cos_Omega)))
        C = 4 / 3 - cos_Omega - cos_Omega ** 3 / 3
        D = 1 / 8 * (5 + sin_Omega ** 2 - cos_Omega ** 2 / sin_Omega *
                     (5 - sin_Omega ** 2) * np.log((1 + sin_Omega) / cos_Omega))
        u = 0.56

        I_tot_coeff = I_0 * np.pi * sigma_e / 2 / z ** 2
        first_item = 2 * I_tot_coeff * ((1 - u) * C + u * D)
        second_item = -I_tot_coeff * np.sin(scatter_angle) ** 2 * ((1 - u) * A + u * B)
        I_tot = first_item + second_item
        I_at_per_z = rho_los * z ** 2 * I_tot
        I = np.nansum(I_at_per_z) * Rs * deltaz * 1e3

        if return_TEC:
            TEC = np.nansum(rho_los)
            return I,TEC
        return I

    def get_d_sun(loc_angle):
        lon_angle = loc_angle[0]  # LOS经度方向上的方位角
        lat_angle = loc_angle[1]  # LOS纬度方向上的方位角

        ray_wispr = [np.cos(np.deg2rad(lat_angle)) * np.sin(np.deg2rad(lon_angle)),
                     -np.sin(np.deg2rad(lat_angle)),
                     np.cos(np.deg2rad(lat_angle)) * np.cos(np.deg2rad(lon_angle)), ]

        ray_spp = np.dot(TM_arr, ray_wispr)
        lon_spp = np.arctan(ray_spp[0] / ray_spp[2])
        lat_spp = -np.arcsin(ray_spp[1])

        elongation_spp = np.arccos(np.cos(lon_spp) * np.cos(lat_spp))
        d_sun = R*np.tan(elongation_spp)
        return d_sun

    filename = 'I_Kcorona_'+'resolution='+str(resolution)+'_'+time_str+'_'+type
    path = 'data/K_corona_from_psi/'

    thetas = np.arange(-20, 20, resolution)
    phis = np.arange(-20, 20, resolution)

    if not os.path.exists(path + filename+'.npz'):
        p = Pool(4)
        # I_matrix,TEC_matrix = p.map(partial(get_K_corona, type=type,return_TEC=True), itertools.product(thetas, phis))
        I_matrix = []
        TEC_matrix = []
        for I, TEC in p.map(partial(get_K_corona, type=type,return_TEC=True), itertools.product(thetas, phis)):
            I_matrix.append(I)
            TEC_matrix.append(TEC)
        D_matrix = p.map(get_d_sun,itertools.product(thetas, phis))
        np.savez(path+filename+'.npz', I_matrix,D_matrix,TEC_matrix)

    r = np.load(path+filename+'.npz')
    print(r)
    I_matrix = r['arr_0'].reshape(len(thetas),len(phis))
    D_matrix = r['arr_1'].reshape(len(thetas),len(phis))
    TEC_matrix = r['arr_2'].reshape(len(thetas),len(phis))

    plt.scatter(TEC_matrix.reshape(1,-1),I_matrix.reshape(1,-1))
    plt.xlabel('TEC')
    plt.ylabel('I')
    plt.show()
    if vignetting == 'quad':
        p=0.01
        I_matrix = I_matrix * (p+(D_matrix)**2/((np.nanmax(D_matrix))**2)*(1-p))
    if vignetting == 'NGRF':
        I_matrix = NGRF_filter(I_matrix,D_matrix,0.2)
    if vignetting == 'maxnorm':
        I_matrix = maxnorm_filter(I_matrix,D_matrix,1)
    if vignetting == 'Gauss':
        I_matrix = gaussian_filter(I_matrix,D_matrix)


    # print(I_matrix)
    # plt.pcolormesh(thetas, phis, I_matrix.T * 4.5e-16 / 6.61e-12 / 1361, cmap='gray')
    plt.pcolormesh(thetas, phis, I_matrix.T , cmap='gray')
    plt.colorbar()
    plt.xlabel('Longitude(deg)')
    plt.ylabel('Latitude(deg)')
    plt.title(filename+'_'+vignetting)
    plt.savefig(path+filename+'_'+vignetting+'.png')
    # plt.clim([5e-12,5e-10])
    # plt.clim([-13.5,-12.5])
    # plt.clim([20,30])
    # plt.clim([1,4])
    plt.show()
    return filename




if __name__ == '__main__':
    filename = get_WL_images('20210117T150000',type='full',resolution=0.5,vignetting='Gauss')

    # path = 'data/K_corona_from_psi'
    # vignetting = 'quad'
    # resolutions =
    # f
    # I_matrix,D_matrix = np.load(path+filename+'.npy')
    #
    # if vignetting == 'quad':
    #     I_matrix = I_matrix * (D_matrix-np.minnan(D_matrix))**2/((np.maxnan(D_matrix)-np.minnan(D_matrix))**2)
    #
    # plt.pcolormesh(thetas, phis, np.log10(I_matrix.T * 4.5e-16 / 6.61e-12 / 1361), cmap='gray')
    # plt.colorbar()
    # plt.xlabel('Longitude(deg)')
    # plt.ylabel('Latitude(deg)')
    # plt.title(filename)
    # plt.savefig(path+filename+'.png')
    # # plt.clim([5e-12,5e-10])
    # plt.clim([-14,12])
    # plt.show()
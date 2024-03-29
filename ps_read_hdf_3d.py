"""
Name:
    ps_read_hdf_3d.
Purpose:
    Reads a 3-dimensional HDF file.
Inputs:
    crid: ID of the Carrington Rotation
    region: 'corona' or 'helio'
    param: e.g. 'br002' (https://www.predsci.com/mhdweb/data_access.php)
Keywords:
    PERIODIC_DIM:
        (Heritage from idl routine)Dimension number to be made periodic (0 - 2PI). Valid values are:
            1 - The first dimension.
            2 - The second dimension.
            3 - The third dimension.
Outputs:
    Dict containing:
        scales1:
            Scale values of the first dimension.
        scales2:
            Scale values of the second dimension.
        scales3:
            Scale values of the third dimension.
        datas:
            Data values.

Name:
    ps_load_hdf.
Purpose:
    Load HDF files from PSI site
Input:
    crid:
    Range: 'corona' or 'helio'
    param: e.g. 'br002'
Output:
    HDF files downloaded
"""
import os
import re

import numpy as np
import requests
from pyhdf.HDF import *
from pyhdf.SD import *
from tqdm import tqdm


def ps_read_hdf_3d(crid, region, param, periodicDim=[], auto=False):
    path = 'Data/cr' + str(crid) + '/' + region + '/'
    filename = param + '.hdf'
    # print(path + filename)
    # print(os.path.exists(path + filename))
    if not os.path.exists(path + filename):
        pp = ps_load_hdf(crid, region, param, auto=auto)
        if bool(pp) is False:
            print('Nothing Found in this CR!')
            return 0

    sdId = SD(path + filename)
    sdkeys = sdId.datasets().keys()

    scales3 = sdId.select('fakeDim0').get()
    scales2 = sdId.select('fakeDim1').get()
    scales1 = sdId.select('fakeDim2').get()
    datas = sdId.select('Data-Set-2').get()

    sdId.end()

    scales3 = np.array(scales3)
    scales2 = np.array(scales2)
    scales1 = np.array(scales1)
    datas = np.array(datas)

    # Periodic is assumed to have a range from 0 to 2PI
    if type(periodicDim) == int:
        nx = len(scales3)
        ny = len(scales2)
        nz = len(scales1)
        tmpDatas = datas

        if periodicDim == 1:
            scales1 = np.append(scales1, scales1[0] + 2.0 * np.pi)
            datas = np.zeros((nx, ny, nz + 1))
            # print(datas.shape)
            datas[:, :, 0:-1] = tmpDatas
            datas[:, :, -1] = tmpDatas[:, :, 0]
        elif periodicDim == 2:
            scales2 = np.append(scales2, scales2[0] + 2.0 * np.pi)
            datas = np.zeros((nx, ny + 1, nz))
            datas[:, 0:- 1, :] = tmpDatas
            datas[:, -1, :] = tmpDatas[:, 0, :]
        elif periodicDim == 3:
            scales3 = np.append(scales3, scales3[0] + 2.0 * np.pi)
            datas = np.zeros((nx + 1, ny, nz))
            datas[0:-1, :, :] = tmpDatas
            datas[-1, :, :] = tmpDatas[0, :, :]

    dict = {'scales1': scales1, 'scales2': scales2, 'scales3': scales3, 'datas': datas}
    # print('Shape of Scales:',scales1.shape,scales2.shape,scales3.shape)
    # print('Shape of Datas',datas.shape)

    return dict


def ps_load_hdf(crid, region, param, auto=False):
    # https://www.predsci.com/data/runs/cr2246-high/hmi_mast_mas_std_0201/corona/
    url = 'https://www.predsci.com/data/runs/'
    url += 'cr' + str(crid) + '-'
    path = 'Data/cr' + str(crid)
    try:
        os.mkdir(path)
    except:
        pass

    if str(requests.get(url + 'high/', verify=False)) != '<Response [404]>':
        url += 'high/'
    else:
        print('No High Data, try Medium')
        url += 'medium/'
        if str(requests.get(url, verify=False)) == '<Response [404]>':
            print('Nothing Found!')
            return 0

    r_links = requests.get(url, verify=False)
    res_url = r"(?<=href=\").+?(?=\")|(?<=href=\').+?(?=\')"
    links = re.findall(res_url, str(r_links.content), re.I | re.S | re.M)
    l = [s for s in links if 'mas' in s]
    for i in range(len(l)):
        print(i, l[i])
    if auto:
        s = 0
        print('choose 0')
        with open('load_PSI_DATA.txt', 'a') as f:
            f.write(str(crid) + ' ')
            f.write(str(l[0]) + '\n')
    else:
        s = int(input('choose one file: (e.g. 0)'))

    url += l[s] + region + '/'
    path += '/' + region + '/'
    filename = param + '.hdf'
    r = requests.get(url + filename, verify=False, stream=True)
    total = int(r.headers.get('content-length', 0))
    if str(r) != '<Response [404]>':
        try:
            os.mkdir(path)
        except:
            pass
        f = open(path + filename, 'wb')
        bar = tqdm(desc=path + filename, total=total, unit='iB', unit_scale=True, unit_divisor=1024, )
        for data in r.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
        f.close
    return 1


if __name__ == '__main__':
    dict = ps_read_hdf_3d(2246, 'corona', 'bp002')
    # print(dict['scales1'])

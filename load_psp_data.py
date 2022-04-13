import os
import re

import requests
import matplotlib.pyplot as plt
from spacepy import pycdf
from datetime import datetime
from datetime import timedelta
from itertools import compress

from tqdm import tqdm
import numpy as np

import plotly.offline as py
import plotly.graph_objects as go

os.environ["CDF_LIB"] = "/usr/local/cdf/lib"
psp_data_path = '/Users/ephe/Documents/PSP_Data_Analysis/Encounter08/'


def load_RTN_1min_data(start_time_str, stop_time_str):
    start_time = datetime.strptime(start_time_str, '%Y%m%d').toordinal()
    stop_time = datetime.strptime(stop_time_str, '%Y%m%d').toordinal()
    filelist = [psp_data_path+'psp_fld_l2_mag_RTN_1min_' + datetime.fromordinal(x).strftime('%Y%m%d') + '_v02.cdf'
                for x in range(start_time, stop_time)]
    # print(filelist)
    data = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    print(data)
    return data

def load_RTN_data(start_time_str,stop_time_str):
    '''psp_fld_l2_mag_rtn_2021042800_v02.cdf'''
    cycles = [0,6,12,18]
    start_time = datetime.strptime(start_time_str, '%Y%m%d%H')
    stop_time = datetime.strptime(stop_time_str,'%Y%m%d%H')
    # start_hour = start_time.hour
    # start_index = divmod(start_hour,6)[0]
    start_file_time = datetime(start_time.year,start_time.month,start_time.day,cycles[divmod(start_time.hour,6)[0]])
    stop_file_time = datetime(stop_time.year,stop_time.month,stop_time.day,cycles[divmod(stop_time.hour,6)[0]])
    if divmod(stop_time.hour,6)[1] == 0:
        stop_file_time -= timedelta(hours=6)
    filelist =[]
    tmp_time = start_file_time
    while tmp_time <= stop_file_time:
        filelist.append(psp_data_path+'psp_fld_l2_mag_rtn_' + tmp_time.strftime('%Y%m%d%H') + '_v02.cdf')
        tmp_time +=timedelta(hours=6)
    print(filelist)


    # filelist = [[psp_data_path+'psp_fld_l2_mag_rtn_' + datetime.fromordinal(x).strftime('%Y%m%d') + '00_v02.cdf',
    #              psp_data_path+'psp_fld_l2_mag_rtn_' + datetime.fromordinal(x).strftime('%Y%m%d') + '06_v02.cdf',
    #              psp_data_path+'psp_fld_l2_mag_rtn_' + datetime.fromordinal(x).strftime('%Y%m%d') + '12_v02.cdf',
    #              psp_data_path+'psp_fld_l2_mag_rtn_' + datetime.fromordinal(x).strftime('%Y%m%d') + '18_v02.cdf',]
    #             for x in range(start_time, stop_time)]
    # # filelist = np.array(filelist).reshape(-1,1)
    # filelist = sum(filelist,[])
    data = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    # print(data)
    return data



def load_spc_data(start_time_str, stop_time_str):
    # psp/sweap/spc/psp_swp_spc_l3i_20210115_v02.cdf
    start_time = datetime.strptime(start_time_str, '%Y%m%d').toordinal()
    stop_time = datetime.strptime(stop_time_str, '%Y%m%d').toordinal()
    filelist = [psp_data_path+'psp_swp_spc_l3i_' + datetime.fromordinal(x).strftime('%Y%m%d') + '_v02.cdf'
                for x in range(start_time, stop_time)]
    # print(filelist)
    data = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    print(data)
    return data


def load_spe_data(start_time_str, stop_time_str):
    # psp/sweap/spe/psp_swp_spa_sf0_L3_pad_20210115_v03.cdf
    start_time = datetime.strptime(start_time_str, '%Y%m%d').toordinal()
    stop_time = datetime.strptime(stop_time_str, '%Y%m%d').toordinal()
    filelist = [psp_data_path+'psp_swp_spa_sf0_L3_pad_' + datetime.fromordinal(x).strftime('%Y%m%d') + '_v03.cdf'
                for x in range(start_time, stop_time)]
    # print(filelist)
    data = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    print(data)
    return data


def load_spi_data(start_time_str, stop_time_str):
    # psp/sweap/spi/psp_swp_spi_sf00_L3_mom_INST_20210115_v03.cdf
    start_time = datetime.strptime(start_time_str, '%Y%m%d').toordinal()
    stop_time = datetime.strptime(stop_time_str, '%Y%m%d').toordinal()
    filelist = [psp_data_path+'psp_swp_spi_sf00_L3_mom_' + datetime.fromordinal(x).strftime('%Y%m%d') + '_v04.cdf'
                for x in range(start_time, stop_time)]
    # print(filelist)
    data = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    print(data)
    return data


if __name__ == '__main__':
    plot_1min = False

    beg_time = datetime(2021,4,29,3)
    end_time = datetime(2021,4,29,6)
    beg_time_str = beg_time.strftime('%Y%m%dT%H%M%S')
    end_time_str = end_time.strftime('%Y%m%dT%H%M%S')

    if plot_1min:
        mag_RTN = load_RTN_1min_data(beg_time.strftime('%Y%m%d'), end_time.strftime('%Y%m%d'))

        epochmag = mag_RTN['epoch_mag_RTN_1min']
        timebinmag = (epochmag > beg_time) & (epochmag < end_time)
        epochmag = epochmag[timebinmag]

        Br = mag_RTN['psp_fld_l2_mag_RTN_1min'][timebinmag, 0]
        Bt = mag_RTN['psp_fld_l2_mag_RTN_1min'][timebinmag, 1]
        Bn = mag_RTN['psp_fld_l2_mag_RTN_1min'][timebinmag, 2]

        filename = 'figures/overviews/Overview_1min('+beg_time_str+'-'+end_time_str+').html'
    else:

        mag_RTN = load_RTN_data(beg_time.strftime('%Y%m%d%H'), end_time.strftime('%Y%m%d%H'))

        epochmag = mag_RTN['epoch_mag_RTN']
        timebinmag = (epochmag > beg_time) & (epochmag < end_time)
        epochmag = epochmag[timebinmag]

        Br = mag_RTN['psp_fld_l2_mag_RTN'][timebinmag, 0]
        Bt = mag_RTN['psp_fld_l2_mag_RTN'][timebinmag, 1]
        Bn = mag_RTN['psp_fld_l2_mag_RTN'][timebinmag, 2]

        filename = 'figures/overviews/Overview('+beg_time_str+'-'+end_time_str+').html'

    pmom_SPI = load_spi_data(beg_time.strftime('%Y%m%d'), end_time.strftime('%Y%m%d'))
    epochpmom = pmom_SPI['Epoch']
    timebinpmom = (epochpmom > beg_time) & (epochpmom < end_time)
    epochpmom = epochpmom[timebinpmom]
    np = pmom_SPI['DENS'][timebinpmom]
    vp_r = pmom_SPI['VEL_RTN_SUN'][timebinpmom,0]
    vp_t = pmom_SPI['VEL_RTN_SUN'][timebinpmom,1]
    vp_n = pmom_SPI['VEL_RTN_SUN'][timebinpmom,2]
    Tp = pmom_SPI['TEMP'][timebinpmom]

    from plotly.subplots import make_subplots

    fig = make_subplots(rows=4, cols=1,
                        specs=[[{"secondary_y": True}], [{"secondary_y": True}],
                               [{"secondary_y": True}], [{"secondary_y": True}]],
                        subplot_titles=("B_R & V_R", "B_T & V_T", "B_N & V_N"), shared_xaxes=True)
    fig.add_trace(go.Scatter(x=epochmag, y=Br, name='Br', mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochmag, y=Bt, name='Bt', mode='lines'), row=2, col=1)
    fig.add_trace(go.Scatter(x=epochmag, y=Bn, name='Bn', mode='lines'), row=3, col=1)
    fig.add_trace(go.Scatter(x=epochpmom, y=np, name='np',mode='lines'),row=4,col=1)

    fig.add_trace(go.Scatter(x=epochpmom, y=vp_r, name='Vr', mode='lines'), row=1, col=1,secondary_y=True)
    fig.add_trace(go.Scatter(x=epochpmom, y=vp_t, name='Vt', mode='lines'), row=2, col=1,secondary_y=True)
    fig.add_trace(go.Scatter(x=epochpmom, y=vp_n, name='Vn', mode='lines'), row=3, col=1,secondary_y=True)
    fig.add_trace(go.Scatter(x=epochpmom, y=Tp, name='np',mode='lines'),row=4,col=1,secondary_y=True)

    fig.update_xaxes(tickformat="%m/%d %H:%M\n%Y", title_text="Epoch")
    fig.update_yaxes(title_text="Br [nT]", row=1, col=1)
    fig.update_yaxes(title_text="Bt [nT]", row=2, col=1)
    fig.update_yaxes(title_text="Bn [nT]", row=3, col=1)
    fig.update_layout(title=dict(text="Magnetic Field",
                                 y=0.9, x=0.5,
                                 xanchor='center',
                                 yanchor='top'),
                      template='simple_white', )
    py.plot(fig,filename='figures/overviews/Overview('+beg_time_str+'-'+end_time_str+').html')

import os

import matplotlib.pyplot as plt
from spacepy import pycdf
from datetime import datetime

os.environ["CDF_LIB"] = "/usr/local/cdf/lib"


def load_RTN_1min_data(start_time_str, stop_time_str):
    start_time = datetime.strptime(start_time_str, '%Y%m%d').toordinal()
    stop_time = datetime.strptime(stop_time_str, '%Y%m%d').toordinal()
    filelist = ['psp/field/mag/1min/psp_fld_l2_mag_RTN_1min_' + datetime.fromordinal(x).strftime('%Y%m%d') + '_v02.cdf'
                for x in range(start_time, stop_time)]
    # print(filelist)
    data = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    # print(data)
    return data


def load_spc_data(start_time_str, stop_time_str):
    # psp/sweap/spc/psp_swp_spc_l3i_20210115_v02.cdf
    start_time = datetime.strptime(start_time_str, '%Y%m%d').toordinal()
    stop_time = datetime.strptime(stop_time_str, '%Y%m%d').toordinal()
    filelist = ['psp/sweap/spc/psp_swp_spc_l3i_' + datetime.fromordinal(x).strftime('%Y%m%d') + '_v02.cdf'
                for x in range(start_time, stop_time)]
    # print(filelist)
    data = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    print(data)
    return data


def load_spe_data(start_time_str, stop_time_str):
    # psp/sweap/spe/psp_swp_spa_sf0_L3_pad_20210115_v03.cdf
    start_time = datetime.strptime(start_time_str, '%Y%m%d').toordinal()
    stop_time = datetime.strptime(stop_time_str, '%Y%m%d').toordinal()
    filelist = ['psp/sweap/spe/psp_swp_spa_sf0_L3_pad_' + datetime.fromordinal(x).strftime('%Y%m%d') + '_v03.cdf'
                for x in range(start_time, stop_time)]
    # print(filelist)
    data = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    print(data)
    return data


def load_spi_data(start_time_str, stop_time_str):
    # psp/sweap/spi/psp_swp_spi_sf00_L3_mom_INST_20210115_v03.cdf
    start_time = datetime.strptime(start_time_str, '%Y%m%d').toordinal()
    stop_time = datetime.strptime(stop_time_str, '%Y%m%d').toordinal()
    filelist = ['psp/sweap/spi/psp_swp_spi_sf00_L3_mom_INST_' + datetime.fromordinal(x).strftime('%Y%m%d') + '_v03.cdf'
                for x in range(start_time, stop_time)]
    # print(filelist)
    data = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    print(data)
    return data


if __name__ == '__main__':
    RTN = load_RTN_1min_data('20210115', '20210121')
    # plt.scatter(RTN['epoch_mag_RTN_1min'],RTN['psp_fld_l2_mag_RTN_1min'][:,0],c=RTN['epoch_mag_RTN_1min'],label=RTN['label_RTN'][0])
    # plt.show()
    import plotly.offline as py
    import plotly.graph_objects as go
    import plotly.express as px

    epoch = RTN['epoch_mag_RTN_1min']
    Br = RTN['psp_fld_l2_mag_RTN_1min'][:, 0]
    Bt = RTN['psp_fld_l2_mag_RTN_1min'][:, 1]
    # fig=px.line(x=RTN['epoch_mag_RTN_1min'],y=RTN['psp_fld_l2_mag_RTN_1min'][:,0])
    # py.plot(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epoch, y=Br, name='Br', mode='lines'))

    fig.update_xaxes(tickformat="%m/%d %H:%M\n%Y")
    fig.update_layout(title=dict(text="Radial Magnetic Field",
                                 y=0.9, x=0.5,
                                 xanchor='center',
                                 yanchor='top'),
                      xaxis_title="Epoch",
                      yaxis_title="Br [nT]",
                      calendar="julian",
                      autosize=False,
                      width=1000,
                      height=300,
                      template='simple_white')
    py.plot(fig)
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("B_R", "B_T"), shared_xaxes=True)
    fig.add_trace(go.Scatter(x=epoch, y=Br, name='Br', mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=epoch, y=Bt, name='Bt', mode='markers',
                             marker=dict(size=2,
                                         color=Bt, colorscale='viridis',
                                         showscale=True,
                                         colorbar=dict(title="Bt",
                                                       lenmode='fraction', len=0.5,
                                                       thicknessmode='pixels', thickness=10,
                                                       yanchor='top', y=0.5),
                                         cmax=200,
                                         cmin=-200)),
                  row=2, col=1)

    fig.update_xaxes(tickformat="%m/%d %H:%M\n%Y", title_text="Epoch")
    fig.update_yaxes(title_text="Br [nT]", row=1, col=1)
    fig.update_yaxes(title_text="Bt [nT]", row=2, col=1)
    fig.update_layout(title=dict(text="Magnetic Field",
                                 y=0.9, x=0.5,
                                 xanchor='center',
                                 yanchor='top'),
                      template='simple_white', )
    py.plot(fig)

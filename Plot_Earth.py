"""
@Filename: Plot_Earth.py
@Aim: plot an animation including HCS(SC&IH), three Observers and correspondent field lines and the Earth in a Carrington Coordinate
@Author: Ziqi Wu
@Date of Last Change: 2022-02-18
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as py


# from ps_read_hdf_3d import get_path


def topo_spheres(r, pos, opacity=1, planet='earth'):
    """
    :param r: A float. Radius of the sphere.
    :param pos: A vector [x,y,z]. Position of the center of the sphere
    :param opacity: A float between 0 and 1. Opacity.
    :param planet: A string. 'earth'/'venus'/'mercury'.
    :return: A trace (go.Surface()) for plotly.
    """

    theta = np.linspace(0, 2 * np.pi, 360)
    phi = np.linspace(0, np.pi, 180)
    tt, pp = np.meshgrid(theta, phi)

    x0 = pos[0] + r * np.cos(tt) * np.sin(pp)
    y0 = pos[1] + r * np.sin(tt) * np.sin(pp)
    z0 = pos[2] + r * np.cos(pp)

    # path = os.path.join(get_path(), 'Data')
    df = pd.read_csv('topo.CSV')
    lon = df.columns.values
    lon = np.array(float(s) for s in lon[1:])
    lat = np.array(df.iloc[:, 0].values)
    topo = np.array(df.iloc[:, 1:].values)

    from PIL import Image, ImageOps
    sunim = Image.Open('data/euvi_aia304_2012_carrington_print.jpeg')
    sunimgray = ImageOps.grayscale(sunim)
    sundata = np.array(sunimgray)
    sunlon = np.linspace(0, 360, sundata.shape[1])
    sunlat = np.linspace(-90, 90, sundata.shape[0])

    # Set up trace
    if planet == 'earth':
        trace = go.Surface(x=x0, y=y0, z=z0, surfacecolor=topo, colorscale='YlGnBu', opacity=opacity)
    elif planet == 'venus':
        trace = go.Surface(x=x0, y=y0, z=z0, surfacecolor=z0, colorscale='solar', opacity=opacity)
    elif planet == 'mercury':
        trace = go.Surface(x=x0, y=y0, z=z0, surfacecolor=z0, colorscale='ice', opacity=opacity)
    elif planet == 'sun':

    trace.update(showscale=False)

    return trace


if __name__ == '__main__':
    plot = go.Figure()
    plot.add_trace(topo_spheres(1, [0, 0, 0], opacity=0.7))
    py.plot(plot)

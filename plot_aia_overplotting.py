"""
Overplotting field lines on AIA maps
====================================

This example shows how to take a PFSS solution, trace some field lines, and
overplot the traced field lines on an AIA 193 map.
"""
import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pfsspy
import pfsspy.tracing as tracing
import sunpy.io.fits
import sunpy.map
from astropy.coordinates import SkyCoord

# from pfsspy.sample_data import get_gong_map

###############################################################################
# Load a GONG magnetic field map
gong_fname = '/Users/ephe/PFSS_data/mrzqs210429t0004c2243_105.fits.gz'
gong_map = sunpy.map.Map(gong_fname)
print(gong_map)

###############################################################################
# Load the corresponding AIA 193 map
print(os.path)
if not os.path.exists('aia___map.fits'):
    import urllib.request

    urllib.request.urlretrieve(
        'http://jsoc2.stanford.edu/data/aia/synoptic/2021/04/29/H0000/AIA20210429_0004_0304.fits',
        'aia___map.fits')

aia = sunpy.map.Map('aia___map.fits')
dtime = aia.date
print(dtime)
ax = plt.subplot(1, 1, 1, projection=aia)
aia.plot(ax)

# import Glymur
# hv = helioviewer.HelioviewerClient()

data, header = sunpy.io.fits.read('/Users/ephe/STEREO_Data/EUVI/195/20210428_235530_n4eua.fts')[0]
stamap = sunpy.map.Map(data, header)
# aia = sta
###############################################################################
# The PFSS solution is calculated on a regular 3D grid in (phi, s, rho), where
# rho = ln(r), and r is the standard spherical radial coordinate. We need to
# define the number of grid points in rho, and the source surface radius.
nrho = 25
rss = 2.5

###############################################################################
# From the boundary condition, number of radial grid points, and source
# surface, we now construct an `Input` object that stores this information
pfss_in = pfsspy.Input(gong_map, nrho, rss)

###############################################################################
# Using the `Input` object, plot the input photospheric magnetic field
m = pfss_in.map
fig = plt.figure()
ax = plt.subplot(projection=m)
m.plot()
plt.colorbar()
ax.set_title('Input field')

###############################################################################
# We can also plot the AIA map to give an idea of the global picture. There
# is a nice active region in the top right of the AIA plot, that can also
# be seen in the top left of the photospheric field plot above.
ax = plt.subplot(1, 1, 1, projection=aia)
aia.plot(ax)

###############################################################################
# Now we construct a 5 x 5 grid of footpoitns to trace some magnetic field
# lines from. These coordinates are defined in the helioprojective frame of the
# AIA image


hp_lon = np.linspace(-1000, -750, 5) * u.arcsec
hp_lat = np.linspace(-250, -500, 5) * u.arcsec
# Make a 2D grid from these 1D points
lon, lat = np.meshgrid(hp_lon, hp_lat)
seeds = SkyCoord(lon.ravel(), lat.ravel(),
                 frame=aia.coordinate_frame)
# print(seeds)
# exit()
# print(lon.ravel())
import pandas as pd

df_trace = pd.read_csv('E8trace0429.csv')
lon = np.deg2rad(df_trace['MFL_photosphere_lon_deg']) * u.rad
lat = np.deg2rad(df_trace['MFL_photosphere_lat_deg']) * u.rad
radius = (df_trace['r_footpoint_on_SourceSurface_rs']) * 696300 * 1000 * u.m

lon_ss = np.deg2rad(df_trace['lon_footpoint_on_SourceSurface_deg']) * u.rad
lat_ss = np.deg2rad(df_trace['lat_footpoint_on_SourceSurface_deg']) * u.rad
radius_ss = (df_trace['r_footpoint_on_SourceSurface_rs']) * 696300 * 1000 * u.m

# print(lon.ravel())
import astropy.constants as const

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
r = 1.01 * const.R_sun
pfss_out = pfsspy.pfss(pfss_in)
seeds_psp = SkyCoord(lon_ss.ravel(), lat_ss.ravel(), radius_ss,
                     frame=pfss_out.coordinate_frame)
tracer = tracing.FortranTracer()
field_lines = tracer.trace(seeds_psp, pfss_out)

for field_line in field_lines:
    color = {0: 'black', -1: 'tab:blue', 1: 'tab:red'}.get(field_line.polarity)
    coords = field_line.coords
    coords.representation_type = 'cartesian'
    ax.plot(coords.x / const.R_sun,
            coords.y / const.R_sun,
            coords.z / const.R_sun,
            color=color, linewidth=1)

ax.set_title('PFSS solution')
plt.show()
exit()
# print(seeds)
# # exit()
# seeds = seeds.transform_to(frame=aia.coordinate_frame)
# seeds_psp.representation_type = 'unitspherical'
# seeds_psp = SkyCoord(seeds_psp.Tx,seeds_psp.Ty,frame=gong_map.coordinate_frame)
# print(seeds)
fig = plt.figure()
ax = plt.subplot(projection=aia)
aia.plot(axes=ax)
ax.plot_coord(seeds, color='white', marker='o', linewidth=0)

###############################################################################
# Plot the magnetogram and the seed footpoints The footpoints are centered
# around the active region metnioned above.
m = pfss_in.map
fig = plt.figure()
ax = plt.subplot(projection=m)
m.plot()
plt.colorbar()
plt.clim(-50, 50)

ax.plot_coord(seeds, color='black', marker='o', linewidth=0, markersize=2)

# Set the axes limits. These limits have to be in pixel values
# ax.set_xlim(0, 180)
# ax.set_ylim(45, 135)
ax.set_title('Field line footpoints')
ax.set_ylim(bottom=0)

#######################################################################
# Compute the PFSS solution from the GONG magnetic field input
# pfss_out = pfsspy.pfss(pfss_in)

###############################################################################
# Trace field lines from the footpoints defined above.

# print(tracer.validate_seeds(seeds_psp))
flines = tracer.trace(seeds_psp, pfss_out)
# print(flines)
###############################################################################
# Plot the input GONG magnetic field map, along with the traced mangetic field
# lines.
m = pfss_in.map
fig = plt.figure()
ax = plt.subplot(projection=m)
m.plot()
plt.colorbar()

for fline in flines:
    # print(fline.coords)
    ax.plot_coord(fline.coords, color='black', linewidth=1)

# Set the axes limits. These limits have to be in pixel values
# ax.set_xlim(0, 180)
# ax.set_ylim(45, 135)
ax.set_title('Photospheric field and traced field lines')
# plt.show()
###############################################################################
# Plot the AIA map, along with the traced magnetic field lines. Inside the
# loop the field lines are converted to the AIA observer coordinate frame,
# and then plotted on top of the map.
fig = plt.figure()
ax = plt.subplot(1, 1, 1, projection=aia)
aia.plot(ax)
for fline in flines:
    ax.plot_coord(fline.coords, alpha=0.8, linewidth=1, color='white')

# ax.set_xlim(500, 900)
# ax.set_ylim(400, 800)
plt.show()

# sphinx_gallery_thumbnail_number = 5

# 辅助函数（画图用的）
def set_axes_lims(ax):
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 180)

# # 1. 安装pfsspy:
# pip3 install pfsspy，需要sunpy, astropy
# # 2. 下载GONG磁图（这里下载的是2020年1月28日12:14的）并解压
# https://gong2.nso.edu/oQR/zqs/202001/mrzqs200128/
# # 3. python代码, 需要的库如下：
import astropy.constants as const
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sunpy.map
import pfsspy
from pfsspy import coords
from pfsspy import tracing

# 4.  读取GONG磁图
gong_path = '/Users/ephe/PFSS_Data/'
gong_fname = 'mrzqs220225t0014c2254_078.fits'  # 文件名改成自己的
gong_map = sunpy.map.Map(gong_path + gong_fname)
# Remove the mean，这里为了使curl B = 0
gong_map = sunpy.map.Map(gong_map.data - np.mean(gong_map.data), gong_map.meta)
# 5. 设置网格数量和source surface高度并计算
nrho = 30
rss = 2.5  # 单位：太阳半径
input = pfsspy.Input(gong_map, nrho, rss)
output = pfsspy.pfss(input)
# 6. 画输入的GONG磁图
m = input.map
fig = plt.figure()
ax = plt.subplot(projection=m)
m.plot()
plt.colorbar()
ax.set_title('Input field')
set_axes_lims(ax)

# 7. 画输出的source surface磁场分布
ss_br = output.source_surface_br
# Create the figure and axes
fig = plt.figure()
ax = plt.subplot(projection=ss_br)

# Plot the source surface map
ss_br.plot()
# Plot the polarity inversion line
ax.plot_coord(output.source_surface_pils[0])
print(output.source_surface_pils)
# Plot formatting
plt.colorbar()
ax.set_title('Source surface magnetic field')
set_axes_lims(ax)

# 8. 画赤道平面的磁场分布
fig, ax = plt.subplots()
ax.set_aspect('equal')

# Take 100 start points spaced equally in theta
r = 1.01 * const.R_sun
lon = np.pi / 2 * u.rad
lat = np.linspace(-np.pi/2, np.pi / 2, 100) * u.rad
seeds = SkyCoord(lon, lat, r, frame=output.coordinate_frame)

tracer = pfsspy.tracing.PythonTracer()
field_lines = tracer.trace(seeds, output)

for field_line in field_lines:
    coords = field_line.coords
    coords.representation_type = 'cartesian'
    color = {0: 'black', -1: 'tab:blue', 1: 'tab:red'}.get(field_line.polarity)
    ax.plot(coords.y / const.R_sun,
            coords.z / const.R_sun, color=color)

# Add inner and outer boundary circles
ax.add_patch(mpatch.Circle((0, 0), 1, color='k', fill=False))
ax.add_patch(mpatch.Circle((0, 0), input.grid.rss, color='k', linestyle='--',
                           fill=False))
ax.set_title('PFSS solution')
plt.show()


# 9. Parker磁场外推
# 利用公式

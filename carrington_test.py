import numpy as np
import spiceypy as spice
import furnsh_kernels
from datetime import datetime
from plot_body_positions import xyz2rtp_in_Carrington

start_utc_str = '20201226T00000'
mid_utc_str = '20210115T000000'
end_utc_str = '20210122T000000'
Rs = 696300 #km
start_et = spice.datetime2et(datetime.strptime(start_utc_str, '%Y%m%dT%H%M%S'))
xyz_start,_ = spice.spkpos('EARTH', start_et, 'IAU_SUN', 'NONE', 'SUN')
rtp_start = xyz2rtp_in_Carrington(xyz_start/Rs)
subpnt_start = spice.subpnt('INTERCEPT/ELLIPSOID','SUN',start_et,'IAU_SUN','None','EARTH')
start_sph = np.rad2deg(spice.reclat(subpnt_start[0])[1:])
start_r = (np.linalg.norm(subpnt_start[0]-subpnt_start[2],2))/Rs
rr = (np.linalg.norm(subpnt_start[2],2))/Rs+1

print('-------START-------')
print('xyz in IAU_SUN (Rs)', xyz_start/Rs)
print('rtp converted (Rs)', np.rad2deg(rtp_start[1:]))
print('Subpoint',subpnt_start)
print('Radial Distance (Rs)',start_r)
print('Sphere degrees (deg)',start_sph)
print('distance from solar center (Rs)',rr)

mid_et = spice.datetime2et(datetime.strptime(mid_utc_str,'%Y%m%dT%H%M%S'))
xyz_mid,_ = spice.spkpos('EARTH', mid_et, 'IAU_SUN', 'NONE', 'SUN')
rtp_mid = xyz2rtp_in_Carrington(xyz_mid/Rs)
subpnt_mid = spice.subpnt('INTERCEPT/ELLIPSOID','SUN',mid_et,'IAU_SUN','None','EARTH')
mid_sph = np.rad2deg(spice.reclat(subpnt_mid[0])[1:])
mid_r = (Rs+np.linalg.norm(subpnt_mid[2],2))/Rs

print('--------middle--------')
print('xyz in IAU_SUN (Rs)', xyz_mid/Rs)
print('rtp converted (Rs)', np.rad2deg(rtp_mid[1:]))
print('Subpoint',subpnt_mid)
print('Radial Distance (Rs)',mid_r)
print('Sphere degrees (deg)',mid_sph)

end_et = spice.datetime2et(datetime.strptime(end_utc_str,'%Y%m%dT%H%M%S'))
xyz_end,_ = spice.spkpos('EARTH', end_et, 'IAU_SUN', 'NONE', 'SUN')
rtp_end = xyz2rtp_in_Carrington(xyz_end/Rs)
subpnt_end = spice.subpnt('INTERCEPT/ELLIPSOID','SUN',end_et,'IAU_SUN','None','EARTH')
end_sph = np.rad2deg(spice.reclat(subpnt_end[0])[1:])
end_r = (Rs+np.linalg.norm(subpnt_end[2],2))/Rs

print('--------end--------')
print('xyz in IAU_SUN (Rs)', xyz_end/Rs)
print('rtp converted (Rs)', np.rad2deg(rtp_end[1:]))
print('Subpoint',subpnt_end)
print('Radial Distance (Rs)',end_r)
print('Sphere degrees (deg)',end_sph)
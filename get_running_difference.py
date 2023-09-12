from datetime import datetime, timedelta

import sunpy.io.fits
from matplotlib import pyplot as plt

rdstep = 30
dtnow = datetime(2021, 4, 27, 19, 42, 20)
# TRACK
xx = []
yy = []

# TRACK


# TRACK05
# 084220
# xx=[306,356]
# yy=[600,615]

# TRACK04
# 131220/134220/141220/144220/151220/154220
# xx=[199,244,301,362,419,478]
# yy=[583,586,594,604,615,628]


# 171220;174220;181220;184220;191220;194220;201220
# #TRACKO3 0427
# xx=[140,187,246,302,370,459,573]
# yy=[584,587,590,594,599,606,631]
# xx=[174,209,250,295,331,372,413]
# yy=[598,603,611,616,622,630,640]


# xx=[413,447,478,504,532]
# yy=[606,611,621,627,630]
# 0426 trace1
# xx = [102,122,136,153,174,190,211,230,247,265,283,300,315,333,347,365,382,399,413,425,441,457,471,485]
# yy = [583,587,590,590,593,595,600,600,603,605,607,609,610,613,617,620,624,626,630,632,635,638,639,642]
# 0426 trace2
# xx=[91,113,130,147,161,182,202,223,240,256,272,290,304,326,339,355,370,390,406,422,439,455,470,481]
# yy=[582,584,589,590,595,596,599,602,603,606,608,612,614,617,618,620,622,625,627,629,631,633,637,640]
# 0427 trace1
# xx = [55, 102, 141, 181, 223, 260, 297, 345, 392, 440, 495, 556]
# yy = [585, 594, 600, 610, 620, 628, 635, 643, 647, 657, 668, 681]
# # 0427 trace2
# xx2 = [87, 126, 170, 214, 251, 288, 339, 390, 441, 488, 533, 593]
# yy2 = [570, 579, 584, 588, 594, 598, 607, 612, 620, 629, 642, 655]
# 0428
# xx=[249,309,358,414,461,500]
# yy=[615,623,634,645,653,660]
# xx=[253,309,360,411,466]
# yy=[617,624,632,638,645]


dtpre = dtnow - timedelta(minutes=rdstep, seconds=0)
dtnow_str = dtnow.strftime('%Y%m%dT%H%M%S')
dtpre_str = dtpre.strftime('%Y%m%dT%H%M%S')
print('Current Time: ', dtnow_str)
print('Previous Time: ', dtpre_str)
data, header = sunpy.io.fits.read('data/orbit08/20210427/psp_L3_wispr_' + dtpre_str + '_V1_1211.fits')[0]
data2, header2 = sunpy.io.fits.read('data/orbit08/20210427/psp_L3_wispr_' + dtnow_str + '_V1_1211.fits')[0]
data = data2 - data
print(header)
print(header['BUNIT'])

plt.figure()
plt.imshow(data2)
plt.xlabel('longitude (pixel)')
plt.ylabel('latitude (pixel)')
plt.gca().invert_yaxis()
plt.title(dtnow_str)
plt.colorbar()
plt.set_cmap('gist_gray')
# plt.clim([-4e-14, 4e-14])
plt.clim([1e-14, 5e-13])
# plt.scatter(xx, yy, c='red', marker='x', s=20, label='Track #1b')
# plt.scatter(xx2, yy2, c='blue', marker='x', s=20, label='Track #1a')
# plt.ylim([300,700])
# plt.ylim([450,700])
# plt.xlim([50,300])
plt.gca().set_aspect(1)
plt.legend()
plt.show()

plt.subplot(121)
plt.imshow(data)

plt.xlabel('longitude (pixel)')
plt.ylabel('latitude (pixel)')
plt.gca().invert_yaxis()
plt.title(dtnow_str + ' Running Difference (' + str(rdstep) + 'min)')
plt.colorbar()
plt.set_cmap('gist_gray')
plt.clim([-2e-14, 2e-14])
# plt.clim([1e-14,1e-12])
plt.scatter(xx, yy, c='red', marker='x', s=20, label='Track #1b')
# plt.scatter(xx2, yy2, c='blue', marker='x', s=20, label='Track #1a')

# plt.ylim([500,700])
# plt.ylim([450,700])
# plt.xlim([50,300])
plt.gca().set_aspect(1)
plt.legend()

plt.subplot(122)
plt.imshow(data2)
plt.xlabel('longitude (pixel)')
plt.ylabel('latitude (pixel)')
plt.gca().invert_yaxis()
plt.title(dtnow_str)
plt.colorbar()
plt.set_cmap('gist_gray')
# plt.clim([-4e-14, 4e-14])
plt.clim([1e-14, 2e-12])
plt.scatter(xx, yy, c='red', marker='x', s=20, label='Track #1b')
# plt.scatter(xx2, yy2, c='blue', marker='x', s=20, label='Track #1a')
# plt.ylim([300,700])
# plt.ylim([450,700])
# plt.xlim([50,300])
plt.gca().set_aspect(1)
plt.legend()
plt.show()

'''
(70,586)
(80,587)
(86,588)
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

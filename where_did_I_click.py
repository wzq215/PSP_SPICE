import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# et1 = spice.datetime2et(datetime.strptime('20210116T010019','%Y%m%dT%H%M%S'))
# et2 = spice.datetime2et(datetime.strptime('20210116T040019','%Y%m%dT%H%M%S'))
# ets = np.linspace(et1,et2,7)
# locs=np.array([(34,228),(49,227),(61,224),(81,222),(98,221),(107,219),(127,216)])


'''
1 - 20210116T010019 - (34,228) - (43,228)
2 - 20210116T013019 - (49,227) - (65,225)
3 - 20210116T020019 - (61,224) - (83,222)
4 - 20210116T023019 - (79,223) - (107,220)
5 - 20210116T030019 - (96,221) - (122,219)
6 - 20210116T033019 - (107,219) - (148,215)
7 - 20210116T040019 - (127,216) - (167,211)
'''

'''
190019 - 67,224
193019 - 
200019 - 78,223 - 109,217
203019 - 88,220 - 135,215
210019 - 100,219 - 150,213
213019 - 111,218 - 163,211
220019 - 
223019 - 127,216 - 179,212
230028 - 141,214 - 
'''

timestr='20210116T040019'
timestr2 = '20210116T033019'
im = Image.open('psp/wispr/images/inner/psp_L3_wispr_'+timestr+'_V1_1221.png')
im2 = Image.open('psp/wispr/images/inner/psp_L3_wispr_'+timestr2+'_V1_1221.png')

a=179
b=212

# plt.imshow(im)
# plt.show()
print(np.array(im,dtype='double'))
print(np.array(im2))
imdelta = np.array(im,dtype='double')-np.array(im2,dtype='double')
print(imdelta)

# imdelta = Image.fromarray(imdelta)
plt.pcolormesh(imdelta,cmap='seismic')
plt.colorbar()
plt.clim(-100,100)
# plt.scatter(a,b,c='r',marker='x')
plt.gca().invert_yaxis()
plt.gca().set_aspect(1)
plt.savefig('figures/running_difference/'+timestr+'diff.png')
plt.show()

# plt.subplot(131)
# plt.imshow(im2.crop((a-30,b-30,a+31,b+31)))
# plt.scatter(30,30,c='r',marker='x')
# # plt.show()
# plt.subplot(132)
# # imcrop = im.crop((a-30,b-30,a+31,b+31))
# plt.imshow(im.crop((a-30,b-30,a+31,b+31)))
# plt.scatter(30,30,c='r',marker='x')
# plt.subplot(133)
# # plt.imshow(imdelta,cmap='gray')
# # plt.colorbar()
# # plt.scatter(30,30,c='r',marker='x')
# plt.pcolormesh(imdelta[b-30:b+31,a-30:a+31],cmap='seismic')
# # plt.colorbar()
# plt.clim(-50,50)
# plt.gca().invert_yaxis()
# plt.gca().set_aspect(1)
# plt.scatter(30,30,c='k',marker='x')
# # plt.scatter()
# # plt.show()
# plt.suptitle('PSP_INNER ('+timestr+') ('+str(a)+','+str(b)+')')
# plt.savefig('figures/trajectory_fit/'+timestr+'.png')
# plt.show()
'''20210116T010019 - (34,228)'''
# pos = plt.ginput(2)
# print(pos)
# a=[]
# b=[]
# for i in range(len(pos)):
#     a.append(pos[i][0])
#     b.append(pos[i][1])
# plt.show()



import imageio
import matplotlib.pyplot as plt
import numpy as np
from spacepy.pybats import IdlFile

# 如果是2d cut的文件：
data_path = '/Users/ephe/THL8/output_0928_002/'
file_name = '2d__mhd_2_n'
n_iters = np.linspace(100, 10000, 100)
para_str = 'Bz'
frames = []
for n_iter in n_iters:
    file_str = file_name + str(int(n_iter)).zfill(8)
    data_2d = IdlFile(data_path + 'SC/' + file_str + '.out')
    fig, ax = plt.subplots()
    # tpc=ax.tripcolor(data_2d['x'], data_2d['y'], np.log10(np.array(data_2d[para_str])))
    tpc = ax.tripcolor(data_2d['x'], data_2d['y'], np.array(data_2d[para_str]))
    # plt.title('log Rho [log(g/cm^3)]'+'(n'+str(int(n_iter)).zfill(8)+')')
    plt.title('Bz [G]' + '(n' + str(int(n_iter)).zfill(8) + ')')
    plt.colorbar(tpc)
    tpc.set_clim(-1, 1)
    # tpc.set_clim(-16,-13)
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(data_path + file_str + '_' + para_str + '.png')
    plt.close()
    # plt.show()
    frames.append(imageio.imread(data_path + file_str + '_' + para_str + '.png'))
imageio.mimsave(data_path + file_str + '_' + para_str + '.mp4', frames, fps=4)
quit()

fig, ax = plt.subplots()
# ax.plot(data_2d['x'], data_2d['y'], 'o', markersize=2, color='grey')
tpc = ax.tripcolor(data_2d['x'], data_2d['y'], np.array(data_2d['He2pRho'] / data_2d['Rho']) * 100)
plt.colorbar(tpc)
tpc.set_clim(0, 5)
plt.title('AHe [%] (z=0 Cut)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

fig, ax = plt.subplots()
# ax.plot(data_2d['x'], data_2d['y'], 'o', markersize=2, color='grey')
tpc = ax.tripcolor(data_2d['x'], data_2d['y'],
                   np.array(np.sqrt(data_2d['Ux'] ** 2 + data_2d['Uy'] ** 2 + data_2d['Uz'] ** 2)))
plt.colorbar(tpc)
tpc.set_clim(5e5, 3e6)
plt.title('｜U｜')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# data_sph = IdlFile('/Users/ephe/THL8/outputsph/SC/3d__ful_1_n00000100.out')
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# sca = ax.scatter(data_sph['x'],data_sph['y'],data_sph['z'],c=np.linspace(0,1,65536),cmap='prism')
# plt.colorbar(sca)

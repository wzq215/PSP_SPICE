import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('alpha_properties.csv')

n = 2
m = 4
i = 1
plt.subplot(n, m, i)
sns.kdeplot(data=df, x=r'Proton Flux', hue='type', common_norm=False, fill=True, alpha=.5, legend=False)
plt.xlabel(r'Proton Flux' + '\n' + r'$[cm^{-3}\cdot km/s]$')
# plt.xlim([0,2.6])

i = 2
plt.subplot(n, m, i)
sns.kdeplot(data=df, x=r'$V_pr$', hue='type', common_norm=False, fill=True, alpha=.5, legend=False)
plt.ylabel('')
plt.xlabel('$V_{pr}$\n $[km/s]$')
# plt.xlim([0,15])

i = 3
plt.subplot(n, m, i)
sns.kdeplot(data=df, x=r'$\sigma_c$', hue='type', common_norm=False, fill=True, alpha=.5, legend=False)
# plt.xlim([-1,2])
plt.ylabel('')

i = 4
plt.subplot(n, m, i)
sns.kdeplot(data=df, x=r'$|\delta V_{A}|$', hue='type', common_norm=False, fill=True, alpha=.5, legend=False)
plt.xlim([-5, 80])
plt.ylabel('')
plt.xlabel(r'$|\delta V_{A}|$' + '\n' + r'$ [km/s]$')

# plt.subplot(2,5,5)
# sns.kdeplot(data=df,x='Tp_para/perp',hue='type',common_norm=False,fill=True,alpha=.5,legend=True)
# # plt.xlim([0,1.2e-8])
# plt.xlabel(r'$T_{p||}/T_{p\perp}$')
# plt.ylabel('')

i = 5
plt.subplot(n, m, i)
sns.kdeplot(data=df, x=r'A_{\alpha}', hue='type', common_norm=False, fill=True, alpha=.5, legend=False)
plt.xlabel(r'$A_{\alpha}$' + '\n' + r'$[\%]$')
plt.xlim([-0.1, 2.6])
plt.plot([0.12, 0.12], [0, 6.5], 'k--', linewidth=1)

i = 6
plt.subplot(n, m, i)
sns.kdeplot(data=df, x=r'$dV_{\alpha p}/V_{A}$', hue='type', common_norm=False, fill=True, alpha=.5, legend=False)
plt.xlim([-1, 2])
plt.ylabel('')
plt.xlabel(r'$V_{\alpha p}/V_{A}$')
plt.plot([0., 0.], [0, 3.], 'k--', linewidth=1)
plt.plot([0.13, 0.13], [0, 3.], 'k--', linewidth=1)

i = 7
plt.subplot(n, m, i)
sns.kdeplot(data=df, x=r'$T_{\alpha}/T_{p}$', hue='type', common_norm=False, fill=True, alpha=.5, legend=False)
plt.ylabel('')
plt.xlabel(r'$T_{\alpha}/T_{p}$')
plt.xlim([-3, 13])
plt.plot([3.28, 3.28], [0, 0.32], 'k--', linewidth=1)
plt.plot([4.5, 4.5], [0, 0.28], 'k--', linewidth=1)
plt.plot([7.0, 7.0], [0, 0.1], 'k--', linewidth=1)

i = 8
plt.subplot(n, m, i)
sns.kdeplot(data=df, x=r'Ac', hue='type', common_norm=False, fill=True, alpha=.5, legend=True)
# plt.xlim([0,1.2e-8])
plt.xlabel('Collisional Age')
plt.ylabel('')

# plt.subplot(2,5,10)
# print(df[df['Ta_para/perp']<0])
# df.loc[df['Ta_para/perp']<0] = np.nan
# print(df[df['Ta_para/perp']<0])
# sns.kdeplot(data=df,x='Ta_para/perp',hue='type',common_norm=False,fill=True,alpha=.5,legend=True)
# # plt.xlim([0,1.2e-8])
# plt.xlabel(r'$T_{\alpha ||}/T_{\alpha \perp}$')
# plt.xlim([-2,6])
# plt.ylabel('')


plt.show()

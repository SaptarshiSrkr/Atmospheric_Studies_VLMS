import emcee
import corner
import matplotlib.pyplot as plt

reader = emcee.backends.HDFBackend('logfile_norm_RVcorr_LHS72.txt.h5')
burnin = 0

corner.corner(reader.get_chain(flat=True, discard=burnin),labels=['Teff', 'log g', '[M/H]'], quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
plt.savefig('corner.png',dpi=500,facecolor='white')

fig, ax = plt.subplots(3, sharex=True,figsize=(10,8))

for i in range(3):
    ax[i].plot(reader.get_chain(discard=burnin)[:, :, i], '-k', alpha=0.5)
    
ax[0].set_ylabel('Teff',fontsize=15)
ax[1].set_ylabel('log g',fontsize=15)
ax[2].set_ylabel('[M/H]',fontsize=15)
fig.savefig('chains.png',dpi=500,facecolor='white')
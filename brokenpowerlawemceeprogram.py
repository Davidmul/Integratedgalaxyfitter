import numpy as np
import emcee
import matplotlib.pyplot as pl
import corner
import scipy.optimize as op
import os
import pandas as pd
import seaborn as sns

# Using colours so plots are accessable to the colour blind
cb_dark_blue = (0/255,107/255,164/255)
cb_orange = (255/255,128/255,14/255)
cb_red = (200/255,82/255,0/255)

#change this into an input from parset
#modeluse = 0

os.system('mkdir galfit_output')

# Need to change
#Gaussian likehood
m_ls = 1
b_ls = -0.5
c_ls = 0.8

def powerlaw(x,m,b):
    return m*(x**b)

def brokenpowerlaw(x,m,b,c):
    return m*(x**b/(1 + np.sqrt(x/c)))

def lnlikepower(theta, x, y, yerr):
    m, b = theta
    model = powerlaw(x,m,b)
    inv_sigma2 = 1.0/(yerr**2)
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))


def lnlikebroken(theta, x, y, yerr):
    m, b, c = theta
    model = brokenpowerlaw(x,m,b,c)
    inv_sigma2 = 1.0/(yerr**2)
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))



#Your prior function
def lnpriorpower(theta):
    m, b = theta
    if 0 < m < 100 and -2 < b < 0:
        return 0.0
    return -np.inf


def lnpriorbroken(theta):
    m, b, c = theta
    if 0 < m < 100 and -2 < b < 0 and 0 < c < 1:
        return 0.0
    return -np.inf

# full log prob function

def lnprobpower(theta, x, y, yerr):
    lp = lnpriorpower(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlikepower(theta, x, y, yerr)


def lnprobbroken(theta, x, y, yerr):
    lp = lnpriorbroken(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlikebroken(theta, x, y, yerr)



def freqplot(modeluse,x,y,yerr, *args):
    ax = pl.subplot()
    ax.set_xscale('log')
    ax.set_yscale('log')
    pl.errorbar(x, y, yerr=yerr,fmt='o',color=cb_dark_blue)
    if modeluse == 0:
        pl.plot(x,powerlaw(x,m_ls,b_ls),'--',color=cb_red)
        pl.plot(x,powerlaw(x,power_m_ml,power_b_ml),'-',color=cb_orange)
        #Need to add more to plot
        pl.savefig('powerlawfreqplot.pdf')
        pl.close()
        os.system('mv powerlawfreqplot.pdf galfit_output')
    elif modeluse == 1:
        pl.plot(x,brokenpowerlaw(x,m_ls,b_ls,c_ls),'--',color=cb_red)
        pl.plot(x,brokenpowerlaw(x,broken_m_ml,broken_b_ml,broken_c_ml),'-',color=cb_orange)
        #Need to add more to plot
        pl.savefig('brokenpowerlawfreqplot.pdf')
        pl.close()
        os.system('mv brokenpowerlawfreqplot.pdf galfit_output')

def compute_mcmc(modeluse,ndim,x,y,yerr,nwalkers=250,samples=1000,nburn=500):
    if modeluse == 0:
        pos = [powerresult["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobpower, args=(x, y, yerr))
        sampler.run_mcmc(pos, samples)
        trace = sampler.chain[:, nburn:, :].reshape((-1, ndim))
    elif modeluse == 1:
        pos = [brokenresult["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobbroken, args=(x, y, yerr))
        sampler.run_mcmc(pos, samples)
        trace = sampler.chain[:, nburn:, :].reshape((-1, ndim))
    return trace


# Input the integrated fluxes

x, y, yerr = np.loadtxt('exampleNGC891.txt',unpack=True, usecols=[0,1,2])
x=x/1e9

#frequencist fitting

nll = lambda *args: -lnlikepower(*args)
powerresult = op.minimize(nll, [m_ls,b_ls], args=(x, y, yerr))
power_m_ml, power_b_ml = powerresult["x"]
print('Fitted values for a power law are:')
print (power_m_ml, power_b_ml)

nll = lambda *args: -lnlikebroken(*args)
brokenresult = op.minimize(nll, [m_ls,b_ls,c_ls], args=(x, y, yerr))
broken_m_ml, broken_b_ml, broken_c_ml = brokenresult["x"]
print('Fitted values for a broken power law are:')
print (broken_m_ml, broken_b_ml, broken_c_ml)

freqplot(0,x,y,yerr,m_ls,b_ls,power_m_ml,power_b_ml)
freqplot(1,x,y,yerr,m_ls,b_ls,c_ls,broken_m_ml,broken_b_ml,broken_c_ml)

#emcee fitting

powertrace=compute_mcmc(0,2,x,y,yerr)

columns = [r'$\theta_{0}$'.format(i) for i in range(3)]
df_2D = pd.DataFrame(powertrace, columns=columns[:2])

with sns.axes_style('ticks'):
    jointplot = sns.jointplot(r'$\theta_0$', r'$\theta_1$',
                              data=df_2D, kind="hex");
pl.savefig('powerlaw.pdf')
pl.close()
os.system('mv powerlaw.pdf galfit_output')


brokentrace=compute_mcmc(1,3,x,y,yerr)

df_3D = pd.DataFrame(brokentrace, columns=columns[:3])

# get the colormap from the joint plot above
cmap = jointplot.ax_joint.collections[0].get_cmap()

with sns.axes_style('ticks'):
    grid = sns.PairGrid(df_3D)
    grid.map_diag(pl.hist, bins=30, alpha=0.5)
    grid.map_offdiag(pl.hexbin, gridsize=50, linewidths=0, cmap=cmap)

pl.savefig('brokenpowerlaw.pdf')
pl.close()
os.system('mv brokenpowerlaw.pdf galfit_output')


#m_mlarray=np.ones(samples)*m_ml
#b_mlarray=np.ones(samples)*b_ml
#c_mlarray=np.ones(samples)*c_ml

#xaxis = np.linspace(1,samples,samples)
#for n in range(0,nwalkers):
#	pl.plot(xaxis,sampler.chain[n,:,0],'k')
#pl.plot(xaxis,m_mlarray,'r')
#pl.xlabel('Step size')
#pl.show()
#for n in range(0,nwalkers):
#        pl.plot(xaxis,sampler.chain[n,:,1],'k')
#pl.plot(xaxis,b_mlarray,'r')
#pl.xlabel('Step size')
#pl.show()
#for n in range(0,nwalkers):
#        pl.plot(xaxis,sampler.chain[n,:,2],'k')
#pl.plot(xaxis,c_mlarray,'r')
#pl.xlabel('Step size')
#pl.show()


xl = np.array([0, 10])
ax = pl.subplot()
ax.set_xscale('log')
ax.set_yscale('log')
for m, b, c in brokentrace[np.random.randint(len(brokentrace), size=10)]:#
    pl.plot(x, m*(x**b/(1 + np.sqrt(x/c))), color="k", alpha=0.1)
pl.plot(x,broken_m_ml*(x**broken_b_ml/(1 + np.sqrt(x/broken_c_ml))),'-',color='r')
pl.errorbar(x, y, yerr=yerr, fmt=".b")
pl.xlim([0.055,16.5])
pl.xlabel('Frequency')
pl.ylim([0,8])
pl.ylabel('Integrated flux(Jy)')
pl.savefig('integratedplotfit.pdf')
pl.show()


#samples[:, 2] = np.exp(samples[:, 2])
#m_mcmc, b_mcmc,f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))

#print (m_mcmc)
#print (b_mcmc)
#print (f_mcmc)

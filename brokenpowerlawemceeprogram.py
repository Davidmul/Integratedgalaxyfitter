import numpy as np
import emcee
import matplotlib.pyplot as pl
import corner
import scipy.optimize as op

#Gaussian likehood

m_ls = 1
b_ls = -0.5
c_ls = 0.8

def lnlike(theta, x, y, yerr):
    m, b, c = theta
    model = m*(x**b/(1 + np.sqrt(x/c)))
    inv_sigma2 = 1.0/(yerr**2)
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

#Your prior function
def lnprior(theta):
    m, b, c = theta
    if 0 < m < 100 and -2 < b < 0 and 0 < c < 1:
        return 0.0
    return -np.inf


# full log prob function

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)



# Input the integrated fluxes

x, y, yerr = np.loadtxt('myentiredata.dat',unpack=True, usecols=[0,1,2])
x=x/1e9


nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [m_ls,b_ls,c_ls], args=(x, y, yerr))
m_ml, b_ml, c_ml = result["x"]
print (m_ml, b_ml, c_ml)

ax = pl.subplot()
ax.set_xscale('log')
ax.set_yscale('log')
pl.errorbar(x, y, yerr=yerr,fmt='ko')
pl.plot(x,m_ls*(x**b_ls/(1 + np.sqrt(x/c_ls))),'--',color='k')
pl.plot(x,m_ml*(x**b_ml/(1 + np.sqrt(x/c_ml))),'-',color='r')
pl.show()


ndim, nwalkers = 3, 250
samples = 3000
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
sampler.run_mcmc(pos, samples)

m_mlarray=np.ones(samples)*m_ml
b_mlarray=np.ones(samples)*b_ml
c_mlarray=np.ones(samples)*c_ml

xaxis = np.linspace(1,samples,samples)
for n in range(0,nwalkers):
	pl.plot(xaxis,sampler.chain[n,:,0],'k')
pl.plot(xaxis,m_mlarray,'r')
pl.xlabel('Step size')
pl.show()
for n in range(0,nwalkers):
        pl.plot(xaxis,sampler.chain[n,:,1],'k')
pl.plot(xaxis,b_mlarray,'r')
pl.xlabel('Step size')
pl.show()
for n in range(0,nwalkers):
        pl.plot(xaxis,sampler.chain[n,:,2],'k')
pl.plot(xaxis,c_mlarray,'r')
pl.xlabel('Step size')
pl.show()

samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
#fig = corner.corner(samples, labels=[r"$F_0$", r"$\alpha$",r"$\nu_b$",r"$F_1$"],truths=[2.66,-0.37, 1.36])
fig = corner.corner(samples, labels=[r"$F_0$", r"$\alpha$",r"$\nu_b$",r"$F_1$"]) # Hummel
#fig = corner.corner(samples, labels=[r"$F_0$", r"$\alpha$",r"$\nu_b$"])
fig.savefig("triangle.eps")
#pl.show()

xl = np.array([0, 10])
ax = pl.subplot()
ax.set_xscale('log')
ax.set_yscale('log')
for m, b, c in samples[np.random.randint(len(samples), size=100)]:
    pl.plot(x, m*(x**b/(1 + np.sqrt(x/c))), color="k", alpha=0.1)
pl.plot(x,m_ml*(x**b_ml/(1 + np.sqrt(x/c_ml))),'-',color='r')
pl.errorbar(x, y, yerr=yerr, fmt=".b")
pl.xlim([0.055,16.5])
pl.xlabel('Frequency')
pl.ylim([0,8])
pl.ylabel('Integrated flux(Jy)')
pl.savefig('integratedplotfit.pdf')
#pl.show()


samples[:, 2] = np.exp(samples[:, 2])
m_mcmc, b_mcmc,f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))

print (m_mcmc)
print (b_mcmc)
print (f_mcmc)

import emcee

def integrated_time(x, c=5, tol=0.05, quiet=True):
    return emcee.autocorr.integrated_time(x, c=c, tol=tol, quiet=quiet)
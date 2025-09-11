# thejoker-jax

# Data
- [SDSS-V Wiki](https://sdss-wiki.atlassian.net/)

# Todo
- Hogg wants a corner plot of the parameters on the posterior
- Hogg wants 16 samples of the posterior (so that means you choose these $\theta$s), so then choose these and plot it as (RV vs JD) (black is data, faint red is our samples, Joker is faint blue)

# Todo Old
# Jonathan (on measure transport & NNs)
- I wonder if there's a way that you can paramterize your neural network using canonical transformation. These transformation are volume perserving, so we can construct a NN which is volume perserving.

## Yifan
- Yifan noticed some discrepancies in performance of my affine invariant HMC implementation vs his numpy implementation. I need to debug this...
- Implement the affine invaraint NUTs sampler in `JAX`
- Implement the affine invariant stop condition

## Hogg
- Implement data: "The velocity you want to use is V_HELIO (although it is *actually* V_BARY), and the associated error."
- Can you plot maybe 40 stars (radial velocity vs. time; data is always black; I want error bars) and I will choose a few of them to be our poster children. Can you also look to see which stars were the poster children in The Joker papers? (Jonathan suggested lookoing at Fengi, Goodman, Hogg 2014 for inspiration on particular stars)
- Implement sampling from posterior function in https://arxiv.org/pdf/1610.07602
- Jonathan: I wonder if we can construct a hierarchical model of RV data to determine how many planets are orbiting a star. From there we can use the model evidence to determine the number of planets.

## Reading
- Read Wang & Landau sampling
- Read Boltzmann Generator by Frank Noe (method of paramterizing NNs for distributions)


import matplotlib as mpl
import matplotlib.pylab as pl
import numpy as np

def summary_plot(state, samples, layer=None):
    def MAD(d):
        return np.median(np.abs(d - np.median(d)))

    s = state
    s.set_current_particle()
    t = s.create_final_image()

    if layer is None:
        layer = t.shape[0]/2

    mu = samples.mean(axis=0)
    std = samples.std(axis=0)

    fig, axs = pl.subplots(2,3, figsize=(20,12))
    axs[0][0].imshow((s.image[s._cmp_region]*s._cmp_mask)[layer], vmin=0, vmax=1)
    axs[0][1].imshow((t[s._cmp_region]*s._cmp_mask)[layer], vmin=0, vmax=1)
    axs[0][2].imshow(((s.image-t)[s._cmp_region]*s._cmp_mask)[layer], vmin=-1, vmax=1)
    axs[0][0].set_xticks([])
    axs[0][0].set_yticks([])
    axs[0][1].set_xticks([])
    axs[0][1].set_yticks([])
    axs[0][2].set_xticks([])
    axs[0][2].set_yticks([])

    axs[1][0].hist(std[s.b_pos], bins=np.logspace(-3,0,50), label='Positions',
            histtype='stepfilled', alpha=0.8)
    axs[1][0].hist(std[s.b_rad], bins=np.logspace(-3,0,50), label='Radii',
            histtype='stepfilled', alpha=0.8)
    axs[1][0].semilogx()
    axs[1][0].legend(loc='upper right')
    axs[1][0].set_xlabel("Estimated standard deviation")

    d = s.state[s.b_rad]
    m = 2*1.4826 * MAD(d)
    mb = d.mean()

    d = d[(d > mb - m) & (d < mb +m)]
    d = s.state[s.b_rad]
    axs[1][1].hist(d, bins=50, histtype='stepfilled', alpha=0.8)
    axs[1][1].set_xlabel("Radii")

    axs[1][2].hist(((s.image-t)[s._cmp_region]*s._cmp_mask).ravel(), bins=150,
            histtype='stepfilled', alpha=0.8)
    axs[1][2].set_xlim(-0.35, 0.35)
    axs[1][2].semilogy()
    axs[1][2].set_xlabel("Pixel value differences")

    pl.tight_layout()

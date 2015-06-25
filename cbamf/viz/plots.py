import matplotlib as mpl
import matplotlib.pylab as pl
import numpy as np
import time

def summary_plot(state, samples, zlayer=None, xlayer=None, truestate=None):
    def MAD(d):
        return np.median(np.abs(d - np.median(d)))

    s = state
    t = s.get_model_image()

    if zlayer is None:
        zlayer = t.shape[0]/2

    if xlayer is None:
        xlayer = t.shape[2]/2

    mu = samples.mean(axis=0)
    std = samples.std(axis=0)

    fig, axs = pl.subplots(3,3, figsize=(20,12))
    axs[0][0].imshow(s.image[zlayer], vmin=0, vmax=1)
    axs[0][1].imshow(t[zlayer], vmin=0, vmax=1)
    axs[0][2].imshow((s.image-t)[zlayer], vmin=-1, vmax=1)
    axs[0][0].set_xticks([])
    axs[0][0].set_yticks([])
    axs[0][1].set_xticks([])
    axs[0][1].set_yticks([])
    axs[0][2].set_xticks([])
    axs[0][2].set_yticks([])

    axs[1][0].imshow(s.image[:,:,xlayer], vmin=0, vmax=1)
    axs[1][1].imshow(t[:,:,xlayer], vmin=0, vmax=1)
    axs[1][2].imshow((s.image-t)[:,:,xlayer], vmin=-1, vmax=1)
    axs[1][0].set_xticks([])
    axs[1][0].set_yticks([])
    axs[1][1].set_xticks([])
    axs[1][1].set_yticks([])
    axs[1][2].set_xticks([])
    axs[1][2].set_yticks([])

    alpha = 0.5 if truestate is not None else 0.8
    axs[2][0].hist(std[s.b_pos], bins=np.logspace(-3,0,50), label='Positions',
            histtype='stepfilled', alpha=alpha)
    if truestate is not None:
        d = np.abs(mu - truestate)
        axs[2][0].hist(d[s.b_pos], bins=np.logspace(-3,0,50), color='red',
                histtype='step', alpha=1)

    axs[2][0].hist(std[s.b_rad], bins=np.logspace(-3,0,50), label='Radii',
            histtype='stepfilled', alpha=alpha)
    if truestate is not None:
        d = np.abs(mu - truestate)
        axs[2][0].hist(d[s.b_rad], bins=np.logspace(-3,0,50), color='blue',
                histtype='step', alpha=1)

    axs[2][0].semilogx()
    axs[2][0].legend(loc='upper right')
    axs[2][0].set_xlabel("Estimated standard deviation")

    d = s.state[s.b_rad]
    m = 2*1.4826 * MAD(d)
    mb = d.mean()

    d = d[(d > mb - m) & (d < mb +m)]
    d = s.state[s.b_rad]
    axs[2][1].hist(d, bins=50, histtype='stepfilled', alpha=0.8)
    axs[2][1].set_xlabel("Radii")

    if truestate is not None:
        axs[2][1].hist(truestate[s.b_rad], bins=50, histtype='step', alpha=0.8)

    axs[2][2].hist((s.image-t)[s.image_mask==1].ravel(), bins=150,
            histtype='stepfilled', alpha=0.8)
    axs[2][2].set_xlim(-0.35, 0.35)
    axs[2][2].semilogy()
    axs[2][2].set_xlabel("Pixel value differences")

    pl.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.05, hspace=0.05)
    pl.tight_layout()

def scan(im, cycles=1, sleep=0.3):
    pl.figure(1)
    pl.show()
    time.sleep(3)
    for c in xrange(cycles):
        for i, sl in enumerate(im):
            print i
            pl.clf()
            pl.imshow(sl, cmap=pl.cm.bone, interpolation='nearest',
                    origin='lower', vmin=0, vmax=1)
            pl.draw()
            time.sleep(sleep)

def scan_together(im, p, delay=2):
    pl.figure(1)
    pl.show()
    time.sleep(3)
    z,y,x = p.T
    for i in xrange(len(im)):
        print i
        sl = im[i]
        pl.clf()
        pl.imshow(sl, cmap=pl.cm.bone, interpolation='nearest', origin='lower',
                vmin=0, vmax=1)
        m = z.astype('int') == i
        pl.plot(x[m], y[m], 'o')
        pl.xlim(0, sl.shape[0])
        pl.ylim(0, sl.shape[1])
        pl.draw()
        time.sleep(delay)


"""
This script runs a stylized version of the first half of the tutorial
Rather than load in images, it creates a fake image automatically and
fits that image.

To ensure reproducibility in the image and fit, we set a seed for
numpy's random number generator as well.  
"""
# All imports that are used in the tutorial
import numpy as np
from peri import util
from peri import models
from peri import states
from peri import runner
from peri.comp import objs
from peri.comp import comp
from peri.comp import ilms
from peri.comp import exactpsf
from peri.viz.interaction import OrthoViewer, OrthoPrefeature, OrthoManipulator

# Setting the seed for reproducibility
np.random.seed(10)

# Some hard-coded constants to create an image like the tutorial:
POS = np.array([
    [ 11.21179157,  17.46765157,  51.33777692],
    [ 11.58861193,  26.21946861,   7.25906492],
    [ 12.22432771,  33.29800381,  60.62781668],
    [ 13.81552881,   3.57275724,  56.57804871],
    [ 12.16116169,   9.64765222,  17.41146506],
    [ 12.99814567,  25.05624667,  19.14195006],
    [ 12.33297547,  27.87231729,  32.28411386],
    [ 11.17763533,   7.7027628 ,  46.04353855],
    [ 12.91657615,  46.61849718,  48.63677961],
    [ 14.81333917,  48.40786972,  60.60469214],
    [ 12.84487501,  57.87788986,  20.86220673],
    [ 13.84516711,  16.29322417,  27.40591646],
    [ 14.02667269,  47.73248733,  33.11841744],
    [ 12.74104265,  60.46075745,  53.37425778],
    [ 15.23753621,  19.69860468,  41.39355538],
    [ 15.27358198,  38.40844312,  26.7318029 ],
    [ 14.73745621,  44.55385863,  11.56340392],
    [ 14.5520839 ,  61.13087086,  38.83619259],
    [ 15.9845791 ,   5.01834357,  27.06605462],
    [ 16.82137875,  10.45300331,   4.88053659],
    [ 17.41277388,  56.40107326,   8.04239777],
    [ 18.59775505,  30.36350004,  51.76679362],
    [ 23.56714996,   8.4685478 ,  45.50606527],
    [ 19.934775  ,  14.18808795,  58.83030076],
    [ 23.99089835,  45.85751264,  41.88734528],
    [ 23.54841012,  34.97472514,   7.23703913],
    [ 25.59591733,  51.71560733,  23.59087041],
    [ 25.25933489,  23.53414842,  30.43759373],
    [ 25.40082208,  31.76936258,  38.99569137],
    [ 27.18259911,  19.49270884,  18.0170834 ],
    [ 28.16163767,  23.28581272,   3.53311514],
    [ 31.05964615,   1.95065449,  16.10347881],
    [ 31.84595942,   3.03071488,  56.75101873],
    [ 32.75471038,  44.04633015,  57.00660619],
    [ 34.06851749,  51.35646408,  45.66477822],
    [ 20.81021722,  -1.61882524,   2.86862165],
    [ 33.79707333,  12.93023334,  64.14614371],
    [ 25.65539255,  54.53882984,  65.37602364],
    [ 11.72914944,  49.52578712,  -3.78024544],
    [ 11.99716121,  -3.42522376,  39.93710065],
    [ 12.07976975,  -3.26316596,  17.33460135],
    [ 11.87397218,  28.44581287,  -4.29102325],
    [ 19.44967917,  -3.85110892,  63.58394976],
    [ 32.08661926,  67.28521317,  39.28653216],
    [ 14.83352727,  18.03760489,  -4.66327152],
    [ 19.82193879,  -4.05004067,  48.38192053],
    [ 26.7578654 ,  69.44148187,  29.26388449]])
ILMVALS = [1.9, -0.14, 0.0064, 0.0062, 0.078, 0.12, 0.12, 0.12, 0.12, 0.12,
            0.12, 0.12, 0.11, 0.11, 0.11, 0.11, -0.086, 0.02, 0.014, -0.00019,
            -0.01, -0.0058, 0.0012, 0.0052, -0.0014, -0.026, -0.037, 0.0053,
            -0.017, -0.012, -0.004, 0.00034, -0.0012, -0.011, -0.056, -0.099,
            -0.015, -0.011, -0.00043, -0.091, -0.51, -0.15, 0.39, -0.3, 0.63,
            -0.27, 0.19, -0.09]
BKGVALS = [-1.2, 0.0082, 0.0043, -0.0047, -0.0059, -1.4, 0.82, -0.59, 0.99,
            0.00067, 0.33]

def scramble_positions(p, delete_frac=0.1):
    """randomly deletes particles and adds 1-px noise for a realistic
    initial featuring guess"""
    probs = [1-delete_frac, delete_frac]
    m = np.random.choice([True, False], p.shape[0], p=probs)
    jumble = np.random.randn(m.sum(), 3)
    return p[m] + jumble


def create_img():
    """Creates an image, as a `peri.util.Image`, which is similar
    to the image in the tutorial"""
    # 1. particles + coverslip
    rad = 0.5 * np.random.randn(POS.shape[0]) + 4.5  # 4.5 +- 0.5 px particles
    part = objs.PlatonicSpheresCollection(POS, rad, zscale=0.89)
    slab = objs.Slab(zpos=4.92, angles=(-4.7e-3, -7.3e-4))
    objects = comp.ComponentCollection([part, slab], category='obj')

    # 2. psf, ilm
    p = exactpsf.FixedSSChebLinePSF(kfki=1.07, zslab=-29.3, alpha=1.17,
            n2n1=0.98, sigkf=-0.33, zscale=0.89, laser_wavelength=0.45)
    i = ilms.BarnesStreakLegPoly2P1D(npts=(16,10,8,4), zorder=8)
    b = ilms.LegendrePoly2P1D(order=(7,2,2), category='bkg')
    off = comp.GlobalScalar(name='offset', value=-2.11)
    mdl = models.ConfocalImageModel()
    st = states.ImageState(util.NullImage(shape=[48,64,64]),
            [objects, p, i, b, off], mdl=mdl, model_as_data=True)
    b.update(b.params, BKGVALS)
    i.update(i.params, ILMVALS)
    im = st.model + np.random.randn(*st.model.shape) * 0.03
    return util.Image(im)


def print_info(i): print('~'*10 + '\t{}\t'.format(i)+ '~'*10)


if __name__ == '__main__':
    im = create_img()

    print_info('Creating Components')
    coverslip = objs.Slab(zpos=6)
    positions_guess = scramble_positions(POS)  # a deliberately bad guess
    particle_radii = 5.0  # a good guess for the particle radii, in pixels
    particles = objs.PlatonicSpheresCollection(positions_guess, particle_radii)
    objects = comp.ComponentCollection([particles, coverslip], category='obj')

    illumination = ilms.BarnesStreakLegPoly2P1D(npts=(16, 10, 8, 4), zorder=8)
    background = ilms.LegendrePoly2P1D(order=(7,2,2), category='bkg')
    offset = comp.GlobalScalar(name='offset', value=0.)
    point_spread_function = exactpsf.FixedSSChebLinePSF()

    print_info('Creating Initial State')
    model = models.ConfocalImageModel()
    st = states.ImageState(im, [objects, illumination, background,
            point_spread_function, offset], mdl=model)

    print_info('Linking zscale   ')
    runner.link_zscale(st)

    # Optimization
    print_info('Start Optimization')
    runner.optimize_from_initial(st)
    runner.finish_state(st)
    # ~~ up to improve the mathematical model ~~


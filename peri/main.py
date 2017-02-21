import peri
from peri.logger import log

desc = """
PERI (Parameter Extraction from Reconstruction of Images), a software package
to extract features from experimental microscope images as well as properties
of the apparatus and experimental setup. Currently, we support 3D line-scanning
confocal image of spherical particles in florescent dye. Due to the nature of
this method, please expect long featuring times with some variation in success
at reaching the best fit.
"""

def main():
    import argparse
    parser = argparse.ArgumentParser(description=desc, version="PERI "+peri.__version__)
    sub = parser.add_subparsers()

    # shared arguments between most of the actions
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("-g", "--debug", action='store_true',
        help="run with debugging logs and information")
    shared.add_argument("-v", "--verbose", action='count',
        help="set the verbosity of log messages")
    shared.add_argument("--fftw-threads", default=None, help="""Number of
        threads for pyfftw to use, -1 sets threads to number of cores on the
        machine. (default: -1)""", metavar='')
    shared.add_argument("--fftw-wisdom", default=None, help="""Filename of
        pyfftw wisdom file. Saving wisdom increases speed of first ffts.
        (default: $HOME/.peri-wisdom.pkl)""", metavar='')
    shared.add_argument("--logfile", default=None, help="""Filename of
        PERI logging file which stores copies of all logged messages
        (default: $HOME/.peri.log)""", metavar='')
    shared.add_argument("--logcolors", default=None, help="""Boolean of
        whether to color logs or not, accepts [true / false]. 
        (default: false)""", metavar='')

    # the sub actions that can be performed
    parse_conf = sub.add_parser(name='conf', parents=[shared],
        help="Configure global options for PERI")
    parse_feature = sub.add_parser(name='feature', parents=[shared],
        help="Exact features from a set of images")

    parse_conf.set_defaults(action='conf')
    parse_feature.set_defaults(action='feature')

    # custom actions for each particular action
    parse_feature.add_argument("filename", type=str, nargs='+',
        help="""File(s) to feature, multiple files can be specified by a list
        or by glob. These files must be in tif(f) format for 3D images, or 
        png,jpg,bmp,tif(f) for 2D images. For confocal data, the first image
        axis should be the direction perpendicular to the coverslip."""
    )

    args = vars(parser.parse_args())

    if args.get("debug"):
        log.set_verbosity('vvvvv')

    if args.get('action') == "feature":
        action_build()
    elif args.get('action') == "install":
        action_install(args, not args['skip_build'])


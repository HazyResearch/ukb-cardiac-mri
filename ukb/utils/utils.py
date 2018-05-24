import time
import logging
import numpy as np
from skimage import draw
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


logger = logging.getLogger(__name__)


def print_key_pairs(v, title="Parameters"):
    """
    Print python dictionary key/value pairs

    :param v:       python dictionary
    :param title:   table title
    :return:
    """
    items = v.items() if type(v) is dict else v
    logger.info("-" * 40)
    logger.info(title)
    logger.info("-" * 40)
    for key,value in items:
        logger.info("{!s:<20}: {!s:<10}".format(key, value))
    logger.info("-" * 40)


def print_dict_pairs(d, title="Parameters"):
    """
    Print python dictionary key/value pairs

    :param v:       python dictionary
    :param title:   table title
    :return:
    """
    logger.info("")
    logger.info("=" * 90)
    logger.info(title)
    logger.info("=" * 90)
    if (d is None):
        logger.info("None")
    else:
        items = d.items() if type(d) is dict else d
        for key,value in items:
            if (type(value) is dict):
                for sub_key, sub_value in value.items():
                    logger.info("{!s:<35}: {!s:<10}".format(key + " " + sub_key, sub_value))
            else:
                logger.info("{!s:<35}: {!s:<10}".format(key, value))
    logger.info("")


def timeit(method):
    """
    Decorator function for timing

    :param method:
    :return:
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        result = (te-ts,) + result
        logger.info('%r %2.2f sec' % (method.__name__, te-ts))
        return result
    return timed


def select_frames(seq, epsilon=0.05, plot=False):
    """
    Select a contiguous subset of frames based on each frame's
    overall standard deviation by pixel. Determine cut points based
    on the first pair of inflection points.

    :param seq:
    :param epsilon:
    :param plot:
    :return:
    """
    # z-score transform frames
    z = (seq - np.mean(seq)) / np.std(seq)

    # standard deviation per frame / threshold
    std = [np.std(z[i]) for i in range(seq.shape[0])]
    std = [v - min(std) for v in std]
    std = [v if v > epsilon else 0 for v in std]

    # find inflection points
    signs = [np.sign(std[i + 1] - std[i]) for i in range(len(std) - 1)]
    inf_pnts = [] if signs[0] <= 0 else [0]
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            inf_pnts.append(i)

    # pathological image sequence
    if len(inf_pnts) < 3:
        return (0, len(seq) - 1)

    if plot:
        plt.plot(std)
        plt.show()

    return (inf_pnts[0], inf_pnts[2])


def z_score_normalize(seq):
    """
    Z-score normalization

    :param seq:
    :return:
    """
    return (seq - np.mean(seq)) / np.std(seq)


def seq_as_float(seq):
    """
    Convert 0-255 ubyte image to 0-1 float range

    :param seq:
    :return:
    """
    seq = seq.astype(np.float32)
    return (seq - np.min(seq)) / (np.max(seq) - np.min(seq))


def seq_as_ubyte(seq):
    """
    Convert 0-1 float image to 0-255 range

    :param seq:
    :return:
    """
    return (seq * 255.0).astype(np.uint8)


def get_bounding_box(pnts):
    """
    Compute bounding box given a region of points.

    :param pnts:
    :return:
    """
    min_x, max_x, min_y, max_y = None, None, None, None
    for x,y in pnts:
        min_x, max_x = min(x, min_x), max(x, max_x)
        min_y, max_y = min(y, min_y), max(y, min_y)
    return [min_x, max_x, min_y, max_y]


def seq_to_video(seq, outfpath, width=4, height=4, normalize=False):
    """
    Export Numpy tensor images to .mp4 video.
    see https://stackoverflow.com/questions/43445103/inline-animations-in-jupyter

    :param seq:
    :param outfpath:
    :param width:
    :param height:
    :return:
    """
    def getImageFromList(x):
        return seq[x]

    # z-score normalization
    seq = z_score_normalize(seq.astype(np.float32)) if normalize else seq

    fig = plt.figure(figsize=(width, height))
    ims = []
    for i in range(seq.shape[0]):
        im = plt.imshow(getImageFromList(i), animated=True, cmap='gray', vmin=0, vmax=np.max(seq))
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=60, blit=True, repeat_delay=5)
    plt.close()

    ani.save('{}.mp4'.format(outfpath))


def generate_random_dataset(outdir, n_samples=100, dim=(30, 192, 192)):
    """
    Create random numpy matrices in the same format as our MRI images.
    Generate some simple circle shapes to test segmentation.

    :param n_samples:
    :param dim:
    :return:
    """
    for i in range(1000000, 1000000 + n_samples):
        fpath = "{}/{}_random_{}x{}x{}".format(outdir, i, *dim)
        X = np.zeros(dim)
        # fix random center points
        centers = [np.random.randint(50, 160, 2) for _ in range(6)]
        # create some elliptical variation around these centers
        for j in range(dim[0]):
            for c in centers:
                c_radius = np.random.randint(2, 10)
                r_radius = np.random.randint(5, 10)
                rr, cc = draw.ellipse(c[0], c[1], r_radius, c_radius, shape=X[j].shape)
                X[j, rr, cc] = 10

        np.save(fpath, X)


def format_time(seconds):

    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

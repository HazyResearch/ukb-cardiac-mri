"""
Hacky script to generate synthetic data to debug model training.
Default configuration generates sequences based on empirical attributes

"""
from __future__ import print_function

import os
import sys
import math
import argparse
import numpy as np
from skimage import draw
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import skimage
from skimage.util import random_noise
from skimage.draw import polygon
from skimage.filters import gaussian

try:
    # for python2
    import cPickle
except ImportError:
    # for python3
    import _pickle as cPickle

from utils import *

# empirical distributions from 5000K MRI samples
window_counts = {2: 4, 3: 7, 4: 5, 5: 13, 6: 31, 7: 188, 8: 609, 9: 1170, 10: 1216,
                 11: 876, 12: 479, 13: 233, 14: 85, 15: 49, 16: 14, 17: 9, 18: 1}
start_counts = {0: 228, 1: 3095, 2: 1536, 3: 113, 4: 12}


def get_empirical_dist(counts):
    prob = []
    w, W = 0.0, float(np.sum(list(counts.values())))
    for i in sorted(counts):
        w += counts[i]
        prob.append(w / W)
    return [sorted(counts.keys()), [0.0] + prob]


window_dist = get_empirical_dist(window_counts)
start_dist = get_empirical_dist(start_counts)


def sample_empirical(dist):
    """

    :param dist:
    :return:
    """
    x = np.random.random()
    keys, values = dist
    for i in range(len(dist[-1])):
        if x >= values[i] and x < values[i + 1]:
            return i
    return i - 1


def sample_mri_interval():
    """
    Use empircal distribution of SD to generate a

    peak value:  mean/SD: 0.62573123135359987, 0.23739626787134235

    NOTE: If this were real, we'd want to sample a SD distribution instead of a peak value

    """
    start = sample_empirical(start_dist)
    duration = sample_empirical(window_dist)
    peak = (duration / 2) + start
    peak_value = np.random.normal(0.62573123135359987, 0.23739626787134235)

    return start, duration, peak, round(peak_value, 4)


def get_curve(start, end, peak, peak_max):
    """
    Fit curve to 3 points (simulate transitions for generating animations)

    :param start:
    :param end:
    :param peak:
    :param peak_max:
    :return:
    """
    points = np.array([(start, 0), (peak, peak_max), (end, 0)])
    # get x and y vectors
    x = points[:,0]
    y = points[:,1]

    # calculate polynomial
    z = np.polyfit(x, y, 2)
    f = np.poly1d(z)

    # calculate new x's and y's
    x_new = np.linspace(x[0], x[-1], 100)
    y_new = f(x_new)

    x_new = map(int, x_new)
    xy = []
    curr = None
    for x,y in zip(x_new, y_new):
        if x == curr:
            continue
        curr = x
        xy.append((x,y))
    x,y = zip(*xy)
    y = [abs(round(y_hat/peak_max,3)) for y_hat in y]
    return y


def add_noise(img):
    """
    Hack to add noise to images

    :param img:
    :return:
    """
    img = gaussian(img, sigma=0.5, preserve_range=True)
    img = random_noise(img, mode='gaussian', var=0.01, mean=0.01)
    img = random_noise(img, mode='speckle', var=0.00001)
    img = gaussian(img, sigma=0.7)
    img = random_noise(img, mode='speckle', var=0.00001)
    img = random_noise(img, mode='gaussian', var=0.001, mean=0.0001)
    return img


def sample_mri_class(bav_prob=0.5, num_frames=30, width=48, height=48):
    """

    :param bav_prob:
    :param num_frames:
    :param width:
    :param height:
    :return:
    """
    start, duration, peak, peak_max = sample_mri_interval()
    curve = get_curve(start, start + duration, peak, peak_max)
    curve = ([0.] * start) + curve
    curve = curve + [0.] * (num_frames - len(curve))
    curve = list(map(lambda x: x ** 2, curve))

    size = np.random.uniform(7.0, 9.25)
    r_radius, c_radius = np.random.uniform(0.9, 1.0), np.random.uniform(0.9, 1.0)

    class_type = False if np.random.random() >= bav_prob else True

    r_radius = r_radius * size
    c_radius = c_radius * size

    seq = np.zeros((num_frames, width, height))
    for i in range(num_frames):
        cx,cy = [width/2, height/2]
        cx += np.random.randint(-1, 1)
        cy += np.random.randint(-1, 1)

        if i >= start and i < start + duration:
            rr, cc = draw.ellipse(cx, cy, r_radius * curve[i] if class_type else r_radius,
                                  c_radius if class_type else c_radius * curve[i])
            seq[i][rr, cc] = 1.0 * math.sqrt(curve[i])

        seq[i] = add_noise(seq[i])

    return seq, class_type


def random_color(prob=0.5, width=48, height=48):
    """
    Generate random class from:
        1) Black image
        2) White image

    :param prob:
    :param width:
    :param height:
    :return:
    """
    class_type = False if np.random.random() >= prob else True
    img = np.zeros((1, width, height)) if class_type else np.ones((1, width, height))
    return img, class_type


def random_shape(prob=0.5, width=48, height=48):
    """
    Sample random class from:
        1) 1 circle
        2) 2 circles

    :param prob:
    :param width:
    :param height:
    :return:
    """
    class_type = False if np.random.random() >= prob else True
    col = 0.0 if class_type else 1.0
    w = width * np.random.uniform(0.4, 0.8)

    if class_type:

        img = np.zeros((1, width, height))
        mx, my = width / 2, height / 2

        cx = mx * np.random.rand()
        cy = height * np.random.rand()
        rr, cc = draw.circle(cx, cy, radius=np.random.random() * width / 6.0 , shape=img.shape[1:3])
        img[0, rr, cc] = 1.0

        cx = (width - mx) * np.random.rand()
        cy = height * np.random.rand()
        rr, cc = draw.circle(cx, cy, radius=np.random.random() * width / 6.0, shape=img.shape[1:3])
        img[0, rr, cc] = 1.0

    else:
        img = np.zeros((1, width, height))
        cx, cy = [width / 2, height / 2]
        cx += np.random.randint(-width / 4, height / 4)
        cy += np.random.randint(-width / 4, height / 4)

        rr,cc = draw.circle(cx, cy, radius=np.random.random() * 15, shape=img.shape[1:3])
        img[0, rr, cc] = 1.0

    img[0] = gaussian(img[0], sigma=0.75, preserve_range=True)
    return img, class_type


def generate_random_dataset(outdir, start_id, instance_generator, n_samples=100, prob=0.5, num_frames=15, width=48, height=48, debug=False):
    """
    Create random numpy matrices in the same format as our MRI images.
    Generate some simple circle shapes to test segmentation.

    :param n_samples:
    :param dim:
    :return:
    """
    labels = {}
    start_id += 1000000

    for i in range(start_id, start_id + n_samples):
        fpath = "{}/{}".format(outdir, i)
        if instance_generator == "mri":
            X, y = sample_mri_class(bav_prob=prob, num_frames=num_frames, width=width, height=height)
        elif instance_generator == "bw":
            X, y = random_color(prob=prob, width=args.dim, height=args.dim)
        else:
            X, y = random_shape(prob=prob, width=args.dim, height=args.dim)

        X = seq_as_ubyte(X)

        if debug:
            seq_to_video(X, fpath, width=4, height=4)
        np.save(fpath, X)
        labels[i] = y

    with open("{}/labels.csv".format(outdir),"w") as fp:
        fp.write("ID,LABEL\n")
        for pid in sorted(labels):
            fp.write("{},{}\n".format(pid, int(labels[pid])))


def main(args):


    np.random.seed(args.seed)

    if not os.path.exists(args.outdir):
        logger.error("{} does not exist!".format(args.outdir))
        return

    if args.full:
        for i, dirname in enumerate(["train","dev","test"]):
            fpath = "{}/{}".format(args.outdir, dirname)
            if not os.path.exists(fpath):
                os.mkdir(fpath)
            generate_random_dataset(fpath, i * args.samples, instance_generator=args.generator, n_samples=args.samples,
                                    prob=args.prob, num_frames=args.num_frames, width=args.dim, height=args.dim,
                                    debug=args.debug)
    else:
        generate_random_dataset(args.outdir, 0, instance_generator=args.generator, n_samples=args.samples, prob=args.prob,
                                num_frames=args.num_frames, width = args.dim, height = args.dim, debug=args.debug)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()

    argparser.add_argument("-o", "--outdir", type=str, default=None, help="save files to outdir")
    argparser.add_argument("-n", "--samples", type=int, default=None, help="create n MRI sequences")
    argparser.add_argument("--full", action="store_true", help="create full train/validation/test splits")

    argparser.add_argument("-I", "--generator", type=str, default="bw", help="bw / shape / MRI sequences")

    argparser.add_argument("-F", "--num_frames", type=int, default=30, help="output frame length - default: 30")
    argparser.add_argument("-D", "--dim", type=int, default=32, help="output dimension - default: 32x32")
    argparser.add_argument("-P", "--prob", type=float, default=0.5, help="prob")

    argparser.add_argument("--debug", action="store_true", help="export debug sequences as MP4")
    argparser.add_argument("--seed", type=int, default=1234, help="random seed")

    FORMAT = '%(levelname)s|%(name)s|  %(message)s'
    logging.basicConfig(format=FORMAT, stream=sys.stdout, level=logging.INFO)

    args = argparser.parse_args()
    print_key_pairs(args.__dict__, title="Parameters")
    main(args)

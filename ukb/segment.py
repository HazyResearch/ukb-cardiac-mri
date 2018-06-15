"""
Phase Contrast Cardiac MRI Segmentation

Prepare MRIs for training a CNN model. Given an input directory of numpy image tensors
containing phase contrast cardiac MRIs:

- Generate candidate value segmentations
- Rank candidates in terms of the most likely atrial value
- Write segmentation masks to numpy files
- Export 32x32, 48x48 cropped images

@author jason-fries [at] stanford [dot] edu

"""
from __future__ import print_function
import os
import re
import sys
import time
import glob
import logging
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation


from skimage.measure import label
from skimage import filters, segmentation

from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, dilation, erosion

from scipy.ndimage.filters import uniform_filter
from skimage.restoration import denoise_wavelet, denoise_nl_means
from skimage.transform import rescale
from skimage.morphology import square, disk
from skimage.filters import threshold_local
from skimage import img_as_float, img_as_ubyte

from utils import *

logger = logging.getLogger(__name__)


def get_centroid(x, y, weights=None):
    """
    Compute average of provided points. Optionally weight points (doesn't usually matter).

    :param x:
    :param y:
    :param weights:
    :return:
    """
    x_mu = np.average(x, weights=weights).astype(int)
    y_mu = np.average(y, weights=weights).astype(int)
    return [x_mu, y_mu]


def score_segmentations(img, labeled, weighted_centroid=True, min_threshold=2, max_threshold=1000):
    """
    Compute a pixel mask for each labeled segment and calculate it's centroid.
    Discard masks with more than max_threshold pixels or less than min_threshold.

    :param img:
    :param labeled:
    :param weighted_centroid:
    :param min_threshold:
    :param max_threshold:
    :return:
    """
    segments = []
    for s_id in range(max(labeled.flatten()) + 1):
        # get coordinates of this segment
        y, x = np.where(labeled == s_id)
        # pixel weights
        w = img[labeled == s_id]
        num_pixels = len(w.flatten())
        if num_pixels >= max_threshold or num_pixels <= min_threshold:
            continue
        segments.append([np.sum(w), s_id, num_pixels, get_centroid(x, y, weights=w)])

    # rank candidates
    return rank_valve_cands(sorted(segments, reverse=1))


def rank_valve_cands(segments):
    """
    Heuristic for selecting probable atrial valve. Take top 2 weighted segments and
    check their spatial orientation. Basic idea is that the atrial valve is *usually*
    the largest, highest intensity region located in the lower left region of the MRI image.

    2/14/2018 Spot check of 194 examples: 192/194 correct

    :param segments:
    :return:
    """
    assert len(segments) > 0

    if len(segments) == 1:
        return segments[0:1]

    # select top 2 candidates
    a = segments[0]
    b = segments[1]
    c = [] if len(segments) > 2 else segments[2:]

    # segments.append([np.sum(w), s_id, num_pixels, get_centroid(x, y, weights=w)])
    a_x, a_y = a[-1]
    b_x, b_y = b[-1]
    a_w = a[0]
    b_w = b[0]

    # when there is a large disparity between weighted areas, use the largest area
    if b_w < 0.50 * a_w:
        return segments

    # check spatial position of 1st ranked segment vs. 2nd ranked
    if (a_x >= b_x and a_y <= b_y) or (a_x <= b_x and a_y <= b_y):
        target = [b, a] + c
    else:
        target = segments

    return target


def get_segmentation_masks(labeled, segments):
    """
    n x height x width
    1...n segmentation masks

    Each layer is a single region, ranked by liklihood of being the atrial valve
    Last layer is the inverse mask (i.e., all non-valve areas)

    :param X:
    :return:
    """
    masks = []
    for seg in segments:
        _, seg_id, _, _ = seg
        mask = np.copy(labeled)
        mask[mask != seg_id] = 0
        mask[mask == seg_id] = 1
        masks.append(mask)

    mask = np.copy(labeled)
    mask[mask == 0] = 100
    mask[mask != 100] = 0
    mask[mask == 100] = 1
    masks.append(mask)

    return np.array(masks, dtype=np.float32)


def get_segmentation_masks_v2(labeled, segments):
    """
    Array of masks, each with a unique int id, 1...n

    Each "layer" is a single region, ranked by liklihood of being the atrial valve 1..n
    0 is the inverse mask (i.e., all non-valve areas)

    :param X:
    :return:
    """
    mask = np.zeros(labeled.shape)
    for i,seg in enumerate(segments):
        _, seg_id, _, _ = seg
        mask = np.copy(labeled)
        mask[np.where(labeled == seg_id)] = i+1

    return mask


def crop(img, bbox):
    """
    Crop image. Accepts frame data (frames X height X width) or a single 2D image

    :param x:
    :param bbox:
    :return:
    """
    assert len(img.shape) >= 2
    if len(img.shape) == 3:
        return img[...,bbox[0]:bbox[1],bbox[2]:bbox[3]]
    else:
        return img[bbox[0]:bbox[1], bbox[2]:bbox[3]]


def get_crop_region(x, y, dim=48):
    """
    Get bounding box centered on the centroid of the point set x,y.

    :param max_dim:
    :return:
    """
    width = max(x) - min(x)
    height = max(y) - min(y)
    x_pad = (dim - width) / 2
    y_pad = (dim - height) / 2

    # add pixels as needed
    x_slack = 0
    y_slack = 0
    if (2 * x_pad) + width != dim:
        x_slack = dim - ((2 * x_pad) + width)
    if (2 * y_pad) + height != dim:
        y_slack = dim - ((2 * y_pad) + height)

    return [min(x) - x_pad - x_slack, max(x) + x_pad, min(y) - y_pad - y_slack, max(y) + y_pad]





def localize_aortic_valve(img, pooling="std", outfpath=None, debug=False):
    """
    Use a set of heuristics to find the region of the aortic valve.

    :return:
    """
    # compute pooled pixel intensities
    X = np.std(img, axis=0) if pooling == "std" else np.max(img, axis=0)

    labeled = segment(X, upscale=1.0, denoise=False)

    # rank segment candidates (most likely atrial valve)
    segments = score_segmentations(X, labeled)
    masks = get_segmentation_masks(labeled, segments)

    # debug: save segmentations as a PNG
    if debug:
        target = segments[0]
        cx, cy = target[-1]

        plt.figure(figsize=(6, 6))
        plt.imshow(labeled, cmap='tab10')
        plt.scatter(x=cx, y=cy, c='r', s=20)
        plt.savefig(outfpath)
        plt.close()

    return masks


def segment(X, upscale=1.0, denoise=False):
    """

    :param X:
    :param upscale:
    :param denoise:
    :return:
    """
    if upscale > 1.0:
        X = rescale(X, upscale)
    if denoise:
        X = denoise_wavelet(X)

    thresh = filters.threshold_otsu(X)
    bw = closing(X > thresh, square(3))
    cleared = clear_border(bw)

    cleared = rescale(cleared, 1.0 / upscale)
    return label(cleared)


def export_segment(fpath, outfpath, dim, pooling="none", mask_type="none", fmt="npy", debug=True):
    """
    Given an MRI numpy image of dim: frames X height X width,
    generate a segmentation mask for valve candidates.

    Segmentation code based on sample from
    http://douglasduhaime.com/posts/simple-image-segmentation-with-scikit-image.html

    :param fpath:
    :param outfpath:
    :param dim:     crop dimensions
    :param fmt:     (frames|max_pool|std_pool|video) image format options
    :param mask_type: (None|hard|soft) DEFAULT: None
    :param debug:
    :return:
    """
    # 1: LOAD/PREPROCESS IMAGE
    img = np.load(fpath)
    if len(img.shape) != 3:
        raise ValueError('DICOM / numpy array is empty')

    # compute pixel intensity SD percentiles
    X = np.std(img, axis=0)

    # 2: SEGMENTATION
    labeled = segment(X, upscale=1.0, denoise=False)

    # rank segment candidates (most likely atrial valve)
    segments = score_segmentations(X, labeled)
    target = segments[0]
    cx, cy = target[-1]

    # debug: save segmentations as a PNG
    if debug:
        plt.figure(figsize=(6, 6))
        plt.imshow(labeled, cmap='tab10')
        plt.scatter(x=cx, y=cy, c='r', s=20)
        plt.savefig(outfpath)
        plt.close()

    # save all valve masks (index 0 is the most likely atrial valve)
    masks = get_segmentation_masks(labeled, segments)

    # debug: dump each image mask as a PNG
    if debug:
        for m in range(masks.shape[0]):
            plt.figure(figsize=(6, 6))
            plt.imshow(masks[m], cmap='tab10')
            plt.savefig(outfpath + "_{}".format(m))
            plt.close()

    # get segment mask points, compute bounding box, and crop original image
    px, py = np.where(masks[0] == 1)
    bbox  = get_crop_region(px, py, dim)
    c_img = crop(img, bbox)

    # mask data: by default, don't mask anything
    mask = np.ones((bbox[1] - bbox[0], bbox[3] - bbox[2]), dtype=np.float32)
    if mask_type in ["soft", "hard"]:
        msk = np.copy(masks[0])
        exp_msk = dilation(msk)
        exp_msk = crop(exp_msk, bbox)
        mask = filters.gaussian(exp_msk, sigma=1.01) if mask_type == "soft" else exp_msk

    # 3: EXPORT IMAGE DATA
    img_path = "{}_{}x{}".format(outfpath, dim, dim)
    img_path = "{}_{}pool".format(img_path, pooling) if pooling != "none" else img_path
    img_path = "{}_{}".format(img_path, mask_type) if mask_type != "none" else img_path

    # pool data
    if pooling in ["max", "std", "z_add"]:
        if pooling == "max":
            c_img = np.max(c_img, axis=0)
        elif pooling == "std":
            c_img = np.std(c_img, axis=0)
        elif pooling == "z_add":
            c_img = z_score_normalize(c_img)
            c_img = np.sum(c_img, axis=0)
            c_img = (c_img - np.min(c_img)) / (np.max(c_img) - np.min(c_img))

    c_img = (mask * c_img)

    # export format
    if fmt == "png":
        plt.figure(figsize=(4, 4))
        plt.imshow(c_img, cmap='gray')
        plt.savefig(outfpath)

    elif fmt == "mp4":
        seq_to_video(c_img, img_path, width=4, height=4)

    else:
        np.save(img_path, c_img)

    # save segmentation masks
    np.save("{}_masks".format(outfpath), masks.astype(np.int8))


@timeit
def main(args):

    np.random.seed(1234)

    # ------------------------------------------------------------------------------
    # Load Files
    # ------------------------------------------------------------------------------
    filelist = glob.glob("{}*.npy".format(args.indir))
    if args.cohort or args.patients:
        # filter images to only include those in the provided cohort
        if args.cohort:
            ids = map(lambda x:x.strip(), open(args.cohort,"rU").read().splitlines())
        else:
            ids = args.patients.strip().split(",")

        rgx = "({})".format("|".join(ids))
        filelist = [fn for fn in filelist if re.search(rgx, fn)]

    filelist = np.random.choice(filelist, args.samples, replace=False) if args.samples and len(filelist)>args.samples else filelist
    logger.info("Loaded {} MRIs".format(len(filelist)))

    # ------------------------------------------------------------------------------
    # Segment MRIs
    # ------------------------------------------------------------------------------
    errors = []
    for fpath in filelist:
        try:
            pid = re.search("^(\d+)[_]", fpath.split("/")[-1]).group(1)
            outfpath = "{}/{}".format(args.outdir, pid)

            #img = np.load(fpath)
            #masks = localize_aortic_valve(img)
            #bbox = [region for region in regionprops(masks[0])][0].bbox
            #bbox = get_crop_region(x, y, dim=48):

            export_segment(fpath, dim=args.dim, outfpath=outfpath, pooling=args.pooling,
                    mask_type=args.mask, fmt=args.format, debug=args.debug)

        except Exception as e:
            logger.error("[{}] segmenting image: {}".format(pid, e))
            errors += [pid]

    num_errors = len(errors)
    if num_errors > 0:
        logger.error("{} images failed during segmentation".format(num_errors))
    logger.info("{} images sucessfully segmented".format(len(filelist) - num_errors))


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()

    argparser.add_argument("-i", "--indir", type=str, default=None, help="load MRIs from indir")
    argparser.add_argument("-o", "--outdir", type=str, default=None, help="save files to outdir")

    argparser.add_argument("-c", "--cohort", type=None, default=None, help="load from list of patient pseudo IDs")
    argparser.add_argument("-p", "--patients", type=str, default=None, help="load string of patient pseudo IDs")
    argparser.add_argument("-n", "--samples", type=int, default=None, help="sample n MRI sequences")

    argparser.add_argument("-D", "--dim", type=int, default=32, help="output dimension - default: 32x32")
    argparser.add_argument("-P", "--pooling", action='store', choices=['none', 'max', 'std', 'mean', 'z_add'], default="none",
                           help="pooling method")
    argparser.add_argument("-M", "--mask", action='store', choices=['none', 'hard', 'soft'],
                           default="none", help="apply segmentation mask to atrial valve")
    argparser.add_argument("-F", "--format", action='store', choices=['npy', 'png', 'mp4'],
                           default="npy", help="export format")

    argparser.add_argument("--create", type=int, default=None, help="create random images")
    argparser.add_argument("--debug", action="store_true", help="dump debug PNGs of all segmentation masks")
    argparser.add_argument("--quiet", action="store_true", help="suppress logging")
    args = argparser.parse_args()

    # enable logging
    if not args.quiet:
        FORMAT = '%(levelname)s|%(name)s|  %(message)s'
        logging.basicConfig(format=FORMAT, stream=sys.stdout, level=logging.INFO)

    # generate a random dataset so that we can test data loading
    if args.create:
        generate_random_dataset(args.outdir, n_samples=args.create, dim=(30, args.dim, args.dim))
        sys.exit()

    if args.format == "mp4" and args.pooling != "none":
        logger.error("pooled data cannot be exported to MP4")

    elif args.format == "png" and args.pooling not in ["max", "std", "mean", "z_add"]:
        logger.error("un-pooled data cannot be exported to PNG")
        sys.exit()

    # print all argument variables
    print_key_pairs(args.__dict__.items(), title="Command Line Args")

    main(args)

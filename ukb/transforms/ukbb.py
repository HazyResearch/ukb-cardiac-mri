"""
Custom preprocessing transformation functions for video/sequential frame
MRI data from the UK Biobank

"""
import numpy as np
from skimage.exposure import rescale_intensity

from torchvision.transforms import Lambda


class NullTransform(Lambda):
    """
    Create a null transformation.

    This is to be used when a given augmentation is not selected in the config
    so that it simply returns the input.
    """

    def __init__(self):
        """
        Instantiate lambda x: x.

        Params
        ------
        None
        """
        super(NullTransform, self).__init__(lambda x: x)

class FrameSelectionStd(object):
    """
    Select subset of MRI frames based on pixel variance
    Assumes NUM_FRAMES X WIDTH X HEIGHT tensors
    TODO: Setup for 3 channel images
    """
    def __init__(self, n_frames=15, channel=1, epsilon=0.05):
        """

        :param n_frames:
        :param channel:
        :param epsilon:
        """
        self.n_frames = n_frames
        self.channel  = channel
        self.epsilon  = epsilon

    def std_by_frame(self, seq, normalize=True):
        """
        Compute standard deviation by frame

        :param seq:
        :param normalize:
        :return:
        """
        # z-score transform frames
        z = (seq - np.mean(seq)) / np.std(seq) if normalize else seq
        # standard deviation per frame
        std = [np.std(z[i]) for i in range(seq.shape[0])]
        return [v - min(std) for v in std]

    def select_frames(self, seq, epsilon=0.05):
        """
        Select a contiguous subset of frames based on each frame's
        overall pixel standard deviation. Determine cut points based
        on the first set of inflection points.

            P2
            /\
           /  \
        __/    \__/\____
         P1    P3

        :param seq:
        :param epsilon:
        :return:
        """
        std = self.std_by_frame(seq)
        # threshold SD
        std = [v if v > epsilon else 0 for v in std]
        # find inflection points
        signs = [np.sign(std[i + 1] - std[i]) for i in range(len(std) - 1)]
        # skip if first point is no slope or negative
        inf_pnts = [] if signs[0] <= 0 else [0]
        for i in range(1, len(signs)):
            if signs[i] != signs[i - 1]:
                inf_pnts.append(i)

        if len(inf_pnts) < 3:
            raise ValueError("No inflection points found")

        return (inf_pnts[0], inf_pnts[2])

    def __call__(self, sample):
        # x,y = sample
        # i,j = self.select_frames(x, self.epsilon)
        if (self.n_frames == 30):
            return sample
        i, j = self.select_frames(sample, self.epsilon)
        j = i + self.n_frames
        # return (x[i:j,...], y)
        return sample[i:j,...]


class FrameSelectionVar():
    """
    Frame selector class.

    Select the N best frames from a series to use for classification.
    In this case, the N best frames are the "brightest" sequential frames. The
    frames with the most variance between dark and light pixels, centered
    around the brightest frame (frame with most variance). Nowing how the
    frames are structured (with a lot of noise), we konw that the best frames
    are where the noise dies down and the consentration is on the aortic valve.
    Therefore, with pixel intensities around the valve going up and intensities
    away from the valve go down, we get our largest variance in pixels with
    these frames.
    """

    def __init__(self, n_frames=6):
        """
        Class initialization function.

        Params
        ------
        None
        """
        self.N = n_frames

    def __call__(self, seq):
        """
        Select the BEST frames from the given series.

        Params
        ------
        npy_series  :   np.array
            - numpy array of the series of DICOM images.

        Return
        ------
        list
            - list of the most best (sequential) frames
        """
        if (self.N == seq.shape[0]):
            return seq

        # Otherwise find correct frames to output
        var = [fr.var() for fr in seq]
        varDict = dict((i, fr.var()) for i, fr in enumerate(seq))
        frameIndx = [np.argmax(var)]
        low_, high_ = frameIndx[-1]-1, frameIndx[-1]+1
        if (self.N > 1):
            for i in range(self.N-1):
                if (low_ >= 0 and high_ <= len(seq)-1):
                    if (varDict[low_] > varDict[high_]):
                        frameIndx.append(low_)
                        low_ = sorted(frameIndx)[0] - 1
                        high_ = sorted(frameIndx)[-1] + 1
                    else:
                        frameIndx.append(high_)
                        low_ = sorted(frameIndx)[0] - 1
                        high_ = sorted(frameIndx)[-1] + 1
                elif (low_ == -1):
                    frameIndx.append(high_)
                    low_ = sorted(frameIndx)[0] - 1
                    high_ = sorted(frameIndx)[-1] + 1
                else:
                    frameIndx.append(low_)
                    low_ = sorted(frameIndx)[0] - 1
                    high_ = sorted(frameIndx)[-1] + 1

        return seq[sorted(frameIndx)]


class RescaleIntensity():
    """Rescale pixel values of a DICOM Series so that they span low-high."""

    def __init__(self, out_range=(0.0,255.0)):
        """
        Class initialization function.

        Params
        ------
        None
        """
        self.out_range = out_range

    def __call__(self, series):
        """
        Execute normalization for the given series.

        Params
        ------
        seris   :   np.array
            - DICOM Series as an np.array

        Return
        ------
        np.array
            - new normalized series
        """
        return np.array(
            [rescale_intensity(1.0*frame, out_range=self.out_range) for frame in series])


class GammaCorrection():
    """Enhance Gray Scale Levels of a DICOM Series frame."""

    def __init__(self, gamma=2.0, intensity=255.0):
        """
        Class initialization function.

        Params
        ------
        gamma   :   float
            - gamma correction amount
        """
        assert isinstance(gamma, (int, float))
        self.gamma = gamma
        self.intensity = intensity

    def __call__(self, series):
        """
        Execute gamma correction for the entire series.

        Params
        ------
        series  :   np.array
            - DICOM Series of images as an np.array

        Return
        ------
        np.array
            - new gamma corrected series
        """
        return np.array([self.intensity*(1.0*frame/frame.max())**self.gamma for frame in series])


class StdNormalize(object):
    """Standard Deviation Normalization mu = 0, std = 1.0."""

    def __call__(self, series):
        """
        Execute std normalization for each individual image in the series.

        Params
        ------
        series  :   np.array
            - series of images

        Return
        ------
        stdNorm series
            - standard normalized series
        """
        stdSeries = []
        for img in series:
            stdSeries.append((img - img.mean())/img.std())
        return np.array(stdSeries)

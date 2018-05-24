"""
Custom preprocessing transformation functions for video/sequential frame
MRI data from the UK Biobank

TO BE USED ON MULTI SERIES INPUTS

"""
import numpy as np
from skimage.exposure import rescale_intensity


class NullTransformMulti():
    """
    Create a null transformation (for multiple series inputs).

    This is to be used when a given augmentation is not selected in the config
    so that it simply returns the input.
    """

    def __call__(self, series1, series2, series3):
        """
        Do nothing and return the same series passed in.

        Params
        ------
        seq1    :   np.array
            - numpy array of the MAIN series of DICOM images

        seq2    :   np.array
            - numpy array of the SUB series of DICOM images

        seq3    :   np.array
            - numpy array of the SUB series of DICOM images

        Return
        ------
        seq1, seq2, seq3
            - same as input
        """
        return series1, series2, series3


class FrameSelectionVarMulti():
    """
    Frame selector class (for multiple series inputs).

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

    def __call__(self, seq1, seq2, seq3):
        """
        Select the BEST frames from seq1 and get those frames only for all seq.

        Params
        ------
        seq1    :   np.array
            - numpy array of the MAIN series of DICOM images

        seq2    :   np.array
            - numpy array of the SUB series of DICOM images

        seq3    :   np.array
            - numpy array of the SUB series of DICOM images

        Return
        ------
        seq1, seq2, seq3
            - series of essential (selected) frames (drop the rest)
        """
        if (self.N == seq1.shape[0]):
            return seq1, seq2, seq3

        # Otherwise find correct frames to output
        var = [fr.var() for fr in seq1]
        varDict = dict((i, fr.var()) for i, fr in enumerate(seq1))
        frameIndx = [np.argmax(var)]
        low_, high_ = frameIndx[-1]-1, frameIndx[-1]+1
        if (self.N > 1):
            for i in range(self.N-1):
                if (low_ >= 0 and high_ <= len(seq1)-1):
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

        return seq1[sorted(frameIndx)], seq2[sorted(frameIndx)], seq3[sorted(frameIndx)]


class RescaleIntensityMulti():
    """Rescale pixel values of multiple DICOM Series so that they span low-high."""

    def __init__(self, out_range=(0.0,255.0)):
        """
        Class initialization function.

        Params
        ------
        None
        """
        self.out_range = out_range

    def __call__(self, series1, series2, series3):
        """
        Execute normalization for all of the given series.

        NOTE :
            series1 is the MAIN series

        Params
        ------
        seris1  :   np.array
            - MAIN DICOM Series as an np.array

        seris2  :   np.array
            - SUB DICOM Series as an np.array

        seris3  :   np.array
            - SUB DICOM Series as an np.array

        Return
        ------
        np.array
            - new normalized series
        """
        return (np.array([rescale_intensity(1.0*frame, out_range=self.out_range) for frame in series1]),
                np.array([rescale_intensity(1.0*frame, out_range=self.out_range) for frame in series2]),
                np.array([rescale_intensity(1.0*frame, out_range=self.out_range) for frame in series3]))


class StdNormalizeMulti(object):
    """Standard Deviation Normalization for multiple series mu = 0, std = 1.0."""

    def __call__(self, series1, series2, series3):
        """
        Execute std normalization for each individual image in all of the series.

        Params
        ------
        seris1  :   np.array
            - MAIN DICOM Series as an np.array

        seris2  :   np.array
            - SUB DICOM Series as an np.array

        seris3  :   np.array
            - SUB DICOM Series as an np.array

        Return
        ------
        stdNorm series1 series2 and series3
            - standard normalized series
        """
        stdSeries1, stdSeries2, stdSeries3 = [], [], []
        for i, img in enumerate(series1):
            stdSeries1.append((img - img.mean())/img.std())
            stdSeries2.append((series2[i] - series2[i].mean())/series2[i].std())
            stdSeries3.append((series3[i] - series3[i].mean())/series3[i].std())
        return np.array(stdSeries1), np.array(stdSeries2), np.array(stdSeries3)

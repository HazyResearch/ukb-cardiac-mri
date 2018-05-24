from .ukbb import *
from .augmentations import *
from .multi_series import *

from torchvision.transforms import Compose


class RandomTransforms(object):
    """Base class for a list of transformations with randomness
    Args:
        transforms (list or tuple): list of transformations
    """

    def __init__(self, transforms, out_range=(0.0, 1.0)):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms
        self.out_range = out_range

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomOrder(RandomTransforms):
    """Apply a list of transformations in a random order
    """
    def __call__(self, img):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            img = self.transforms[i](img)
        rescale = RescaleIntensity(out_range=self.out_range)
        img = rescale(img)
        return img


class ComposeMultiChannel(object):
    """Composes several transforms together for multi channel operations.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2, img3):
        for t in self.transforms:
            img1, img2, img3 = t(img1, img2, img3)
        return img1, img2, img3

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


##############################################################################
#   SINGLE Series Transforms (to be used on flow_250_*_MAG)
##############################################################################
############################
#   Preprocessing Transforms
############################
def compose_preprocessing(preprocessing):
    """
    Compose a preprocessing transform to be performed.

    Params
    ------
    preprocessing   :   dict
        - dictionary defining all preprocessing steps to be taken with their
            values

            e.g. {"FrameSelector" : "var",
                  "Rescale_Intensity" : [0, 255],
                  "Gamma_Correction" : 2.0}

    Return
    ------
    torchvision.transforms.Compose
    """
    # Frame Selector
    if (preprocessing["FrameSelector"]["name"] == "FrameSelectionVar"):
        frame_selector = FrameSelectionVar(n_frames=preprocessing["n_frames"])
    else:
        frame_selector = FrameSelectionStd(n_frames=preprocessing["n_frames"],
                                           channel=preprocessing["FrameSelector"]["channel"],
                                           epsilon=preprocessing["FrameSelector"]["epsilon"])

    # Rescale Intensity
    if ("Rescale_Intensity" in preprocessing):
        intensity_rescale = RescaleIntensity(out_range=tuple(preprocessing["Rescale_Intensity"]))
    else:
        intensity_rescale = NullTransform()

    # Gamma Correction
    if ("Gamma_Correction" in preprocessing):
        gamma_correct = GammaCorrection(gamma=preprocessing["Gamma_Correction"]["gamma"],
                                        intensity=preprocessing["Gamma_Correction"]["intensity"])
    else:
        gamma_correct = NullTransform()

    return Compose([frame_selector, intensity_rescale, gamma_correct])


###########################
#   Augmentation Transforms
###########################
def compose_augmentation(augmentations, seed=1234):
    """
    Compose an augmentation transform to be performed.

    Params
    ------
    augmentations   :   dict
        - dictionary defining all augmentation steps to be taken with their
            values

            e.g.
                {
                    "RandomCrop" : {
                        "size"  :   28,
                        "padding"   :   12
                    },
                    "RandomRotation"    :   {
                        "degrees"   :   25
                    },
                    "RandomTranslation" :   {
                        "translate" :   (0.2, 0.8)
                    },
                    "RandomShear"   :   {
                        "shear" :   12.5
                    },
                    "RandomAffine"  :   {
                        "degrees"   :   5,
                        "translate" :   (0.5, 0.5),
                        "scale" :   0.8,
                        "shear" :   15.0
                    },
                    "Randomize" :   0
                }

    Return
    ------
    torchvision.transforms.Compose (ordered transforms)
        OR
    torchvision.transforms.RandomOrder (randomly ordered transforms)
    """
    # Padding
    if ("Pad" in augmentations):
        if ("padding" in augmentations["Pad"]):
            padding = augmentations["Pad"]["padding"]
        else:
            padding = 0

        if ("fill" in augmentations["Pad"]):
            fill = augmentations["Pad"]["fill"]
        else:
            fill = 0

        if ("padding_mode" in augmentations["Pad"]):
            padding_mode = augmentations["Pad"]["padding_mode"]
        else:
            padding_mode = 'constant'

        pad = Pad(
            padding=padding,
            fill=fill, padding_mode=padding_mode)
    else:
        pad = NullAugmentation()

    # Random Horizontal Flip
    if ("RandomHorizontalFlip" in augmentations):
        if ("probability" in augmentations["RandomHorizontalFlip"]):
            probability = augmentations["RandomHorizontalFlip"]["probability"]
        else:
            probability = 0.5

        random_horizontal = RandomHorizontalFlip(p=probability, seed=seed)
    else:
        random_horizontal = NullAugmentation()

    # Random Vertical Flip
    if ("RandomVerticalFlip" in augmentations):
        if ("probability" in augmentations["RandomVerticalFlip"]):
            probability = augmentations["RandomVerticalFlip"]["probability"]
        else:
            probability = 0.5

        random_vertical = RandomVerticalFlip(p=probability, seed=seed)
    else:
        random_vertical = NullAugmentation()

    # Random Cropping
    if ("RandomCrop" in augmentations):
        if ("padding" in augmentations["RandomCrop"]):
            padding = augmentations["RandomCrop"]["padding"]
        else:
            padding = 0

        random_crop = RandomCrop(
            augmentations["RandomCrop"]["size"],
            padding=padding, seed=seed)
    else:
        random_crop = NullAugmentation()

    # Random Rotation
    if ("RandomRotation" in augmentations):
        if ("resample" in augmentations["RandomRotation"]):
            resample = augmentations["RandomRotation"]["resample"]
        else:
            resample = False

        if ("center" in augmentations["RandomRotation"]):
            center = augmentations["RandomRotation"]["center"]
        else:
            center = None

        random_rotation = RandomRotation(
                augmentations["RandomRotation"]["degrees"],
                resample=resample, center=center, seed=seed)
    else:
        random_rotation = NullAugmentation()

    # Random Translation
    if ("RandomTranslation" in augmentations):
        if ("resample" in augmentations["RandomTranslation"]):
            resample = augmentations["RandomTranslation"]["resample"]
        else:
            resample = False

        random_translation = RandomTranslation(
            augmentations["RandomTranslation"]["translate"], resample=resample,
            seed=seed)
    else:
        random_translation = NullAugmentation()

    # Random Shear
    if ("RandomShear" in augmentations):
        if ("resample" in augmentations["RandomShear"]):
            resample = augmentations["RandomShear"]["resample"]
        else:
            resample = False

        random_shear = RandomShear(
            augmentations["RandomShear"]["shear"], resample=resample,
            seed=seed)
    else:
        random_shear = NullAugmentation()

    # Random Affine
    if ("RandomAffine" in augmentations):
        if ("translate" in augmentations["RandomAffine"]):
            translate = augmentations["RandomAffine"]["translate"]
        else:
            translate = None

        if ("scale" in augmentations["RandomAffine"]):
            scale = augmentations["RandomAffine"]["scale"]
        else:
            scale = None

        if ("shear" in augmentations["RandomAffine"]):
            shear = augmentations["RandomAffine"]["shear"]
        else:
            shear = None

        if ("resample" in augmentations["RandomAffine"]):
            resample = augmentations["RandomAffine"]["resample"]
        else:
            resample = False

        if ("fillcolor" in augmentations["RandomAffine"]):
            fillcolor = augmentations["RandomAffine"]["fillcolor"]
        else:
            fillcolor = 0

        random_affine = RandomAffine(
                augmentations["RandomAffine"]["degrees"],
                translate=translate, scale=scale, shear=shear,
                resample=resample, fillcolor=fillcolor, seed=seed)
    else:
        random_affine = NullAugmentation()

    try:
        if (augmentations["Randomize"]):
            if ("PixelRange" in augmentations):
                return RandomOrder(
                    [random_crop, random_rotation, random_translation,
                     random_shear, random_affine])
            else:
                return RandomOrder(
                    [random_crop, random_rotation, random_translation,
                     random_shear, random_affine])
    except:  # This will fail when "Randomize" is not defined in augmentations
        pass

    return Compose([pad, random_horizontal, random_vertical, random_crop,
                    random_rotation, random_translation, random_shear,
                    random_affine])


##############################################################################
#   Postprocessing Transforms
##############################################################################
def compose_postprocessing(postprocessing):
    """
    Compose a postprocessing transform to be performed.

    Params
    ------
    postprocessing   :   dict
        - dictionary defining all preprocessing steps to be taken with their
            values

            e.g. {"Name" : "RescaleIntensity"}
                    OR
                 {"Name" : "StdNormalize"}

    Return
    ------
    torchvision.transforms.Compose
    """
    if (postprocessing["Name"] == "StdNormalize"):
        postprocess = StdNormalize()
    else:
        postprocess = RescaleIntensity(out_range=(0.0, 1.0))

    return Compose([postprocess])


##############################################################################
#   MULTIPLE Series Transforms (to be used on ALL flow_250_* series)
##############################################################################
############################
#   Preprocessing Transforms
############################
def compose_preprocessing_multi(preprocessing):
    """
    Compose a preprocessing transform to be performed on MULTI series.

    Params
    ------
    preprocessing   :   dict
        - dictionary defining all preprocessing steps to be taken with their
            values

            e.g. {"FrameSelector" : "var",
                  "Rescale_Intensity" : [0, 255],
                  "Gamma_Correction" : 2.0}

    Return
    ------
    torchvision.transforms.Compose
    """
    # Frame Selector
    if (preprocessing["FrameSelector"]["name"] == "FrameSelectionVarMulti"):
        frame_selector = FrameSelectionVarMulti(n_frames=preprocessing["n_frames"])

    # Rescale Intensity
    if ("RescaleIntensityMulti" in preprocessing):
        intensity_rescale = RescaleIntensityMulti(out_range=tuple(preprocessing["RescaleIntensityMulti"]))
    else:
        intensity_rescale = NullTransformMulti()

    return ComposeMultiChannel([frame_selector, intensity_rescale])


#############################
#   Postprocessing Transforms
#############################
def compose_postprocessing_multi(postprocessing):
    """
    Compose a postprocessing transform to be performed on MULTI series.

    Params
    ------
    postprocessing   :   dict
        - dictionary defining all preprocessing steps to be taken with their
            values

            e.g. {"Name" : "RescaleIntensity"}
                    OR
                 {"Name" : "StdNormalize"}

    Return
    ------
    torchvision.transforms.Compose
    """
    if (postprocessing["Name"] == "StdNormalizeMulti"):
        postprocess = StdNormalizeMulti()
    else:
        postprocess = RescaleIntensityMulti(out_range=(0.0, 1.0))

    return ComposeMultiChannel([postprocess])

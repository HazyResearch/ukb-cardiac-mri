"""
Custom augmentation transform functions for video/sequential frame
MRI data from the UK Biobank

"""
import math
import random
import numbers
import collections
import numpy as np

from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import Lambda


##############################################################################
#   Helper Functions
##############################################################################
def _numpy_to_PIL(img):
    """
    Convert a given numpy array image into a PIL Image.

    Params
    ------
    img :   np.array
        - image to be converted

    Return
    ------
    PIL.Image
        - Image type of numpy array
    """
    return Image.fromarray(img)


def _PIL_to_numpy(img):
    """
    Convert a given PIL.Image into a numpy array.

    Params
    ------
    img :   PIL.Image
        - image to be converted

    Return
    ------
    np.array
        - numpy array of PIL.Image
    """
    return np.array(img)


def _get_inverse_affine_matrix(center, angle, translate, scale, shear):
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, scale, shear) = [ cos(a)*scale    -sin(a + shear)*scale     0]
    #                              [ sin(a)*scale    cos(a + shear)*scale     0]
    #                              [     0                  0          1]
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    angle = math.radians(angle)
    shear = math.radians(shear)
    scale = 1.0 / scale

    # Inverted rotation matrix with scale and shear
    d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
    matrix = [
        math.cos(angle + shear), math.sin(angle + shear), 0,
        -math.sin(angle), math.cos(angle), 0
    ]
    matrix = [scale / d * m for m in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
    matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += center[0]
    matrix[5] += center[1]
    return matrix


def affine(img, angle, translate, scale, shear, resample=0, fillcolor=None):
    """Apply affine transformation on the image keeping image center invariant
    Args:
        img (PIL Image): PIL Image to be rotated.
        angle ({float, int}): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float): shear angle value in degrees between -180 to 180, clockwise direction.
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (int): Optional fill color for the area outside the transform in the output image.
    """
    if not F._is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"

    assert scale > 0.0, "Argument scale should be positive"

    output_size = img.size
    center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
    return img.transform(output_size, Image.AFFINE, matrix, resample, fillcolor=fillcolor)


##############################################################################
#   Augmentation Functions
##############################################################################
class NullAugmentation(Lambda):
    """
    Create a null augmentation transformation.

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
        super(NullAugmentation, self).__init__(lambda x: x)


class RandomCrop(object):
    """Crop a given MRI Image Series at a random location."""

    def __init__(self, size, padding=0, seed=1234):
        """
        Class initialization function.

        Params
        ------
        size    :   (sequence or int)
            - Desired output size of the crop. If size is an int instead
                of sequence like (h, w), a square crop (size, size) is made

        padding :   (int or sequence)
            - (Optional) padding on each border of the image
                Default is 0, i.e no padding. If a sequence of length 4 is
                provided, it is used to pad left, top, right, bottom borders
                respectively
        """
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

        np.random.seed(seed)

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = np.random.randint(0, h - th)
        j = np.random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, series):
        """
        Execute random cropping of series.

        Params
        ------
        series  :   np.array
            MRI Series to be cropped

        Return
        ------
        np.array
            - Cropped series
        """
        # Convert all images in the series to PIL.Image types
        PIL_series = [_numpy_to_PIL(img) for img in series]

        # Pad all images in the series
        if (self.padding > 0):
            PIL_series = [F.pad(img, self.padding) for img in PIL_series]

        # Find crop params
        i, j, h, w = self.get_params(PIL_series[0], self.size)

        # Crop the entire series
        PIL_series = [F.crop(img, i, j, h, w) for img in PIL_series]

        # Convert all images back to numpy array
        return np.array([_PIL_to_numpy(img) for img in PIL_series])


class RandomRotation(object):
    """Rotate a MRI Image Series by an angle."""

    def __init__(self, degrees, resample=False, center=None, seed=1234):
        """
        Class initialization function.

        Params
        ------
        degrees :   (sequence or float or int):
            - Range of degrees to select from. If degrees is a number instead
            of sequence like (min, max), the range of degrees will be
            (-degrees, +degrees)

        resample:   (PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.

        center  :   (2-tuple, optional)
            - Optional center of rotation. Origin is the upper left corner.
                Default is the center of the image.
        """
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        if (resample):
            if (resample == 'NEAREST'):
                self.resample = Image.NEAREST
            elif (resample == 'BILINEAR'):
                self.resample = Image.BILINEAR
            else:
                self.resample = Image.BICUBIC
        else:
            self.resample = Image.BILINEAR
        self.center = center

        np.random.seed(seed)

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = np.random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, series):
        """
        Execute random cropping of series.

        Params
        ------
        series  :   np.array
            MRI Series to be rotated

        Return
        ------
        np.array
            - Rotated series
        """
        # Randomly select angle to rotate the series
        angle = self.get_params(self.degrees)

        # Convert numpy series to PIL series
        PIL_series = [_numpy_to_PIL(img) for img in series]

        # Rotate all images in the series
        PIL_series = [F.rotate(img, angle, self.resample, self.center) for img in PIL_series]

        # Return numpy array
        return np.array([_PIL_to_numpy(img) for img in PIL_series])

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomTranslation(object):
    """Random translation (shifting) of a MRI Image Series."""

    def __init__(self, translate=None, resample=False, seed=1234):
        """
        Class initialization function.

        Params
        ------
        translate   :   (tuple, optional)
            - tuple of maximum absolute fraction for horizontal and vertical
                translations. For example translate=(a, b), then horizontal
                shift is randomly sampled in the range
                    -img_width * a < dx < img_width * a
                and vertical shift is randomly sampled in the range
                    -img_height * b < dy < img_height * b
                Will not translate by default.

        resample:   (PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC)
            - An optional resampling filter.
                See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
                If omitted, or if the image has mode "1" or "P", it is set to
                PIL.Image.NEAREST.
        """
        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if (resample):
            if (resample == 'NEAREST'):
                self.resample = Image.NEAREST
            elif (resample == 'BILINEAR'):
                self.resample = Image.BILINEAR
            else:
                self.resample = Image.BICUBIC
        else:
            self.resample = Image.BILINEAR

        np.random.seed(seed)

    @staticmethod
    def get_params(translate, img_size):
        """Get parameters for translation
        Returns:
            sequence: params to be passed to the translation
        """
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(np.random.uniform(-max_dx, max_dx)),
                            np.round(np.random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        angle = 0.0
        scale = 1.0
        shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, series):
        """
        Execute random translation of series.

        Params
        ------
        series  :   np.array
            MRI Series to be translated

        Return
        ------
        np.array
            - Translated series
        """
        # Convert numpy series to PIL series
        PIL_series = [_numpy_to_PIL(img) for img in series]

        # Get random params for Affine Transform
        ret = self.get_params(self.translate, PIL_series[0].size)

        # Compute Affine Transform for all images in the series
        PIL_series = [affine(img, *ret, resample=self.resample, fillcolor=0) for img in PIL_series]

        # Return numpy array
        return np.array([_PIL_to_numpy(img) for img in PIL_series])

    def __repr__(self):
        if self.translate is not None:
            s = 'translate={translate}'
        if self.resample > 0:
            s += ', resample={resample}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _pil_interpolation_to_str[d['resample']]
        return s.format(name=self.__class__.__name__, **d)


class RandomShear(object):
    """Random shear (stretching) of a MRI Image Series."""

    def __init__(self, shear=None, resample=False, seed=1234):
        """
        Class initialization function.

        Params
        ------
        shear   :   (sequence or float or int, optional)
            - Range of degrees to select from. If degrees is a number instead
                of sequence like (min, max), the range of degrees will be
                (-degrees, +degrees). Will not apply shear by default

        resample:   (PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC)
            - An optional resampling filter.
                See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
                If omitted, or if the image has mode "1" or "P", it is set to
                PIL.Image.NEAREST.
        """
        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        if (resample):
            if (resample == 'NEAREST'):
                self.resample = Image.NEAREST
            elif (resample == 'BILINEAR'):
                self.resample = Image.BILINEAR
            else:
                self.resample = Image.BICUBIC
        else:
            self.resample = Image.BILINEAR

        np.random.seed(seed)

    @staticmethod
    def get_params(shears):
        """Get parameters for shearing
        Returns:
            sequence: params to be passed to the shearing transformation
        """
        angle = 0.0
        translations = (0, 0)
        scale = 1.0

        if shears is not None:
            shear = np.random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, series):
        """
        Execute random shear transform of series.

        Params
        ------
        series  :   np.array
            MRI Series to be transformed

        Return
        ------
        np.array
            - Shear transformed series
        """
        # Get random params for Affine Transform
        ret = self.get_params(self.shear)

        # Convert numpy series to PIL series
        PIL_series = [_numpy_to_PIL(img) for img in series]

        # Compute Affine Transform for all images in the series
        PIL_series = [affine(img, *ret, resample=self.resample, fillcolor=0) for img in PIL_series]

        # Return numpy array
        return np.array([_PIL_to_numpy(img) for img in PIL_series])

    def __repr__(self):
        if self.shear is not None:
            s = 'shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _pil_interpolation_to_str[d['resample']]
        return s.format(name=self.__class__.__name__, **d)


class RandomAffine(object):
    """Random affine transformation of a MRI Image Series."""

    def __init__(self, degrees, translate=None, scale=None, shear=None,
                 resample=False, fillcolor=0, seed=1234):
        """
        Class initialization function.

        Args:
        degrees :   (sequence or float or int)
            - Range of degrees to select from. If degrees is a number instead
                of sequence like (min, max), the range of degrees will be
                (-degrees, +degrees). Set to 0 to desactivate rotations.
        translate   :   (tuple, optional)
            - tuple of maximum absolute fraction for horizontal and vertical
                translations. For example translate=(a, b), then horizontal
                shift is randomly sampled in the range
                    -img_width * a < dx < img_width * a
                and vertical shift is randomly sampled in the range
                    -img_height * b < dy < img_height * b
                Will not translate by default.
        scale   :   (tuple, optional)
            - scaling factor interval, e.g (a, b), then scale is randomly
                sampled from the range
                    a <= scale <= b
                Will keep original scale by default.
        shear   :   (sequence or float or int, optional)
            - Range of degrees to select from. If degrees is a number instead
                of sequence like (min, max), the range of degrees will be
                (-degrees, +degrees). Will not apply shear by default
        resample:   (PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC)
            - An optional resampling filter.
                See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
                If omitted, or if the image has mode "1" or "P", it is set to
                PIL.Image.NEAREST.
        fillcolor   :   (int)
            - Optional fill color for the area outside the transform in the
                output image.
        """
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        if (resample):
            if (resample == 'NEAREST'):
                self.resample = Image.NEAREST
            elif (resample == 'BILINEAR'):
                self.resample = Image.BILINEAR
            else:
                self.resample = Image.BICUBIC
        else:
            self.resample = Image.BILINEAR
        self.fillcolor = fillcolor
        np.random.seed(seed)

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation
        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = np.random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(np.random.uniform(-max_dx, max_dx)),
                            np.round(np.random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = np.random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = np.random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, series):
        """
        Execute random affine transform of series.

        Params
        ------
        series  :   np.array
            MRI Series to be transformed

        Return
        ------
        np.array
            - Affine transformed series
        """
        # Convert numpy series to PIL series
        PIL_series = [_numpy_to_PIL(img) for img in series]

        # Get random params for Affine Transform
        ret = self.get_params(self.degrees, self.translate, self.scale,
                              self.shear, PIL_series[0].size)

        # Compute Affine Transform for all images in the series
        PIL_series = [affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor) for img in PIL_series]

        # Return numpy array
        return np.array([_PIL_to_numpy(img) for img in PIL_series])

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _pil_interpolation_to_str[d['resample']]
        return s.format(name=self.__class__.__name__, **d)


class Pad(object):
    """Pad the given Series on all sides with the given "pad" value.
    Args:
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            constant: pads with a constant value, this value is specified with fill
            edge: pads with the last value at the edge of the image
            reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
            symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, padding, fill=0, padding_mode='constant'):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, series):
        """
        Execute padding for series.

        Params
        ------
        series  :   np.array
            MRI Series to be padded

        Return
        ------
        np.array
            - Padded series
        """
        # Convert numpy series to PIL series
        PIL_series = [_numpy_to_PIL(img) for img in series]

        # Return numpy array
        return np.array([_PIL_to_numpy(F.pad(img, self.padding, self.fill)) for img in PIL_series])

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.padding, self.fill, self.padding_mode)


class RandomHorizontalFlip(object):
    """Horizontally flip the given Series randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5, seed=1234):
        self.p = p
        np.random.seed(seed)

    def __call__(self, series):
        """
        Execute random horizontal flip of series.

        Params
        ------
        series  :   np.array
            MRI Series to be padded

        Return
        ------
        np.array
            - Horizontally flipped series (or not)
        """
        if np.random.random() < self.p:
            # Convert numpy series to PIL series
            PIL_series = [_numpy_to_PIL(img) for img in series]
            # Compute Flip
            PIL_series = [F.hflip(img) for img in PIL_series]
            # Convert PIL series to numpy series
            series = np.array([_PIL_to_numpy(img) for img in PIL_series])

        return series

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    """Vertically flip the given Series randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5, seed=1234):
        self.p = p
        np.random.seed(seed)

    def __call__(self, series):
        """
        Execute random vertical flip of series.

        Params
        ------
        series  :   np.array
            MRI Series to be padded

        Return
        ------
        np.array
            - Vertically flipped series (or not)
        """
        if np.random.random() < self.p:
            # Convert numpy series to PIL series
            PIL_series = [_numpy_to_PIL(img) for img in series]
            # Compute Flip
            PIL_series = [F.vflip(img) for img in PIL_series]
            # Convert PIL series to numpy series
            series = np.array([_PIL_to_numpy(img) for img in PIL_series])

        return series

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

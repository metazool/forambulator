"""Utilities for image processing
Building on the region thresholding examples in skimage
"""
import os
import logging
from PIL import Image
from skimage.measure import label, regionprops
import skimage.io
import skimage.filters


def list_image_filenames(image_dir):
    """Recurse through image_dir, return paths to jpg files"""
    matches = []
    for root, dirnames, filenames in os.walk(image_dir):
        for filename in filenames:
            if filename.lower().endswith('jpg') or filename.lower().endswith('png'):  # noqa: E501
                matches.append(os.path.join(root, filename))
    return matches


def best_guess_crop(props):
    """The foram will often be in the region with second biggest area
    It ought to be in the squarest area in the largest couple of regions
    TODO restore the dominant colour filtering of square text
    """
    props = sorted(props, key=lambda prop: prop.area)
    props.reverse()
    ratios = []
    for index, prop in enumerate(props[0:2]):
        ratio = prop.minor_axis_length / prop.major_axis_length
        ratios.append(ratio)
    use_index = ratios.index(max(ratios))
    best_guess = props[use_index]
    # In some cases we can't threshold the foram and select the
    # largest character instead; in which case return nothing
    if best_guess.area < 100:
        best_guess = None
    return best_guess


class NoForamFound(Exception):
    pass


def crop_foram(filename, directory=None, size=256, pad=4):
    """Accepts a filename of an image collected from Endless Forams
    Finds the region with the actual foram in it, resizes,
    Saves the results in directory if specified,
    Returns the result of the crop
    Accepts image size (default 256) and padding around selection"""

    image = skimage.io.imread(fname=filename)
    image = skimage.color.rgb2gray(image)

    region = regions_threshold(image)
    # In some cases Yen threshold fails, we see the whole image; use Otsu
    # There must be several better ways
    if not region or region.area > 100000:
        region = regions_threshold(
            image, method=skimage.filters.threshold_otsu)
    if not region:
        raise NoForamFound("couldn't identify the foram")

    minr, minc, maxr, maxc = region.bbox
    cropped = None
    try:
        cropped = resize(image[minr - pad:maxr + pad, minc - pad:maxc + pad],
                         (size, size),
                         preserve_range=True)
    except ValueError:
        raise NoForamFound("couldnt resize the crop")

    if directory:
        if not os.path.exists(directory):
            os.makedirs(directory)

        # save each cropped image by its original filename
        filename = filename.split('/')[-1]
        filename = filename.replace('.jpg', '.png')
        skimage.io.imsave(os.path.join(directory, filename), cropped)
    return cropped


def regions_threshold(image, method=skimage.filters.threshold_yen):
    t = method(image)
    mask = image > t

    label_img = label(mask, connectivity=mask.ndim)
    props = regionprops(label_img)

    region = best_guess_crop(props)
    return region


def resize(directory, out_dir, size=(128, 128)):
    for infile in list_image_filenames(directory):
        outfile = os.path.join(out_dir,
                               infile.split('/')[-1])
        if infile != outfile:
            try:
                im = Image.open(infile).convert('L')
                im.thumbnail(size, Image.ANTIALIAS)
                im.save(outfile, "PNG")
            except IOError as err:
                logging.error("cannot resize '%s'" % infile)
                logging.error(err)

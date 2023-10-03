'''
Description: Set of class and function that help handle images data from
different source (jpeg, png, fits, ...)

Created on Feb 22, 2012

@author: fmertens

Requirement: astropy version >= 0.3
'''

import os
import copy
import datetime
import pkg_resources
import numpy as np
import PIL.Image

from scipy import misc
from scipy.ndimage.interpolation import rotate, zoom

RESSOURCE_PATH = 'ressource'

import lightwise.nputils as nputils
import lightwise.signalutils as signalutils


GALAXY_GIF_PATH = os.path.join(RESSOURCE_PATH, "aa.gif")

def p2i(xy_pixel):
    ''' Tramsform [[x, y], ...] to [[y, x]...]'''
    return np.array(xy_pixel).T[::-1].T

def draw_rectangle(array, a, b, value=1):
    x1, x2 = sorted([a[0], b[0]])
    y1, y2 = sorted([a[1], b[1]])
    array[x1:x2 + 1, y1:y2 + 1] = 1

    return array

def gaussian(size, nsigma=None, width=None, center=None, center_offset=None, angle=0):
    ''' size_or_data: if 1D, give the shape of the output, a new array will be created.
                      if 2D, draw the gaussian into the data
        angle: in radians '''
    if nsigma is None and width is None:
        raise ValueError("You need to specify either nsigma or width")

    size = nputils.get_pair(size)

    if width is not None:
        width = nputils.get_pair(width)
        sigmax, sigmay = width / (2. * np.sqrt(2 * np.log(2)))
    else:
        nsigma = nputils.get_pair(nsigma)
        sigmax, sigmay = size / nsigma / 2.

    if center is None:
        center = np.floor(size / 2.)
    if center_offset is not None:
        center += nputils.get_pair(center_offset)

    indices = np.indices(np.array(size, dtype=int))
    p = [0, 1, center, [sigmax, sigmay], angle]

    return nputils.gaussian_fct(*p)(indices)


def multiple_gaussian(size, heights, widths, centers):
    indicies = np.indices(size)
    indicies = np.expand_dims(indicies, -1)
    indicies = np.repeat(indicies, len(heights), -1)

    x, y = indicies
    center_x, center_y = centers.T
    center_x = center_x[None, None, :]
    center_y = center_y[None, None, :]

    sigma = widths / (2. * np.sqrt(2 * np.log(2)))
    sigmax, sigmay = sigma.T
    sigmax = sigmax[None, None, :]
    sigmay = sigmay[None, None, :]

    u = (x - center_x) ** 2 / (2. * sigmax ** 2) + (y - center_y) ** 2 / (2. * sigmay ** 2)
    g = heights * np.exp(-u)

    return g.sum(axis=-1)


def gaussian_cylinder(size, nsigma=None, width=None, angle=None, center_offset=0):
    if nsigma is None and width is None:
        raise ValueError("You need to specify either nsigma or width")

    sizex, sizey = nputils.get_pair(size)

    if width is not None:
        a = 1 / (2. * np.sqrt(2 * np.log(2)))
        sigma = lambda y: a * nputils.make_callable(width)(y)
    else:
        a = sizex / 2.
        sigma = lambda y: a / nputils.make_callable(nsigma)(y)

    if angle is not None:
        off = center_offset
        center_offset = lambda y: off + np.tan(angle) * y
    else:
        center_offset = nputils.make_callable(center_offset)

    hsx = sizex / 2
    x, y = np.mgrid[-hsx:hsx + sizex % 2, 0:sizey]

    g = np.exp(-(x + center_offset(y)) ** 2 / (2. * sigma(y) ** 2))

    return g


def ellipsoide(size, a, b=None):
    if b is None:
        b = a
    hs = size / 2
    a = float(a)
    b = float(b)
    x, y = np.mgrid[-hs:hs + size % 2, -hs:hs + size % 2]
    x = x.astype(np.complex)
    z = np.sqrt(1 - x ** 2 / a ** 2 - y ** 2 / b ** 2).real
    return z / z.max()

def galaxy():
    gif = PIL.Image.open(pkg_resources.resource_stream(__name__, GALAXY_GIF_PATH))
    return np.array(list(gif.getdata())).reshape(256, 256)


def lena():
    return misc.face()

class AbstractBeam(object):

    def __init__(self):
        self._beam = None

    def convolve(self, img, boundary="zero"):
        if self._beam is None:
            self._beam = self.build_beam()
        if isinstance(self._beam, tuple):
            c = nputils.convolve(img, self._beam[0], mode='same', boundary=boundary, axis=0)
            return nputils.convolve(c, self._beam[1], mode='same', boundary=boundary, axis=1)
        return nputils.convolve(img, self._beam, mode='same', boundary=boundary)

    def build_beam(self):
        raise NotImplementedError()

    def set_header(self, header, projection):
        pass


class IdleBeam(AbstractBeam):

    def __init__(self):
        AbstractBeam.__init__(self)

    def __str__(self):
        return "IdleBeam"

    def convolve(self, img, boundary="zero"):
        return img

    def build_beam():
        return np.array([[1, ], ])


class GaussianBeam(AbstractBeam):

    def __init__(self, bmaj, bmin, bpa=0, nsigma=None):
        ''' bmaj, bmin in pixel, bpa in radians'''
        self.bmin = bmin
        self.bmaj = bmaj
        self.bpa = bpa
        self.nsigma = nsigma
        AbstractBeam.__init__(self)

    def __str__(self):
        return "GaussianBeam:X:%s,Y:%s,A:%s" % (self.bmaj, self.bmin, self.bpa)

    def build_beam(self):
        sigmax = nputils.gaussian_fwhm_to_sigma(self.bmaj)
        sigmay = nputils.gaussian_fwhm_to_sigma(self.bmin)
        support_x = nputils.get_next_odd(nputils.gaussian_support(sigmax, nsigma=self.nsigma))
        support_y = nputils.get_next_odd(nputils.gaussian_support(sigmay, nsigma=self.nsigma))
        support = max(support_x, support_y)
        width = [self.bmaj, self.bmin]
        # if beam is separable, go for the faster computation
        if self.bpa != 0 and self.bmaj != self.bmin:
            beam = gaussian(support, width=width, angle=self.bpa)
            beam = beam / beam.sum()
        elif self.bmaj != self.bmin:
            beam = (signalutils.gaussian(support_x, width=self.bmaj),
                    signalutils.gaussian(support_y, width=self.bmin))
            beam = (beam[0] / beam[0].sum(), beam[1] / beam[1].sum())
        else:
            beam = signalutils.gaussian(support, width=self.bmaj)
            beam = beam / beam.sum()
        return beam

    def set_header(self, header, projection=None):
        if projection:
            scale = projection.mean_pixel_scale()
        else:
            scale = 1
        header.set("BMIN", self.bmin * scale)
        header.set("BMAJ", self.bmaj * scale)
        header.set("BPA", np.degrees(self.bpa))

class ImageSet(object):
    ''' Object used to store beam information
        until we can save full detection result'''

    def __init__(self):
        self.images = dict()

    def merge(self, other):
        self.images.update(other.images)

    def add(self, epoch, filename, beam):
        self.images[epoch] = (filename, beam)

    def add_img(self, img):
        epoch = img.get_epoch()
        beam = img.get_beam()
        filename = "image"
        if isinstance(img, PilImage):
            filename = img.get_filename()
        assert epoch not in self.images
        self.add(epoch, filename, beam)

    def get_filename(self, epoch):
        return self.images[epoch][0]

    def get_beam(self, epoch):
        return self.images[epoch][1]

    def get_epochs(self):
        return sorted(self.images.keys())

    def to_file(self, filename, projection):
        '''Format is: filename, epoch, bmaj, bmin, bpa'''
        array = []
        scale = float(projection.mean_pixel_scale())
        for epoch, (img_filename, beam) in nputils.get_items_sorted_by_keys(self.images):
            epoch = float(nputils.datetime_to_epoch(epoch))
            if isinstance(beam, GaussianBeam):
                beam_data = [beam.bmaj * scale, beam.bmin * scale, beam.bpa]
            else:
                beam_data = [0, 0, 0]
            array.append([img_filename, str(epoch)] + beam_data)
        np.savetxt(filename, np.array(array, dtype=object), fmt=["%s", "%s", "%.5f", "%.5f", "%.5f"])
        print("Saved image set ", filename)

    @staticmethod
    def from_file(filename, projection):
        new = ImageSet()
        array = np.loadtxt(filename, dtype=str)
        scale = float(projection.mean_pixel_scale())
        for file, epoch, bmaj, bmin, bpa in array:
            epoch = nputils.epoch_to_datetime(float(epoch))
            if float(bmaj) != 0:
                beam = GaussianBeam(float(bmaj) / scale, float(bmin) / scale, float(bpa))
            else:
                beam = IdleBeam()
            new.add(epoch, file, beam)

        print("Loaded image set from ", filename)
        return new

class Image(object):

    EPOCH_COUNTER = 0

    def __init__(self, data, epoch=None, beam=None, pix_ref=None):
        self.data = data
        if beam is None:
            beam = IdleBeam()
        self.beam = beam
        if pix_ref is None:
            pix_ref = np.array(self.data.shape)[::-1] / 2
        self.pix_ref = pix_ref
        if epoch is None:
            self.epoch = Image.EPOCH_COUNTER
            Image.EPOCH_COUNTER += 1
        else:
            self.epoch = epoch

    def __add__(self, other):
        if isinstance(other, Image):
            other = other.get_data()
        new = self.copy()
        new.get_data() + other
        return new

    def __sub__(self, other):
        if isinstance(other, Image):
            other = other.get_data()
        new = self.copy()
        new.get_data() - other
        return new

    def __mul__(self, other):
        if isinstance(other, Image):
            other = other.get_data()
        new = self.copy()
        new.get_data() * other
        return new

    def __div__(self, other):
        if isinstance(other, Image):
            other = other.get_data()
        new = self.copy()
        new.get_data() / other
        return new

#    def get_meta(self):
        # MEM ISSUE
#        return ImageMeta(self.get_epoch(), self.get_coordinate_system(), self.get_beam())'''

#    def get_coordinate_system(self):
        # MEM ISSUE
#        return PixelCoordinateSystem(self.data.shape, self.pix_ref, pix_unit=self.pix_unit)

#    def get_projection(self, *args, **kargs):
#        if len(args) == 1 and isinstance(args[0], ProjectionSettings):
#            settings = args[0]
#        else:
#            settings = ProjectionSettings(*args, **kargs)
#        return self.get_coordinate_system().get_projection(settings)'''

    def has_beam(self):
        return self.get_beam() is not None

    def get_beam(self):
        return self.beam

    def get_pix_ref(self):
        return self.pix_ref

    def set_pix_ref(self, pix_ref):
        self.pix_ref = pix_ref

    def get_title(self):
        return ""

    def get_epoch(self, str_format=None):
        if str_format and isinstance(self.epoch, datetime.time):
            return self.epoch.strftime('%Y-%m-%d')
        return self.epoch

    def resize(self, shape, padding_mode="center"):
        self.data, padding_index, array_index = nputils.resize(self.data, shape,
                                                               padding_mode=padding_mode, output_index=True)

        def i(n):
            if n is None:
                return 0
            return n
        shift = [i(array_index[0]) - i(padding_index[0]), i(array_index[1]) - i(padding_index[1])]
        self.set_pix_ref(np.round(self.get_pix_ref() - shift))
        return shift

    # def partition(self, bx, by, ox=0, oy=0):
    #     img = self.get_data()
    #     for x in range(0, img.shape[0] - ox, bx - ox):
    #         for y in range(0, img.shape[0] - oy, by - oy):
    #             yield ImageRegion(img, (x, y, min(x + bx, img.shape[0]), min(y + by, img.shape[1])))

    def crop(self, xy_p1, xy_p2, projection=None):
        ''' xy_p1 and xy_p2 as pix, except if projection is provided '''
        if projection is not None:
            xy_p1, xy_p2 = np.round(projection.s2p([xy_p1, xy_p2])).astype(int)
        ex = [0, self.data.shape[1]]
        ey = [0, self.data.shape[0]]
        xlim1, xlim2 = sorted([nputils.clamp(xy_p1[0], *ex), nputils.clamp(xy_p2[0], *ex)])
        ylim1, ylim2 = sorted([nputils.clamp(xy_p1[1], *ey), nputils.clamp(xy_p2[1], *ey)])
        # print xlim1, xlim2, ylim1, ylim2
        self.data = self.data[ylim1:ylim2, xlim1:xlim2].copy()
        xy_p1 = np.array([xlim1, ylim1])
        xy_p2 = np.array([xlim2, ylim2])
        self.set_pix_ref(np.round(self.get_pix_ref() - xy_p1))
        return xy_p1, xy_p2

    def shift(self, delta, projection=None):
        ''' delta as xy_pix, except if projection is provided '''
        if projection is not None:
            delta = delta / projection.pixel_scales()
        # print "Shift:", delta
        self.data = nputils.shift2d(self.data, np.round(p2i(delta)))

    def rotate(self, angle_rad, spline_order=0, smooth_len=3):
        self.data = rotate(self.data, - angle_rad / (2 * np.pi) * 360, reshape=False, order=spline_order)

        if smooth_len > 0:
            self.data = nputils.smooth(self.data, smooth_len, mode='same')

        rmatrix = np.array([[np.cos(angle_rad), np.sin(angle_rad)], [-np.sin(angle_rad), np.cos(angle_rad)]])
        center = p2i(np.array(self.data.shape) / 2.)
        self.set_pix_ref(center + np.dot(self.get_pix_ref() - center, rmatrix))

        return rmatrix

    def zoom(self, factor, order=3, mode='constant', cval=0):
        self.data = zoom(self.data, factor, order=order, mode=mode, cval=cval)

        self.set_pix_ref(factor * self.get_pix_ref())

    @staticmethod
    def from_image(image, data=None):
        ''' Return a copy of image with new data set to zeros or to an optional data'''
        new = image.copy()
        if data is None:
            data = np.zeros_like(image.data)
        new.set_data(data)
        return new

    def copy(self, full=False):
        new = copy.copy(self)
        if full:
            new.data = self.data.copy()
        return new

    def set_data(self, data):
        self.data = data

    def get_data(self):
        return self.data

class PilImage(Image):

    def __init__(self, file):
        self.file = file
        self.pil_image = PIL.Image.open(file)
        data = np.array(self.pil_image, dtype=np.float64)
        if data.ndim == 3:
            data = data.mean(axis=2)

        Image.__init__(self, data)

    def __str__(self):
        return "Image(%s)" % os.path.basename(self.file)

    def get_filename(self):
        return self.file

    def save(self, filename, format=None, **params):
        img = PIL.Image.fromarray(self.data)
        img.save(filename, format=format, **params)

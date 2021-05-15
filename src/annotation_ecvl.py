import openslide
from pyecvl import ecvl
import numpy
import cv2
import xml.etree.ElementTree as etree
import math

#pyecvl             0.9.1

pi = math.pi
DEFAULT_OFFSET_X = 0
DEFAULT_OFFSET_Y = 0
DEFAULT_MPPX = 0.44
DEFAULT_MPPY = 0.44
DEFAULT_DIM = (0,0)


def center_in_pixels_absolute(nanozoomer_img):
    if nanozoomer_img is None:
       dim = DEFAULT_DIM
    else:
       dim = nanozoomer_img.dimensions
    return (int(dim[0] / 2), int(dim[1] / 2))


def point_rel_to_centre(nanozoomer_img, point):
    if nanozoomer_img is None:
        offset_x = DEFAULT_OFFSET_X
        offset_y = DEFAULT_OFFSET_Y
    else:
        offset_x = int(nanozoomer_img.properties['hamamatsu.XOffsetFromSlideCentre']) 
        offset_y = int(nanozoomer_img.properties['hamamatsu.YOffsetFromSlideCentre']) 

    center = (offset_x, offset_y)
    return (point[0] - center[0], point[1] - center[1])


def point_rel_to_centre_pixels(nanozoomer_img, point):

    if nanozoomer_img is None:
        mppx = DEFAULT_MPPX
        mppy = DEFAULT_MPPY 
    else:
        mppx = float(nanozoomer_img.properties['openslide.mpp-x'])
        mppy = float(nanozoomer_img.properties['openslide.mpp-y'])

    p_rel_hamamatsu_coords = point_rel_to_centre(nanozoomer_img, point)
    mppx_nms = mppx * 1000
    mppy_nms = mppy * 1000
    p_rel_pixels = (p_rel_hamamatsu_coords[0] / mppx_nms, p_rel_hamamatsu_coords[1] / mppy_nms)
    p_rel_pixels_int = (int(p_rel_pixels[0]), int(p_rel_pixels[1]))
    return p_rel_pixels_int


def dimenions_to_pixels(nanozoomer_img, dimensions):
    if nanozoomer_img is None:
        mppx = DEFAULT_MPPX
        mppy = DEFAULT_MPPY 
    else:
        mppx = float(nanozoomer_img.properties['openslide.mpp-x'])
        mppy = float(nanozoomer_img.properties['openslide.mpp-y'])

    mppx_nms = mppx * 1000
    mppy_nms = mppy * 1000
    return (int(dimensions[0] / mppx_nms), int(dimensions[1] / mppy_nms))


def point_absolute_pixels(nanozoomer_img, point):
    c = center_in_pixels_absolute(nanozoomer_img)
    p_rel = point_rel_to_centre_pixels(nanozoomer_img, point)

    #avoid overflowing maps
    if nanozoomer_img is None:
       dim = DEFAULT_DIM
    else:
       dim = nanozoomer_img.dimensions

    x = min(max(0,p_rel[0] + c[0]),dim[0])
    y = min(max(0,p_rel[1] + c[1]),dim[1])   

    return (x, y)


def crop_polygon(img, polygon, rgba = False):
    # read image as RGB and add alpha (transparency)
    
    #NOT IMPLEMENTED
    #ecvl.ChangeColorSpace(img, img, ecvl.ColorType.RGBA)
    img = numpy.moveaxis(numpy.array(img), [0, 1, 2], [2, 1, 0])
    if rgba:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

    # convert to numpy (for convenience)
    imArray = numpy.asarray(img)
    
    # create mask
    maskIm = numpy.full((imArray.shape[0],imArray.shape[1]), 0, dtype=numpy.uint8)
    
    #polygon = [numpy.array(p) for p in polygon]
    polygon = numpy.array(polygon).reshape((-1,1,2)).astype(numpy.int32)
    #maskIm = cv2.polylines(maskIm, polygon, isClosed=True, color=255, thickness=-1)
    #maskIm = cv2.fillPoly(maskIm, pts = polygon, color=255)
    maskIm = cv2.drawContours(maskIm, [polygon], contourIdx=-1, color=255, thickness=-1)

    mask = numpy.array(maskIm)

    # assemble new image (uint8: 0-255)
    newImArray = numpy.empty(imArray.shape, dtype='uint8')

    # colors (three first columns, RGB)
    newImArray[:, :, :3] = imArray[:, :, :3]

    # transparency (4th column)
    if rgba:
        newImArray[ :, :, 3] = mask # * 255
    else:
        newImArray = newImArray * (mask//255)[:,:,numpy.newaxis]

    # back to Image from numpy

    #NOT IMPLEMENTED
    #ecvl.ChangeColorSpace(newImArray, newIm, ecvl.ColorType.RGBA)
    newImArray = numpy.moveaxis(newImArray, [0, 1, 2], [2, 1, 0])
    if rgba:
        newIm = ecvl.Image.fromarray(newImArray,'cxy',ecvl.ColorType.RGBA)
    else:
        newIm = ecvl.Image.fromarray(newImArray,'cxy',ecvl.ColorType.RGB)
    #newIm = cv2.cvtColor(newImArray, cv2.COLOR_RGB2RGBA)

    return newIm


class Annotation(object):
    def __init__(self, xml_annotation, ndpi_filename=None):
     
        self.ndpi_filename = ndpi_filename
        self.xml = xml_annotation
        self.title = xml_annotation.find('title').text
        self.x = int(xml_annotation.find('x').text)
        self.y = int(xml_annotation.find('y').text)
        self.z = int(xml_annotation.find('z').text)
        self.lens = float(xml_annotation.find('lens').text.replace(',','.'))

        self.annotation = xml_annotation.find("annotation")
        self.color = self.annotation.get("color")
        self.type = self.annotation.get("type")
        self.displayname = self.annotation.get("displayname")

        if ndpi_filename is not None:
            self.img = openslide.OpenSlide(ndpi_filename)
            self.dim = self.img.dimensions
        else:
            self.img = None

    @property
    def points(self):
        if self.type == "freehand":
            #raise TypeError("Wrong annotation type!")
            point_list = self.annotation.findall('pointlist/point')

            ps = []

            for i, p in enumerate(point_list):
                point = (int(p.find('x').text), int(p.find('y').text))
                ps.append(point)
            return ps
        elif self.type == "circle":
            def PointsInCircum(r,cx,cy,n=100):
                return [(math.cos(2*pi/n*x)*r + cx,math.sin(2*pi/n*x)*r + cy) for x in range(0,n)]

            return PointsInCircum(int(self.annotation.find('radius').text),int(self.annotation.find('x').text),int(self.annotation.find('y').text),n=32)
        else:
            raise TypeError("Wrong annotation type!")


    @property
    def points_absolute(self):
        ps = []
        for p in self.points:
            ps.append(point_absolute_pixels(self.img, p))
        return ps

    @property
    def points_xs(self):
        ps = self.points
        xs = []
        for x, y in ps:
            xs.append(x)
        return xs

    @property
    def points_ys(self):
        ps = self.points
        ys = []
        for x, y in ps:
            ys.append(y)
        return ys

    def get_enclosing_rectangle(self):
        ps = self.points
        xs = self.points_xs
        ys = self.points_ys

        min_x, max_x, max_y, min_y = min(xs), max(xs), max(ys), min(ys)

        w = abs(max_x - min_x)
        h = abs(max_y - min_y)

        return (min_x, min_y, w, h)

    def get_enclosing_rectangle_pixels(self):

        if self.img is not None:
            min_x, min_y, w, h = self.get_enclosing_rectangle()
            topleft_pixel = point_absolute_pixels(self.img, (min_x, min_y))
            w_pixels, h_pixels = dimenions_to_pixels(self.img, (w, h))

            min_x,min_y = topleft_pixel
            min_x = max(min_x,0)
            min_y = max(min_y,0)
            w_pixels = min(w_pixels,self.dim[0]-min_x)
            h_pixels = min(h_pixels,self.dim[1]-min_y)

            return (min_x, min_y, w_pixels, h_pixels)

    def get_enclosing_image(self, boxcut=None,level=0):
        if self.img is not None:
            if boxcut is not None:
                rect = boxcut
            else:
                rect = self.get_enclosing_rectangle_pixels()

            scale = 2**level

            ## (x,y) have to be at level 0 reference
            ##OLD
            ##return self.img.read_region((rect[0], rect[1]), level, (rect[2]//scale, rect[3]//scale))
            return ecvl.OpenSlideRead( self.ndpi_filename , level, (rect[0], rect[1], rect[2]//scale, rect[3]//scale))

    def get_points_within_enclosing_rectangle(self,boxcut=None,level=0):

        if boxcut is not None:
            x, y, w_pixels, h_pixels = boxcut
        else:
            x, y, w_pixels, h_pixels = self.get_enclosing_rectangle_pixels()

        scale = 2**level
        ps = self.points_absolute
        
        x = x // scale
        y = y // scale

        ps_rel = []
        for p in ps:
            p_rel = (p[0]//scale - x, p[1]//scale - y)
            #p_rel = (p_rel[0]//scale, p_rel[1]//scale)
            ps_rel.append(p_rel)

        return ps_rel

    def get_image_cropped(self,boxcut=None, level=0, rgba = False):
        polygon = self.get_points_within_enclosing_rectangle(boxcut,level)
        img = crop_polygon(self.get_enclosing_image(boxcut,level), polygon, rgba = rgba)
        #img = crop_polygon(self.img.read_region((boxcut[0], boxcut[1]), level, (boxcut[2], boxcut[3])), polygon)
        return img


def get_xml_viewstates_list(annotation_filename):
    xmltree = etree.parse(annotation_filename)
    root = xmltree.getroot()
    viewstate_list = root.findall('ndpviewstate')

    return viewstate_list


def get_annotation_list(annotation_filename, ndpi_filename=None):
    annotation_list = get_xml_viewstates_list(annotation_filename)
    aa = []
    for anno in annotation_list:
        a = Annotation(anno, ndpi_filename=ndpi_filename)
        if len(a.points)>2:
           aa.append(a)
    return aa

def ecvl_to_cv2(img ):
  return numpy.moveaxis(numpy.array(img), [0, 1, 2], [2, 1, 0])

def cv2_to_ecvl(img, ct = ecvl.ColorType.RGB ):
  return ecvl.Image.fromarray(numpy.moveaxis(img, [0, 1, 2], [2, 1, 0]), 'cxy', ct)
  
if __name__ == '__main__':
  import annotation_ecvl as annotation
  aa = annotation.get_annotation_list( "/wholeslides/metadata/100-B2-TVALG.ndpi.ndpa","/wholeslides/wsi/TVA.LG/100-B2-TVALG.ndpi")
  print(len(aa))  
  for i,a in enumerate(aa):
    print(a.get_enclosing_rectangle_pixels())
  
    
    #print(a.img.level_dimensions)
    maxsize = (133,133)

    img = a.get_enclosing_image(level=0)
    ecvl.ResizeDim(img, img, list(maxsize), interp=ecvl.InterpolationType.linear)
    ecvl.ImWrite('imgs/test{}-1.jpg'.format(i), img)


    img = a.get_image_cropped(level=0)
    #NOT IMPLEMENTED for alpha channel
    ecvl.ResizeDim(img, img, list(maxsize), interp=ecvl.InterpolationType.linear)
    ecvl.ImWrite('imgs/test{}-2.jpg'.format(i), img)
    #img = numpy.moveaxis(numpy.array(img), [0, 1, 2], [2, 1, 0])
    #img = cv2.resize(img,maxsize,interpolation=cv2.INTER_LINEAR)
    #cv2.imwrite('imgs/test{}-2.jpg'.format(i), img)


    img = a.get_enclosing_image(level=6)
    ecvl.ImWrite('imgs/test{}-3a.jpg'.format(i), img)
    img = numpy.moveaxis(numpy.array(img), [0, 1, 2], [2, 1, 0])
    cv2.imwrite('imgs/test{}-3b.jpg'.format(i), img)


    img = a.get_image_cropped(level=6, rgba = True)
    #NOT IMPLEMENTED for alpha channel
    #ecvl.ImWrite('imgs/test{}-4.jpg'.format(i), img)
    img = numpy.moveaxis(numpy.array(img), [0, 1, 2], [2, 1, 0])
    cv2.imwrite('imgs/test{}-4.jpg'.format(i), img)
    img = img[:,:,3]
    cv2.imwrite('imgs/test{}-4g.jpg'.format(i), img)




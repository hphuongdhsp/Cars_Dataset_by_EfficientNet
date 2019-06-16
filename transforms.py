"""
image transform. 

"""
from PIL import Image
import random
import cv2
import numpy as np
import math

class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class OneOf:
    def __init__(self, transforms, prob=0.5):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, x):
        if random.random() < self.prob:
            t = random.choice(self.transforms)
            t.prob = 1.
            x = t(x)
        return x



class OneOrOther:
    def __init__(self, first, second, prob=0.5):
        self.first = first
        first.prob = 1.
        self.second = second
        second.prob = 1.
        self.prob = prob

    def __call__(self, x):
        if random.random() < self.prob:
            x  = self.first(x)
        else:
            x = self.second(x)
        return x


    
class Randompadding:
    def __init__(self, size=(128,128)):
        self.size=size
    def __call__(self, img):
        h_start = np.random.randint(0, self.size[0] - img.shape[0])
        w_start = np.random.randint(0, self.size[1] - img.shape[1])
    
        img = cv2.copyMakeBorder(img, h_start, self.size[0] - img.shape[0]-h_start,
                             w_start, self.size[1] - img.shape[1]-w_start,
                                      borderType=cv2.BORDER_REFLECT_101)
        return img
        
class Randompadding_V1:
    def __init__(self, size=(128,128)):
        self.size=size
    def __call__(self, img):
        h_start = np.random.randint(0, self.size[0] - img.shape[0])
        w_start = np.random.randint(0, self.size[1] - img.shape[1])
    
        img = cv2.copyMakeBorder(img, 0, 0,
                             w_start, self.size[1] - img.shape[1]-w_start,
                                      borderType=cv2.BORDER_REFLECT_101)
        img = cv2.copyMakeBorder(img, h_start, self.size[0] - img.shape[0]-h_start,
                             0, 0,
                                      borderType=cv2.BORDER_REPLICATE)

        return img

class VerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            img = cv2.flip(img, 0)

        return img


class HorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            img = cv2.flip(img, 1)

        return img


class RandomFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)

        return img


class Transpose:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            img = img.transpose(1, 0, 2)
        return img

class RandomRotate90:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)

        return img.copy()


class Rotate:
    def __init__(self, limit=10, prob=0.5):
        self.prob = prob
        self.limit = limit

    def __call__(self, img,):
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)

            height, width = img.shape[0:2]
            mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            img = cv2.warpAffine(img, mat, (height, width),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)

        return img


class RandomCrop:
    def __init__(self, size):
        self.h = size[0]
        self.w = size[1]

    def __call__(self, img):
        height, width, _ = img.shape

        h_start = np.random.randint(0, height - self.h)
        w_start = np.random.randint(0, width - self.w)

        img = img[h_start: h_start + self.h, w_start: w_start + self.w]

        assert img.shape[0] == self.h
        assert img.shape[1] == self.w


        return img




class ShiftScale:
    def __init__(self, limit=4, prob=.25):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        limit = self.limit
        if random.random() < self.prob:
            height, width, channel = img.shape
            #assert (width == height)
            size0 = width
            size1 = width + 2 * limit
            size = round(random.uniform(size0, size1))

            dx = round(random.uniform(0, size1 - size))
            dy = round(random.uniform(0, size1 - size))

            y1 = dy
            y2 = y1 + size
            x1 = dx
            x2 = x1 + size

            img1 = cv2.copyMakeBorder(img, limit, limit, limit, limit, borderType=cv2.BORDER_REFLECT_101)
            img = (img1[y1:y2, x1:x2, :] if size == size0
                   else cv2.resize(img1[y1:y2, x1:x2, :], (size0, size0), interpolation=cv2.INTER_LINEAR))


        return img

class Scale:
    def __init__(self, size=128):
        self.size=size
    def __call__(self, img):
        img=cv2.resize(img,(self.size,self.size),interpolation=cv2.INTER_LINEAR)

        return img

            
    
class ShiftScaleRotate:
    def __init__(self, shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, prob=0.5):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            height, width, channel = img.shape
            
            angle = random.uniform(-self.rotate_limit, self.rotate_limit)
            scale = random.uniform(1 - self.scale_limit, 1 + self.scale_limit)
            dx = round(random.uniform(-self.shift_limit, self.shift_limit)) * width
            dy = round(random.uniform(-self.shift_limit, self.shift_limit)) * height

            cc = math.cos(angle / 180 * math.pi) * scale
            ss = math.sin(angle / 180 * math.pi) * scale
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            img = cv2.warpPerspective(img, mat, (width, height),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)


        return img


class CenterCrop:
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)

        self.height = size[0]
        self.width = size[1]

    def __call__(self, img):
        h= img.shape[0]
        w= img.shape[1]
        dy = int((h - self.height) // 2)
        dx = int((w - self.width) // 2)

        y1 = dy
        y2 = y1 + self.height
        x1 = dx
        x2 = x1 + self.width
        img = img[y1:y2, x1:x2]


        return img


class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        max_pixel_value = 255.0

        img = img.astype(np.float32) / max_pixel_value

        img -= np.ones(img.shape) * self.mean
        img /= np.ones(img.shape) * self.std
        return img


class Distort1:
    """"
    ## unconverntional augmnet ################################################################################3
    ## https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion
    ## https://stackoverflow.com/questions/10364201/image-transformation-in-opencv
    ## https://stackoverflow.com/questions/2477774/correcting-fisheye-distortion-programmatically
    ## http://www.coldvision.io/2017/03/02/advanced-lane-finding-using-opencv/
    ## barrel\pincushion distortion
    """

    def __init__(self, distort_limit=0.35, shift_limit=0.25, prob=0.5):
        self.distort_limit = distort_limit
        self.shift_limit = shift_limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            height, width, channel = img.shape

            if 0:
                img = img.copy()
                for x in range(0, width, 10):
                    cv2.line(img, (x, 0), (x, height), (1, 1, 1), 1)
                for y in range(0, height, 10):
                    cv2.line(img, (0, y), (width, y), (1, 1, 1), 1)

            k = random.uniform(-self.distort_limit, self.distort_limit) * 0.00001
            dx = random.uniform(-self.shift_limit, self.shift_limit) * width
            dy = random.uniform(-self.shift_limit, self.shift_limit) * height

            #  map_x, map_y =
            # cv2.initUndistortRectifyMap(intrinsics, dist_coeffs, None, None, (width,height),cv2.CV_32FC1)
            # https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion
            # https://stackoverflow.com/questions/10364201/image-transformation-in-opencv
            x, y = np.mgrid[0:width:1, 0:height:1]
            x = x.astype(np.float32) - width / 2 - dx
            y = y.astype(np.float32) - height / 2 - dy
            theta = np.arctan2(y, x)
            d = (x * x + y * y) ** 0.5
            r = d * (1 + k * d * d)
            map_x = r * np.cos(theta) + width / 2 + dx
            map_y = r * np.sin(theta) + height / 2 + dy

            img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
           
        return img





def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)

class do_horizontal_shear:
    def __init__(self, limit=0.3, prob=0.5):
        self.limit = limit
        self.prob  = prob
    def __call__(self, img):
        if random.random() < self.prob:
            borderMode=cv2.BORDER_REFLECT_101
            height, width = img.shape[:2]
            dx=random.uniform(0,self.limit)*width
            
            box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ],np.float32)
            box1 = np.array([ [+dx,0], [width+dx,0],  [width-dx,height], [-dx,height], ],np.float32)

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0,box1)

            img = cv2.warpPerspective(img, mat, (width,height),flags=cv2.INTER_LINEAR,
                                borderMode=borderMode,borderValue=(0,0,0,))
            
        return img

class RandomFilter:
    """
    blur sharpen, etc
    """

    def __init__(self, limit=.5, prob=.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            alpha = self.limit * random.uniform(0, 1)
            kernel = np.ones((3, 3), np.float32) / 9 * 0.2

            colored = img[..., :3]
            colored = alpha * cv2.filter2D(colored, -1, kernel) + (1 - alpha) * colored
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[..., :3] = clip(colored, dtype, maxval)

        return img


# https://github.com/pytorch/vision/pull/27/commits/659c854c6971ecc5b94dca3f4459ef2b7e42fb70
# color augmentation

# brightness, contrast, saturation-------------
# from mxnet code, see: https://github.com/dmlc/mxnet/blob/master/python/mxnet/image.py




class RandomContrast:
    def __init__(self, limit=.1, prob=.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            alpha = 1.0 + self.limit * random.uniform(-1, 1)

            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[:, :, :3] = clip(alpha * img[:, :, :3] + gray, dtype, maxval)
        return img


class RandomSaturation:
    def __init__(self, limit=0.3, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        # dont work :(
        if random.random() < self.prob:
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            alpha = 1.0 + random.uniform(-self.limit, self.limit)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            img[..., :3] = alpha * img[..., :3] + (1.0 - alpha) * gray
            img[..., :3] = clip(img[..., :3], dtype, maxval)
        return img


class RandomHueSaturationValue:
    def __init__(self, hue_shift_limit=(-20, 20), sat_shift_limit=(-35, 35), val_shift_limit=(-35, 35), prob=0.5):
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(image)
            hue_shift = np.random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1])
            h = cv2.add(h, hue_shift)
            sat_shift = np.random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1])
            s = cv2.add(s, sat_shift)
            val_shift = np.random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])
            v = cv2.add(v, val_shift)
            image = cv2.merge((h, s, v))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image

class RandomErasing(object):
    
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return (img)

        for attempt in range(100):
            area = img.shape[0] * img.shape[1]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[0] - w)
                if img.shape[2] == 3:
                    img[ x1:x1+h, y1:y1+w,0] = self.mean[0]
                    img[ x1:x1+h, y1:y1+w,1] = self.mean[1]
                    img[ x1:x1+h, y1:y1+w,2] = self.mean[2]
                else:
                    img[ x1:x1+h, y1:y1+w] = self.mean[0]

                    
                return img


class CLAHE:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def __call__(self, im):
        img_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output

class Centerpad:
    def __init__(self, size=(224,224)):
        self.size=size
    def __call__(self, img):
        h_start = int((self.size[0]-img.shape[0])/2)
        w_start = int((self.size[1]-img.shape[1])/2)
    
        img = cv2.copyMakeBorder(img, h_start, self.size[0] - img.shape[0]-h_start,
                             w_start, self.size[1] - img.shape[1]-w_start,
                                      borderType=cv2.BORDER_REFLECT_101)
        return img

class Centerpad_V1:
    def __init__(self, size=(128,128)):
        self.size=size
    def __call__(self, img):
        h_start = int((self.size[0]-img.shape[0])/2)
        w_start = int((self.size[1]-img.shape[1])/2)
    
        img = cv2.copyMakeBorder(img, 0, 0,
                             w_start, self.size[1] - img.shape[1]-w_start,
                                      borderType=cv2.BORDER_REFLECT_101)
        img = cv2.copyMakeBorder(img, h_start, self.size[0] - img.shape[0]-h_start,
                             0, 0,
                                      borderType=cv2.BORDER_REPLICATE)
        return img

        return img
class Brightness_shift:
    def __init__(self, limit=0.1, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            alpha = self.limit * random.uniform(-1, 1)

            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[..., :3] = clip(alpha + img[..., :3], dtype, maxval)
        return img
    
class Brightness_multiply:
    def __init__(self, limit=0.1, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            alpha = 1.0 + self.limit * random.uniform(-1, 1)

            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[..., :3] = clip(alpha * img[..., :3], dtype, maxval)
        return img
    
class do_Gamma:
    def __init__(self, limit=0.1, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            alpha = 1.0 + self.limit * random.uniform(-1, 1)

            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[..., :3] = clip(img[..., :3]**(alpha), dtype, maxval)
        return img
class GaussianBlur:
    def __init__(self, ksize, prob=.5):
        self.ksize = ksize
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            img= cv2.GaussianBlur(img,(self.ksize, self.ksize),0)
        return img

class Blur:
    def __init__(self, ksize, prob=.5):
        self.ksize = ksize
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            img= cv2.blur(img, (self.ksize, self.ksize))
        return img

class Median_blur:
    def __init__(self, ksize, prob=.5):
        self.ksize = ksize
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            img= cv2.medianBlur(img, self.ksize)
        return img

class Motion_blur:
    def __init__(self, ksize, prob=.5):
        self.ksize = ksize
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            assert self.ksize > 2
            kernel = np.zeros((self.ksize, self.ksize), dtype=np.uint8)
            xs, xe = random.randint(0, self.ksize - 1), random.randint(0, self.ksize - 1)
            if xs == xe:
                ys, ye = random.sample(range(self.ksize), 2)
            else:
                ys, ye = random.randint(0, self.ksize - 1), random.randint(0, self.ksize - 1)
            cv2.line(kernel, (xs, ys), (xe, ye), 1, thickness=1)
            img= cv2.blur(img, (self.ksize, self.ksize))
        return img
class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class GaussianNoise:
    """
    The class `:class noise` is used to perfrom random noise on images passed
    to its :func:`perform_operation` function.
    """
    def __init__(self, probability, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        w, h = img.size
        c = len(img.getbands())
        
        noise = np.random.normal(self.mean, self.std, (h, w, c))

        return Image.fromarray(np.uint8(np.array(img) + noise))    

def valid_augment(img):
    return DualCompose([Scale(size=300),Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(img)
valid_transform = DualCompose([Scale(size=300),Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train_transform = DualCompose([VerticalFlip(prob=0.5),OneOf([ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20),
                                                            ShiftScale(limit=4,prob=0.5),
                                                            #do_horizontal_shear(limit=0.3)     
                                                                  ]),
                        OneOf([Brightness_shift(limit=0.1),
                               Brightness_multiply(limit=0.1),
                               do_Gamma(limit=0.1),
                               RandomContrast(limit=0.2),
                               RandomSaturation(limit=0.2)],prob=0.5),   
                        Scale(size=300),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                        ])
def train_augment(img,prob=0.5):
    return DualCompose([VerticalFlip(prob=0.5),OneOf([ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20),
                                                            ShiftScale(limit=4,prob=0.5),
                                                            #do_horizontal_shear(limit=0.3)     
                                                                  ]),
                        OneOf([Brightness_shift(limit=0.1),
                               Brightness_multiply(limit=0.1),
                               do_Gamma(limit=0.1),
                               RandomContrast(limit=0.2),
                               RandomSaturation(limit=0.2)],prob=0.5),   
                        Scale(size=300),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                        ])(img)

    





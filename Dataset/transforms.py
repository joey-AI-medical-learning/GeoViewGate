import collections.abc
import numpy as np

class Base(object):
    def sample(self, *shape):
        return shape

    def tf(self, img, k=0):
        return img

    def __call__(self, img, dim=3, reuse=False): # class -> func()
        # image: nhwtc
        # shape: no first dim
        if not reuse:
            im = img if isinstance(img, np.ndarray) else img[0]
            shape = im.shape[1:dim+1]
            self.sample(*shape)

        if isinstance(img, collections.abc.Sequence):
            return [self.tf(x, k) for k, x in enumerate(img)] # img:k=0,label:k=1

        return self.tf(img)

    def __str__(self):
        return 'Identity()'

Identity = Base

class RandomIntensityChange(Base):
    def __init__(self,factor):
        shift,scale = factor
        assert (shift >0) and (scale >0)
        self.shift = shift
        self.scale = scale

    def tf(self,img,k=0):
        if k==1:
            return img
        shift_factor = np.random.uniform(-self.shift,self.shift,size=[1,img.shape[1],1,1,img.shape[4]]) # [-0.1,+0.1]
        scale_factor = np.random.uniform(1.0 - self.scale, 1.0 + self.scale,size=[1,img.shape[1],1,1,img.shape[4]]) # [0.9,1.1)
        return img * scale_factor + shift_factor

    def __str__(self):
        return 'random intensity shift per channels on the input image, including'

class NumpyType(Base):
    def __init__(self, types, num=-1):
        self.types = types # ('float32', 'int64')
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        return img.astype(self.types[k])

    def __str__(self):
        s = ', '.join([str(s) for s in self.types])
        return 'NumpyType(({}))'.format(s)

class Compose(Base):
    def __init__(self, ops):
        if not isinstance(ops, collections.abc.Sequence):
            ops = ops,
        self.ops = ops

    def sample(self, *shape):
        for op in self.ops:
            shape = op.sample(*shape)

    def tf(self, img, k=0):
        for op in self.ops:
            img = op.tf(img, k) # do not use op(img) here

        return img

    def __str__(self):
        ops = ', '.join([str(op) for op in self.ops])
        return 'Compose([{}])'.format(ops)

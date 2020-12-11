'''skimage and opencv image format conversion'''

import cv2
import skimage as io
import numpy as np

def skimage2opencv(src):
    src *= 255
    src.astype(np.int)
    cv2.cvtColor(src,cv2.COLOR_RGB2BGR)
    return src

def opencv2skimage(src):
    cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
    src.astype(np.float32)
    src /= 255
    return src

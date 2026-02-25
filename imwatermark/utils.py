import numpy as np
from scipy.fft import dctn, idctn
from PIL import Image


def bgr2yuv(bgr):
    """Convert BGR uint8 image to YUV, matching cv2.cvtColor COLOR_BGR2YUV (BT.601)."""
    bgr_float = bgr.astype(np.float64)
    b = bgr_float[:, :, 0]
    g = bgr_float[:, :, 1]
    r = bgr_float[:, :, 2]

    yuv = np.empty_like(bgr_float)
    yuv[:, :, 0] = 0.299 * r + 0.587 * g + 0.114 * b
    yuv[:, :, 1] = -0.169 * r - 0.331 * g + 0.500 * b + 128.0
    yuv[:, :, 2] = 0.500 * r - 0.419 * g - 0.081 * b + 128.0

    return np.clip(yuv, 0, 255).astype(bgr.dtype)


def yuv2bgr(yuv):
    """Convert YUV image back to BGR, matching cv2.cvtColor COLOR_YUV2BGR."""
    yuv_float = yuv.astype(np.float64)
    y = yuv_float[:, :, 0]
    u = yuv_float[:, :, 1] - 128.0
    v = yuv_float[:, :, 2] - 128.0

    bgr = np.empty_like(yuv_float)
    bgr[:, :, 0] = y + 1.773 * u
    bgr[:, :, 1] = y - 0.344 * u - 0.714 * v
    bgr[:, :, 2] = y + 1.403 * v

    return np.clip(bgr, 0, 255).astype(yuv.dtype)


def dct2(block):
    """2D DCT matching cv2.dct() — Type-II with orthonormal normalization."""
    return dctn(block, type=2, norm='ortho')


def idct2(block):
    """2D inverse DCT matching cv2.idct()."""
    return idctn(block, type=2, norm='ortho')


def imread(path):
    """Read image as BGR numpy array (uint8), matching cv2.imread()."""
    img = Image.open(path).convert('RGB')
    rgb = np.array(img, dtype=np.uint8)
    return rgb[:, :, ::-1].copy()


def imwrite(path, bgr):
    """Write BGR numpy array to image file, matching cv2.imwrite()."""
    rgb = bgr[:, :, ::-1]
    Image.fromarray(rgb).save(path)

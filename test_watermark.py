#!/usr/bin/env python3
"""Test script to verify watermark encode/decode round-trip works without cv2."""

import numpy as np
from PIL import Image
from imwatermark import WatermarkEncoder, WatermarkDecoder


def test_method(bgr, method, wm_text):
    encoder = WatermarkEncoder()
    encoder.set_watermark('bytes', wm_text.encode('utf-8'))
    bgr_encoded = encoder.encode(bgr, method)

    decoder = WatermarkDecoder('bytes', len(wm_text) * 8)
    wm = decoder.decode(bgr_encoded, method)
    decoded = wm.decode('utf-8')

    status = 'PASS' if decoded == wm_text else 'FAIL'
    print(f'  [{status}] {method}: encoded="{wm_text}" decoded="{decoded}"')
    return decoded == wm_text


def main():
    img = Image.open('test_vectors/original.jpg').convert('RGB')
    bgr = np.array(img)[:, :, ::-1].copy()
    print(f'Image: {bgr.shape[1]}x{bgr.shape[0]}')

    wm_text = 'hello'
    results = []

    print('\nRound-trip tests:')
    results.append(test_method(bgr, 'dwtDct', wm_text))
    results.append(test_method(bgr, 'dwtDctSvd', wm_text))

    # Verify no cv2 imports remain
    import importlib
    import sys
    cv2_loaded = 'cv2' in sys.modules
    status = 'PASS' if not cv2_loaded else 'FAIL'
    print(f'\n  [{status}] No cv2 module loaded')
    results.append(not cv2_loaded)

    print(f'\nAll tests passed: {all(results)}')
    return 0 if all(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())

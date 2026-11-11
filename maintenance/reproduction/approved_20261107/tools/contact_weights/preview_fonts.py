"""Portable Chinese preview font selection; no font download is required."""
from __future__ import annotations
import os
from pathlib import Path
from PIL import ImageFont


def preview_font(size):
    candidates = [os.environ.get('PLANT_PREVIEW_FONT', ''),
        r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc']
    for name in candidates:
        if name and Path(name).is_file():
            try:
                return ImageFont.truetype(name, size), True
            except OSError:
                pass
    for name in ('DejaVuSans.ttf', 'LiberationSans-Regular.ttf'):
        try:
            return ImageFont.truetype(name, size), False
        except OSError:
            pass
    return ImageFont.load_default(), False

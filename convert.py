#!/usr/bin/env python3

import math

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

import scipy.misc

from cc_pathlib import Path

template_svg = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="512" height="512" viewBox="0 0 512 512" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:svg="http://www.w3.org/2000/svg">
	<rect x="0" y="0" height="512" width="512" style="fill:#ffffff" />
	<g style="opacity:0">
		<rect x="0" y="0" height="512" width="512" style="fill:#ffdddd" />
		<g style="fill:none;stroke:#ff0000;stroke-width:1;stroke-linecap:butt;stroke-linejoin:miter">
		<path	d="M 96,0 V 512" />
		<path d="M 0,384 H 512" />
		</g>
	</g>
	<text alignement-baseline="baseline" text-anchor="start" style="font-style:normal;font-weight:normal;font-size:384px;font-family:sans-serif;fill:#000000;stroke:none" x="96" y="384">{c}</text>
</svg>"""

def compute_sdf(img, radius=8) :
	w, h = img.shape
	sdf = 1.0 - img
	img = np.hstack((np.ones((h, radius)), img, np.ones((h, radius))))
	img = np.vstack((np.ones((radius, 2*radius+w)), img, np.ones((radius, 2*radius+w))))
	x, y = np.mgrid[-radius:radius+1, -radius:radius+1]
	knl = np.sqrt(x**2 + y**2)

	inside = np.where(img < 0.5, True, False)
	for r in range(h) :
		for c in range(w) :
			if inside[r+radius, c+radius] :
				ext = inside[r:r+2*radius+1, c:c+2*radius+1]
				msk = ma.array(knl, mask=ext)
				if not msk.mask.all() :
					sdf[r, c] = (np.clip(msk.min(), -radius, radius) / (2*radius)) + 0.5

	outside = np.where(0.5 <= img, True, False)
	for r in range(h) :
		for c in range(w) :
			if outside[r+radius, c+radius] :
				ext = outside[r:r+2*radius+1, c:c+2*radius+1]
				msk = ma.array(knl, mask=ext)
				if not msk.mask.all() :
					sdf[r, c] = 0.5 - (np.clip(msk.min(), -radius, radius) / (2*radius))
	return sdf

svg_dir = Path("svg").make_dirs()
png_dir = Path("png").make_dirs()
sdf_dir = Path("sdf").make_dirs()

for i in range(128) :
	c = chr(i)
	if c.isprintable() and c.strip() :
		if not (svg_dir / f"{i:02X}.svg").is_file() :
			(svg_dir / f"{i:02X}.svg").write_text(template_svg.format(c=c))
		if not (png_dir / f"{i:02X}.png").is_file() :
			Path().run("inkscape", "--export-dpi=96", "--export-type=png", "-o", (png_dir / f"{i:02X}.png"), (svg_dir / f"{i:02X}.svg"))
		if not (sdf_dir / f"{i:02X}.png").is_file() :
			img = plt.imread(png_dir / f"{i:02X}.png")[:,:,0]
			print(img.shape, img.dtype, c, hex(i))
			sdf = compute_sdf(img)
			plt.imsave(sdf_dir / f"{i:02X}.png", sdf, cmap='binary')
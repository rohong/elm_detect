#!/usr/bin/env python
# --*-- coding: utf-8 --*--
"""
Detect ELM burst in filter-scope data.

Created on April 18, 2018
Author: Rongjie Hong
"""
from __future__ import division
import MDSplus as mds
import numpy as np
import matplotlib.pyplot as plt
import detect as det
from functools import partial

# %%
conn = mds.Connection('atlas')
shot = 176472
conn.openTree('spectroscopy', shot)
fs = np.asarray(conn.get(r'\fssasda'))
t_fs = np.asarray(conn.get(r'dim_of(\fssasda)'))
# %%
t0 = 4.5e3
ind = (t_fs > t0) & (t_fs < t0 + 50)
y, t = fs[ind], t_fs[ind]
# %%
R, maxes = det.detect(y, partial(det.constant_hazard, 250),
                      det.StudentTest(0.1, 1., 1., 1.))
# %%
Nw = 10
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(4, 5))
ax[0].plot(t, y, lw=0.5)
ax[1].plot(t[Nw-1:-1], R[Nw, Nw:-1], '.--', lw=0.5)
plt.tight_layout()
plt.show()
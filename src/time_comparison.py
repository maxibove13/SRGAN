#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to plot a time comparison between caffa3d and SRGAN"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "09/22"

import os
import time

import numpy as np
from matplotlib import pyplot as plt


fig, ax = plt.subplots(1,1)

N = np.arange(1,1000, 50)

srgan = (10800 + 900 + 0.02*N)/3600
caffa = (14200 + 27*N)/3600


ax.plot(srgan, N, c='C2', marker='.', ls='none',label='SRGAN')
ax.plot(caffa, N, c='C3', marker='.', ls='none',label='CHAMAN-LES')

ax.set_xlabel('$t$ [h]')
ax.set_ylabel('$N$')

ax.grid()

ax.legend()

fig.savefig(
    os.path.join('figures', 'time_comparison.png'),
    dpi=600,
    bbox_inches='tight'
    )
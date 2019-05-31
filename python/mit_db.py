#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-05-31 15:21
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : mit_db.py
# @Software: PyCharm
# @Description

"""
mit_db.py

Description:
Contains the classes for store the MITBIH database and some utils

VARPA, University of Coruna
Mondejar Guerra, Victor M.
24 Oct 2017
"""

import matplotlib.pyplot as plt
import numpy as np


# Show a 2D plot with the data in beat
def display_signal(beat):
    plt.plot(beat)
    plt.ylabel('Signal')
    plt.show()


# Class for RR intervals features
class RRIntervals:
    def __init__(self):
        # Instance atributes
        self.pre_R = np.array([])
        self.post_R = np.array([])
        self.local_R = np.array([])
        self.global_R = np.array([])


class MitDb:
    def __init__(self):
        # Instance atributes
        self.filename = []
        self.raw_signal = []
        self.beat = np.empty([])  # record, beat, lead
        self.class_ID = []
        self.valid_R = []
        self.R_pos = []
        self.orig_R_pos = []
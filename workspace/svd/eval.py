#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in 11-03-2022 at 16:25
@author: J. Sha
"""

import sys
from os.path import dirname

sys.path.append(dirname(__file__))

from help_funs import mu


def eval(vertex, farthest_neighbour):
    normal, normal_img = mu.vertex2normal(vertex, farthest_neighbour)
    return normal, normal_img, None, None


if __name__ == '__main__':
    pass

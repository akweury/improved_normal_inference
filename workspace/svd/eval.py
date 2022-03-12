#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in 11-03-2022 at 16:25
@author: J. Sha
"""

import sys
from os.path import dirname

sys.path.append(dirname(__file__))

from help_funs import mu, chart


def eval(vertex, farthest_neighbour):
    return mu.vertex2normal(vertex, farthest_neighbour)


if __name__ == '__main__':
    pass

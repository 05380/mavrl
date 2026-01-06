
import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

setup(
    name='mav_baselines',
    version='0.0.1',
    author='Hang Yu',
    author_email='h.y.yu@tudelft.nl',
    description='A simulator for reinforcement learning',
    long_description='',
    packages=['mav_baselines'],
)

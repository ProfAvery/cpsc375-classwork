# To import libraries and datasets into the global namespace:
#
#   from cpsc375 import *
#
# In Google Colab, first download with
#
#   !mkdir -p ./cpsc375 && wget --quiet --timestamping \
#       https://raw.githubusercontent.com/ProfAvery/cpsc375-classwork/refs/heads/main/cpsc375/__init__.py \
#       --output-document=./cpsc375/__init__.py
#

import io
import sys
import subprocess
import importlib
import importlib.util
import pprint
import re

import nltk

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import urllib

from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud

if importlib.util.find_spec("cairosvg") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cairosvg"])

try:
    from mockr import run_stream_job, run_pandas_job
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mockr"])
    from mockr import run_stream_job, run_pandas_job

try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ucimlrepo"])
    from ucimlrepo import fetch_ucirepo

try:
    import lets_plot
    from lets_plot import *
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lets-plot"])
    import lets_plot
    from lets_plot import *


def _repr_svg_(self):
    from io import BytesIO
    from sys import stdout

    file_like = BytesIO()
    self.to_svg(file_like)
    return file_like.getvalue().decode(stdout.encoding)


LetsPlot.setup_html()

# Patch for use with colab2pdf
lets_plot.plot.core.PlotSpec._repr_svg_ = _repr_svg_

billboard_dataset = sm.datasets.get_rdataset("billboard", "tidyr")
billboard = billboard_dataset.data

iris_dataset = sm.datasets.get_rdataset("iris")
iris = iris_dataset.data

airquality_dataset = sm.datasets.get_rdataset("airquality")
airquality = airquality_dataset.data

mpg_dataset = sm.datasets.get_rdataset("mpg", "ggplot2")
mpg = mpg_dataset.data

nycflights13_dataset = sm.datasets.get_rdataset("flights", "nycflights13")
nycflights13 = nycflights13_dataset.data

auto_mpg = fetch_ucirepo(id=9)
auto_mpg_dataset = type(
    "auto_mpg_dataset", (), {"__doc__": pprint.pformat(auto_mpg.metadata)}
)

datasets = [
    "billboard_dataset",
    "billboard",
    "iris_dataset",
    "iris",
    "airquality_dataset",
    "airquality",
    "mpg_dataset",
    "mpg",
    "nycflights13_dataset",
    "nycflights13",
    "auto_mpg",
    "auto_mpg_dataset",
]

__all__ = (
    [
        "distance",
        "fetch_ucirepo",
        "KMeans",
        "KNeighborsClassifier",
        "LinearRegression",
        "MinMaxScaler",
        "nltk",
        "np",
        "pd",
        "Pipeline",
        "plt",
        "re",
        "run_pandas_job",
        "run_stream_job",
        "sm",
        "sns",
        "train_test_split",
        "urllib",
        "WordCloud",
    ]
    + lets_plot.__all__
    + datasets
)

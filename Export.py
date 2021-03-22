!pip install fastai --upgrade && pip install  kornia pydicom
!pip install nbdev

from fastai.basics import *
from fastai.callback.all import *
from fastai.vision.all import *
from fastai.medical.imaging import *

import pydicom

import pandas as pd
import numpy as np

def get_x(x):
  return x[3]

def get_y(x):
  return x[2]

pneumothorax = DataBlock(blocks=(ImageBlock(cls=PILDicom), CategoryBlock),
                   get_x=get_x,
                   get_y=get_y,
                   batch_tfms=aug_transforms(size=224))
                   
path = Path()
learn_inf = load_learner(path/"...")

learn_inf.predict("test_path")

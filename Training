#Training réaliser sur google colab

!pip install fastai --upgrade && pip install pydicom kornia
!pip install nbdev

from fastai.basics import *
from fastai.callback.all import *
from fastai.vision.all import *
from fastai.medical.imaging import *

import pydicom

import pandas as pd
import numpy as np

#Rework du CSV file et du labelling :

df = pd.read_csv('path_to_csv')

for x in df.index:
  if len(df.loc[x, ' EncodedPixels']) > 3:
    df.loc[x, ' EncodedPixels'] = 'Pneumothorax'
  
  else :
    df.loc[x, ' EncodedPixels'] = 'No Pneumothorax'

df_new2 = df.rename(columns={' EncodedPixels': 'Labels'})
df_new2.to_csv('train-rle-NEW2.csv')

#Data path
items = get_dicom_files("path_to_dir_train")
items[0]

#ajout des data path au csv
imageid2path = {}
for item in items:
    imageid2path['.'.join(str(item).split('/')[-1].split('.')[:-1])] = item
    
df_new2['path'] = df_new2['ImageId'].map(imageid2path)
df_new2.head()

#Datablock et dataloader
trn,val = RandomSplitter()(items)
def get_x(x):
  return x[3]

def get_y(x):
  return x[2]

pneumothorax = DataBlock(blocks=(ImageBlock(cls=PILDicom), CategoryBlock),
                   get_x=get_x,
                   get_y=get_y,
                   batch_tfms=aug_transforms(size=224))
 
dls = pneumothorax.dataloaders(df_new2.values)
dls.show_batch(max_n=16)

#Compile
learn = cnn_learner(dls, resnet34, metrics=accuracy)
learn.loss_func
learn.opt_func
learn.lr_find()

learn.fit(50)

#results and save
learn.show_results(max_n=16)
purge=False
learn.export(fname='CNNe50_PROJET_PNEUMOTHORAX_FASTAI')

#Evaluation des résultats
interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(dls.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(7,7))

interp.plot_top_losses(2, nrows=2)

upp, low = interp.confusion_matrix()
tn, fp = upp[0], upp[1]
fn, tp = low[0], low[1]
print(tn, fp, fn, tp)

sensitivity = tp/(tp + fn)
sensitivity

specificity = tn/(fp + tn)
specificity

ppv = tp/(tp+fp)
ppv

npv = tn/(tn+fn)
npv


from __future__ import print_function, division, absolute_import
from AudioDataGenerator import BalancedAudioDataGenerator
import os
from collections import Counter
import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.random.set_seed(1)
from datetime import datetime
import argparse
from keras.callbacks import TensorBoard, Callback, ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger
import seaborn as sns
sns.set()
from modules import heartnetTop
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam as optimizer
from keras.layers import Dense,Flatten,Dropout
from keras.initializers import he_normal as initializer
from utils import load_data, sessionLog, log_metrics
from utils import plot_coeff, get_weights, load_model, predict_parts, calc_metrics, plotRoc
from utils import McnemerStats, grad_cam_logs, plot_confidence_logs, cc2parts
from utils import idx_parts2cc, log_fusion, plot_freq, plot_metric, get_activations
from utils import model_confidence, cc2rec, parts2cc
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import pandas as pd

#--- Plot Co-efficients

ax = plot_coeff([
            "potes_fold0_noFIR 2019-03-02 13_01_33.636778",
            "fold0_noFIR 2019-03-07 14_44_47.022240"],min_epoch=80)
plt.show()

#--- Load model and Data

fold_dir = '../data/feature/folds/'
foldname = 'fold_0'
# x_train, y_train, train_files,train_parts, q_train, \
#     x_val, y_val,val_files,val_parts, q_val = load_data(foldname,fold_dir,quality=True) # also return recording quality

# train_parts = train_parts[np.nonzero(train_parts)] ## Some have zero cardiac cycle
# val_parts = val_parts[np.nonzero(val_parts)]

# x_test = x_val
# y_test = y_val
# test_parts = val_parts
# test_files = val_files

x_train, y_train, train_files,train_parts, q_train, \
    x_val, y_val,val_files,val_parts, q_val = load_data(foldname,fold_dir,quality=True)
    
test_parts = train_parts[0][np.asarray(train_files) =='x']
test_parts = np.concatenate([test_parts,val_parts[np.asarray(val_files)=='x']],axis=0)
train_files = parts2cc(train_files,train_parts[0])
val_files = parts2cc(val_files,val_parts)
x_test = x_train[train_files == 'x']
x_test = np.concatenate([x_test,x_val[val_files=='x']])
y_test = y_train[train_files == 'x']
y_test = np.concatenate([y_test,y_val[val_files=='x']])
test_files = np.concatenate([train_files[train_files == 'x'],
                            val_files[val_files == 'x']])
q_test = np.concatenate([q_train[train_files == 'x'],
                            q_val[val_files == 'x']])
del x_train, y_train, train_files,train_parts, q_train, \
    x_val, y_val,val_files,val_parts, q_val

#--- Select Model

## Heartnet
log_name = "fold0_noFIR 2019-03-07 14_44_47.022240" # Type2 macc 80 epoch

## Potes 
# log_name = "potes_fold0_noFIR 2019-03-02 13_01_33.636778"

model = load_model(log_name,verbose=1)
weights = get_weights(log_name,min_epoch=80,min_metric=0.7)

#--- Load model weights

metric = 'val_macc'
model_dir = '../models/'

checkpoint_name = os.path.join(model_dir+log_name,weights[metric])
model.load_weights(checkpoint_name)
print("Checkpoint loaded:\n %s" % checkpoint_name)


#--- Model.predict

# print('Calculating metrics for Training set')
# pred,true,files = predict_parts(model,x_train,y_train,train_parts,train_files)
# res = calc_metrics(true,pred,files)
# print(res)


# print('Calculating metrics for all of Validation')
# pred,true,files = predict_parts(model,x_val,y_val,val_parts,val_files,soft=True)
# res = calc_metrics(true,pred,files)
# print(res)

print('\n\nCalculating metrics for test')
pred,true,files = predict_parts(model,x_test,y_test,test_parts,test_files,soft=True)
res = calc_metrics(true,pred,files)
print(res)

###########################   added by rakib   #########################
## Keeping the results of the validation set
## Run with a potes model, and with a proposed model
pred_potes = None
pred_proposed = None
if("potes" in log_name):
    pred_potes = pred
else:
    pred_proposed = pred
    
##################################################################
    
# print('\n\nCalculating metrics for good quality only')
# pred,true,files = predict_parts(model,
#                                 x_val[q_val>0],y_val[q_val>0],
#                                 val_parts[cc2parts(q_val,val_parts)>0],
#                                 np.asarray(val_files)[q_val>0])
# res = calc_metrics(true,pred,files)
# print(res)

##################   Not running the test data  ###############

# print('\n\nCalculating metrics for test')
# pred,true,files = predict_parts(model,x_test,y_test,test_parts,test_files,soft=True)
# res = calc_metrics(true,pred,files)
# print(res)
# pred_fir4 = pred
# pred_proposed=pred

#--- PLot ROC

fig, ax = plt.subplots(figsize=(12,8))
if(pred_proposed is not None):
    plotRoc(true,pred_proposed,ax=ax,label='Type 2 tConv: AUC 0.83')
if(pred_potes is not None):
    plotRoc(true,pred_potes,ax=ax,label='Potes-CNN: AUC 0.499',control=False)
# plotRoc(true,pred_fir4,ax=ax,label='TypeIV tConv-CNN: AUC 0.864',control=False)

fig.savefig('roc.eps')

#--- McNemer Test

if(pred_potes is None):
    print("Please run the model with the potes Model")
if(pred_proposed is None):
    print("Please run the model with the hearnet type2 Model")
if(pred_potes is not None and pred_proposed is not None):
    McnemerStats(true,pred_proposed,pred_potes)

#--- Gabor vs Type 2

gabor = pd.read_csv('gabor_result.csv')
val_wav = pd.read_csv('val_file_names.txt',header=None)
val_wav = [x[0] for x in val_wav.values]
gtrue =[(gabor.loc[gabor['filenames']=='train_'+x[:-4],'true'].iloc[0]) for x in val_wav]
gpred =[(gabor.loc[gabor['filenames']=='train_'+x[:-4],'pred'].iloc[0]) for x in val_wav]

McnemerStats(true,pred_proposed,gpred,lab1='heartnet',lab2='gabor')

fig, ax = plt.subplots(figsize=(12,8))
plotRoc(true,pred_proposed,ax=ax,label='GammaTone tConv: AUC 0.798')
plotRoc(true,gpred,ax=ax,label="Gabor's algorithm: AUC 0.449",control=False)
# plotRoc(true,pred_fir4,ax=ax,label='TypeIV tConv-CNN: AUC 0.864',control=False)
fig.savefig('gabor_vs_gammtonetconv.png')

import sklearn
sklearn.metrics.auc(true,gpred)

#EER
# pred = model.predict(x_train)
# pred = cc2parts(pred,train_parts)[:,1]
# true = cc2parts(y_train,train_parts)[:,1]
# files = cc2parts(train_files,train_parts)
# res = calc_metrics(true,pred,files,thresh='EER')
# print(res)


# pred = model.predict(x_val)
# pred = cc2parts(pred,val_parts)[:,1]
# true = cc2parts(y_val,val_parts)[:,1]
# files = cc2parts(val_files,val_parts)
# res = calc_metrics(true,pred,files,thresh='EER')
# print(res)


pred = model.predict(x_test)
pred = cc2parts(pred,test_parts)[:,1]
true = cc2parts(y_test,test_parts)[:,1]
files = cc2parts(test_files,test_parts)
res = calc_metrics(true,pred,files,thresh='EER')
print(res)


res = calc_metrics(true,np.random.rand(len(true)),thresh='EER')

from sklearn.metrics import roc_curve
# from sklearn.metrics import precision_recall_curve

preds = model.predict(x_val)
preds = cc2parts(preds[:,1],val_parts)
true = cc2parts(y_val[:,1],val_parts)
fpr,tpr,thresh = roc_curve(true,preds)
plt.figure()
plt.plot(fpr,tpr)
diff = abs(tpr-(1-fpr))
preds = preds > thresh[np.where(diff == min(diff))[0]]
print(thresh[np.where(diff == min(diff))[0]])

calc_metrics(true,preds,cc2parts(val_files,val_parts))

preds = model.predict(x_test)
preds = cc2parts(preds[:,1],test_parts)
true = cc2parts(y_test[:,1],test_parts)
fpr,tpr,thresh = roc_curve(true,preds)
plt.figure()
plt.plot(fpr,tpr)
diff = abs(tpr-(1-fpr))
preds = preds > thresh[np.where(diff == min(diff))[0]]
print(thresh[np.where(diff == min(diff))[0]])

calc_metrics(true,preds,cc2parts(test_files,test_parts))

#--- Balanced Test
rus = RandomUnderSampler(random_state=1)
_,y,partidx = rus.fit_resample(np.expand_dims(range(len(test_parts)),axis=-1),cc2parts(y_test[:,1],test_parts))
ccidx= idx_parts2cc(partidx,test_parts)
_parts = test_parts[partidx]
x = x_test[ccidx]
y = y_test[ccidx]
_files = test_files[ccidx]

pred,true,files = predict_parts(model,x,y,_parts,_files,soft=True)
res = calc_metrics(true,pred,files,thresh='EER')
print(res)

#--- Weight Fusion predict

print('Fusion Predict Val')
model_dir = '../models/'
# fusion_weights = [.8,1.2,.8,1.2]
fusion_weights = [1,1,.4,1]

pred = np.zeros((x_val.shape[0],2))
for metric,weight in zip(weights.keys(),fusion_weights):
    checkpoint_name = os.path.join(model_dir+log_name,weights[metric])
    model.load_weights(checkpoint_name)
    pred += model.predict(x_val,verbose=1)*weight
pred /= sum(fusion_weights)
# pred = np.argmax(pred,axis=-1)
pred = pred[:,1]
res = calc_metrics(cc2parts(np.argmax(y_val,axis=-1),val_parts),np.round(cc2parts(pred,val_parts)))
print(res)

print('\n\nFusion Predict Test')
pred = np.zeros((x_test.shape[0],2))
for metric,weight in zip(weights.keys(),fusion_weights):
    checkpoint_name = os.path.join(model_dir+log_name,weights[metric])
    model.load_weights(checkpoint_name)
    
    pred += model.predict(x_test,verbose=1)*weight
pred /= sum(fusion_weights)
# pred = np.argmax(pred,axis=-1)
pred = pred[:,1]
res = calc_metrics(cc2parts(np.argmax(y_test,axis=-1),test_parts),np.round(cc2parts(pred,test_parts)))
print(res)

#--- Fold model fusion predict

logs=[
(0,"fold0_noFIR 2019-02-24 18:02:57.053839",'val_macc',100,.7), # Type1 macc
(1,"fold0_noFIR 2019-03-09 01:34:03.547265",'val_macc',100,.7), #gamma stage 1
(0,"fold0_noFIR 2019-03-07 14:44:47.022240",'val_macc',80,.7), # Type2 macc 80 epoch
(0,"fold0_noFIR 2019-03-08 03:28:46.740442",'val_sensitivity',100,.65), # Type3 sensitivity/spec for balanced
(0,"fold0_noFIR 2019-03-08 14:50:52.332924",'val_acc',100,.7), # type4 val_acc
(0,"fold0_noFIR 2019-03-06 21:42:10.719836",'val_macc',100,.7), #zero stage2
]
pred_fusion=0
for weight,log,metric,epoch,min_metric in logs:
    if not weight:
        continue
    model=load_model(log_name=log)
    model_weights = get_weights(log_name=log,
                                min_epoch=epoch,
                                min_metric=min_metric)
    model_dir = '../models/'
    checkpoint_name = os.path.join(model_dir+log,model_weights[metric])
    model.load_weights(checkpoint_name)
    pred = model.predict(x_test)
    pred = cc2parts(pred,test_parts)
    pred_fusion += weight*pred

pred_fusion /= sum([each[0] for each in logs])
# pred_fusion = cc2parts(pred_fusion,test_parts)
print(calc_metrics(true=cc2parts(y_test,test_parts)[:,1],pred=pred_fusion[:,1],verbose=True))

logs = [
    "fold0_noFIR 2019-02-24 18:02:57.053839", #Type1
#     "fold1_noFIR 2019-02-23 17:59:17.240365"
 
           ]
pred = log_fusion(logs,x_test,y_test,min_metric=.7,
                  metric='val_specificity',verbose=0)
pred = pred[:,1]
res = calc_metrics(cc2parts(np.argmax(y_test,axis=-1),test_parts),
                   np.round(cc2parts(pred,test_parts)))
print(res.items())

#--- Analysis

logs=[
"potes_fold0_noFIR 2019-03-02 13:01:33.636778", # potes
"fold0_noFIR 2019-02-24 18:02:57.053839", # Type1 macc
"fold0_noFIR 2019-03-07 14:44:47.022240", # Type2 macc 80 epoch
"fold0_noFIR 2019-03-08 03:28:46.740442", # Type3 sensitivity
"fold0_noFIR 2019-03-08 14:50:52.332924", # type4 val_acc
"fold0_noFIR 2019-03-09 01:34:03.547265", # gamma stage 1
"fold0_noFIR 2019-03-06 21:42:10.719836", # zero stage2
]
lognames=[
"Static FIR",
"Type I tConv",
"Type II tConv",
"Type III tConv",
"Type IV tConv",
"Gammatone tConv",
"Zero Phase tConv",
]
branchnames=[
'Branch 1',
'Branch 2',
'Branch 3',
'Branch 4',
]
ax = plot_freq(logs=logs,min_epoch=100,metric='val_macc',min_metric=.6,figsize=(17,7),phase=True)
# ax[3,0].set_ylim([-6,6])
# ax[3,2].set_xlim([0,59])
# ax[3,4].set_xlim([0,59])
# for axes,branch in zip(ax[:,0],branchnames):
#     axes.set_ylabel('%s Gain' % branch)
for axes,log in zip(ax[3,:],lognames):
    axes.set_xlabel('%s Weights' % log)
# plt.subplots_adjust(left=0.035,bottom=0.065)
# # plt.savefig('coeffs.eps')
# # plt.savefig('coeffs.png')

# for axes,log in zip(ax[3,:],lognames):
#     axes.set_xlabel('%s Weights' % log)
# plt.subplots_adjust(left=0.035,bottom=0.065)
plt.savefig('coeffsFreq.eps')

logs=[
"potes_fold0_noFIR 2019-03-16 18:44:45.597226", # potes non balanced
"potes_fold0_noFIR 2019-03-02 13:01:33.636778", # potes
"fold0_noFIR 2019-02-27 19:52:21.543329", # Type1 macc
"fold0_noFIR 2019-03-07 14:44:47.022240", # Type2 macc 80 epoch
"fold0_noFIR 2019-03-08 03:28:46.740442", # Type3 sensitivity
"fold0_noFIR 2019-03-08 14:50:52.332924", # type4 val_acc
"fold0_noFIR 2019-03-09 01:34:03.547265", # gamma stage 1
"fold0_noFIR 2019-03-06 14:21:29.823568", # zero stage2
]
lognames=[
"Potes-CNN",
"Potes-CNN DBT",
"Type I tConv",
"Type II tConv",
"Type III tConv",
"Type IV tConv",
"Gammatone tConv",
"Zero Phase tConv",
]
colors = [
'#434B77',
'#669966',
'#c10061',
'#ff51a5',
'k',
'#ffbe4f',
#'#008080',
'#DBBBBB',
'#008080',
         ]
plot_metric(logs,lognames=lognames,smoothing=0.5,metric='val_loss',colors=colors,figsize=(10,8.5))
plt.ylabel('Validation Loss per Cardiac Cycle')
plt.ylim([0.44,0.65])
# plt.savefig('validationLoss.eps')


metrics=[
#     'acc_a',
    'acc_b',
#     'acc_c',
#     'acc_d',
    'acc_e'
]
labels=[
    'subset-a',
#     'subset-b',
#     'subset-c',
#     'subset-d',
    'subset-e'
]
ax = plot_metric([logs[0],logs[2]],metrics,smoothing=0.7,legendLoc=0,ylim=[.4,1.01])

ax.set_ylabel('Subset-wise Validation Accuracy')
ax.legend(['Subset-a w/o DBT','Subset-e w/o DBT','Subset-a w/ DBT','Subset-e w/ DBT'],loc=0)


#--- Get Activations and TSNE
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA, TruncatedSVD

recBins = [117,385,7,27,1867,80,116,292,104,24,28,151,34,566]

meta_labels = np.asarray([ord(each)- 97 for each in train_files+val_files+list(test_files)])
meta_labels[meta_labels == 23] = 6
y = np.argmax(np.concatenate([y_train,y_val,y_test]),axis=-1)

for idx,each in enumerate(np.unique(meta_labels)):
        indices = np.where(np.logical_and(y==1,meta_labels == each))
        meta_labels[indices] = 7 +idx

activations = np.array(get_activations(model,np.concatenate([x_train,x_val,x_test],axis=0),
                                       batch_size=64,layer_name='flatten_1'))
if activations.ndim > 2:
    activations = np.reshape(activations,(len(activations),-1))
activations.shape

meta_labels=meta_labels[0:len(activations)]
### quality_labels=np.concatenate([q_train,q_val,q_test],axis=0)[0:len(activations)]

idx = []
for subset,each in zip(np.unique(meta_labels),recBins):
    np.random.seed(1)
    idx = idx+list(np.random.choice(np.where([meta_labels==subset])[1],size=(each,),replace=False))

# rus = RandomUnderSampler(random_state=1,return_indices=True)
# x,y,idx = rus.fit_resample(activations[quality_labels>0],meta_labels[quality_labels>0])
# np.random.seed(1)
# idx = np.random.choice(range(len(meta_labels)),size=(3792,),replace=False)
x = activations[idx]
y = meta_labels[idx]
X_embed = scale(x)

# X_embedded = PCA(n_components=50).fit_transform(X_embed)

X_embedded = TSNE(n_components=2,
#                   learning_rate=60,
#                   early_exaggeration=1140.,
                  perplexity=480, #480-2, 150-3 without exagg and lr
                  init='random',
                  n_iter=4000,
                  verbose=1,
                  ).fit_transform(X_embed)
X_embedded.shape

sns.set_style('whitegrid')
import matplotlib.font_manager as font_manager
font_prop = font_manager.FontProperties(size=14)
font_title = font_manager.FontProperties(size=20)

colors = ['#434B77',
          '#669966',
          '#c10061',
          '#ff51a5',
          'k',
          '#ffbe4f',
#           '#008080',
          '#DBEEEE',
          '#008080',
         ]
# y_ = y_>6
subsets = ["Eko CORE Bluetooth",
"Welch Allyn Meditron",
"3M Littmann E4000",
"AUDIOSCOPE",
"Infral Corp. Prototype",
"MLT201/Piezo",
"JABES",
"3M Littmann"]
parser = dict(zip(np.unique(y_val),subsets))
fig = plt.figure(figsize=(11,8))
for stage,color in zip(np.unique(y_val),colors):
    mask = y_val == stage
    plt.scatter(X_embedded[mask,0],X_embedded[mask,1],c=color,label=parser[stage],s=30)
plt.legend(markerscale=2,fontsize=14)
fig.set_tight_layout(tight=1)
plt.xlabel('TSNE Component 1',fontproperties=font_prop)
plt.ylabel('TSNE Component 2',fontproperties=font_prop)
plt.show()

# plt.savefig('potesTSNE.eps')
plt.savefig('gammaTSNEbal.eps')

meta_labels = np.asarray([ord(each)- 97 for each in train_files+val_files+list(test_files)])
meta_labels[meta_labels == 23] = 6
y = np.argmax(np.concatenate([y_train,y_val,y_test]),axis=-1)

for idx,each in enumerate(np.unique(meta_labels)):
        indices = np.where(np.logical_and(y==1,meta_labels == each))
        meta_labels[indices] = 7 +idx
meta_labels=meta_labels[0:len(activations)]
quality_labels=np.concatenate([q_train,q_val,q_test],axis=0)[0:len(activations)]

idx = []
for subset,each in zip(np.unique(meta_labels),recBins):
    np.random.seed(1)
    idx = idx+list(np.random.choice(np.where([meta_labels==subset])[1],size=(each,),replace=False))
    
y_= meta_labels[idx]
y_[y_==11] = 14
y_[y_>6] = y_[y_>6] - 7 # 0-7 steth labels
y_ = y_+1
y_[y_==7] = 0
y_[y_==8] = 7
print(np.unique(y_),np.bincount(y_))

#--- Recording Level TSNE
files = np.asarray(train_files+val_files+list(test_files))
parts = np.asarray(list(train_parts)+list(val_parts)+list(test_parts))

data = np.concatenate([x_train,x_val,x_test],axis=0)

activations = np.array(get_activations(model,data[:-13],
                                       batch_size=64,layer_name='flatten_1'))
rem = np.array(get_activations(model,data[-13:],
                                       batch_size=1,layer_name='flatten_1'))
activations = np.concatenate([activations,rem],axis=0)

del data, rem

if activations.ndim > 2:
    activations = np.reshape(activations,(len(activations),-1))
activations.shape

files = cc2parts(files,parts)
activations = cc2parts(activations,parts)
activations.shape

meta_labels = np.asarray([ord(each)- 97 for each in files])
meta_labels[meta_labels == 23] = 6
y = cc2parts(np.argmax(np.concatenate([y_train,y_val,y_test]),axis=-1),parts)

for idx,each in enumerate(np.unique(meta_labels)):
        indices = np.where(np.logical_and(y==1,meta_labels == each))
        meta_labels[indices] = 7 +idx
np.unique(meta_labels)

y_= meta_labels
y_[y_==11] = 14
y_[y_>6] = y_[y_>6] - 7 # 0-7 steth labels
y_ = y_+1
y_[y_==7] = 0
y_[y_==8] = 7
print(np.unique(y_),np.bincount(y_))

from scipy.io import savemat
savemat('typeII.mat',{'X':activations,'y':y_})

X_embed = scale(activations)
# X_embedded = PCA(n_components=50).fit_transform(X_embed)
X_embedded = TSNE(n_components=2,
#                   learning_rate=60,
#                   early_exaggeration=1140.,
                  perplexity=480, #480-2, 150-3 without exagg and lr
                  init='random',
                  n_iter=4000,
                  verbose=1,
                  ).fit_transform(X_embed)
X_embedded.shape

sns.set_style('whitegrid')
import matplotlib.font_manager as font_manager
font_prop = font_manager.FontProperties(size=14)
font_title = font_manager.FontProperties(size=20)

colors = ['#434B77',
          '#669966',
          '#c10061',
          '#ff51a5',
          'k',
          '#ffbe4f',
#           '#008080',
          '#DBEEEE',
          '#008080',
         ]
# y_ = y_>6
subsets = ["Eko CORE Bluetooth",
"Welch Allyn Meditron",
"3M Littmann E4000",
"AUDIOSCOPE",
"Infral Corp. Prototype",
"MLT201/Piezo",
"JABES",
"3M Littmann"]
parser = dict(zip(np.unique(y_),subsets))
fig = plt.figure(figsize=(11,7))
for stage,color in zip(np.unique(y_),colors):
    mask = y_ == stage
    plt.scatter(X_embedded[mask,0],X_embedded[mask,1],c=color,label=parser[stage])
plt.legend(markerscale=2,fontsize=14)
fig.set_tight_layout(tight=1)
plt.xlabel('TSNE Component 1',fontproperties=font_prop)
plt.ylabel('TSNE Component 2',fontproperties=font_prop)
plt.show()

# plt.savefig('rec_gammatoneTSNE.eps')

plt.savefig('rec_typeIITSNE.eps')

fig = plt.figure(figsize=(7,5))
conf = model_confidence(model,x_val,y_val)
conf = cc2parts(conf,val_parts)
plt.hist(conf)
plt.show()

potes = "/media/mhealthra2/Data/Heart_Sound/Physionet/answers.txt"
pdf = pd.read_csv(potes, header=None)
pdf.set_index(0,inplace=True)
gt = "/media/mhealthra2/Data/Heart_Sound/Physionet/2016-07-25_Updated files for Challenge 2016/Online Appendix_training set.csv"
gtdf = pd.read_csv(gt)
gtdf.set_index('Challenge record name',inplace=True)
pdf = pdf.join(gtdf,how='left')
files = pdf.index.str[0]
calc_metrics(true=pdf['Class (-1=normal 1=abnormal)']>0,pred=pdf[1]>0,subset=pdf.index.str[0])

#--- Grad-CAM

fig = plt.figure(figsize=(6,6))
conf = model_confidence(model=model,data=x_val,labels=y_val, verbose=1)
conf = cc2parts(conf,val_parts)
plt.hist(conf,40)
plt.show()

cond = np.logical_and(conf>.8,conf<.9)
_,idx = np.where([cond])
print('Number of Recordings within condition',len(idx))

target_idx = np.random.randint(len(idx))
print('Target Recording from subset-',cc2parts(val_files,val_parts)[idx[target_idx]])

cc_idx = idx_parts2cc([idx[target_idx]],val_parts)

target_data = x_val[cc_idx]
target_labels = y_val[:,1][cc_idx]
print('Target Recording Class',target_labels[0])
target_data.shape

#--- Inspect Training Sample

np.linspace(0,len(target_data)/1000,num=len(target_data))
cc2parts(train_files,train_parts).shape

target = 'a0182.wav'

filenames = pd.read_csv('../data/feature/folds/text/train_files.txt',header=None)
idx = np.where(filenames[0]==target)[0]
print(idx)
cc_idx = idx_parts2cc(idx,train_parts)
target_data = x_train[cc_idx]
target_labels = y_train[:,1][cc_idx]
print('Target Recording Class',target_labels[0])
print('Number of cc',target_data.shape[0])

fig = plt.figure()
rec = cc2rec(target_data[:6])
plt.plot(np.linspace(0,len(rec)/1000,num=len(rec)),rec)
plt.show()

#--- Inspect Validation Sample
target = 'b0003'
filenames = pd.read_csv('../data/feature/folds/text/validation0.txt',header=None)
idx = np.where(filenames[0]==target)[0]
cc_idx = idx_parts2cc(idx,val_parts)

target_data = x_val[cc_idx]
target_labels = y_val[:,1][cc_idx]
print('Target Recording Class',target_labels[0])
print('Target Recording cc',target_labels.shape)
target_data.shape

fig = plt.figure()
rec = cc2rec(target_data)
plt.plot(np.linspace(0,len(rec)/1000,num=len(rec)),rec)
plt.show()

#--- Inspect Test Sample

idx = 188
cc_idx = idx_parts2cc(idx,test_parts)

target_data = x_test[cc_idx]
target_labels = y_test[:,1][cc_idx]
print('Target Recording Class',target_labels[0])
target_data.shape

logs=[
"potes_fold0_noFIR 2019-03-16 18:44:45.597226", # potes non balanced
"potes_fold0_noFIR 2019-03-02 13:01:33.636778", # potes
"fold0_noFIR 2019-02-27 19:52:21.543329", # Type1 macc
# "fold0_noFIR 2019-03-07 14:44:47.022240", # Type2 macc 80 epoch
# "fold0_noFIR 2019-03-08 03:28:46.740442", # Type3 sensitivity
# "fold0_noFIR 2019-03-08 14:50:52.332924", # type4 val_acc
# "fold0_noFIR 2019-03-09 01:34:03.547265", # gamma stage 1
"fold0_noFIR 2019-03-06 14:21:29.823568", # zero stage2
]
lognames=[
"Potes-CNN",
"Potes-CNN DBT",
"Type I tConv",
# "Type II tConv",
# "Type III tConv",
# "Type IV tConv",
# "Gammatone tConv",
"Zero Phase tConv",
]
colors = [
'#434B77',
'#669966',
'#c10061',
'#ff51a5',
'k',
'#ffbe4f',
'#DBBBBB',
'#008080',
         ]

cc_start = 0
cc_end = 8
ax = grad_cam_logs(logs,'concatenate_1',target_data[cc_start:cc_end],target_labels[cc_start:cc_end],win_size=10,
                   lognames=lognames,colors=colors,output_class='pred',normalize=True)
ax[1].set_yticks([0,.25,.5,.75,1])

fig = plt.gcf()
fig.set_size_inches(3.5,5)
ax[1].set_yticks([0,.25,.5,.75,1])
ax[2].set_ylim([-.1,6])
# fig.savefig('Normal.eps')

# plt.savefig('MRgradCAM.eps')
# plt.xlim([0,4.7])

ax[2].legend_ = None
# ax[2].legend(lognames)


#--- Error Analysis
ax = plot_confidence_logs(logs,lognames)

chartBox = ax[2].get_position()
# ax[0].set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
ax[2].legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)




























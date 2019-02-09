import sys
import os
import numpy as np
import pandas as pd

if len(sys.argv) != 7:
	print "USAGE: python main.py ${DATASET} ${EXP} ${PHASE} ${NUM_CLS} ${NUM_SAMPLES} ${GT}"
	exit(1)

import myAugmentation as ma
import os
from shutil import copyfile


dataset = sys.argv[1]

exp = int(sys.argv[2])
phase = int(sys.argv[3])
num_cls = int(sys.argv[4])
num_samples = int(sys.argv[5])
gt = sys.argv[6]

csv_path = os.path.join("./data/%s/Annotations/"%dataset, gt.split('.')[0]+'.csv')
table = pd.read_csv(csv_path)[:][0:num_cls].sort_values(by='class_ind', ascending=True)

classes = np.array(table['class_name'].tolist())


if not os.path.exists(('./results/%s_%s_result%d/' % (dataset, gt, exp)) ):
	os.mkdir(('./results/%s_result%d/' % (dataset, exp)))

#log the code
copyfile('./main.py', './results/%s_%s_result%d/log_code_%s_exp%d_phase%d.py' % (dataset, gt, exp, dataset, exp, phase))


#gt_path = '/home/hossam/git/mtsd_augmentation/data/MTSD/Annotations/gt_after_correction.txt'
gt_path = './data/%s/Annotations/%s'%(dataset, gt)
#keep = ma.getKeep(gt_path, 50, "Classes Histogram - Malaysian dataset", 66, dataset, exp, phase)
keep = ma.getKeep(gt_path, 50, "Classes Histogram - %s dataset before augmentation"%dataset, num_cls, dataset, exp, phase, gt)


#ma.convertGT(gt_path, exp)

#ma.blurObjectsGenMask('./data/MTSD/Images/', keep, exp)

ma.convertGT(gt_path, exp, dataset, phase, classes)

ma.blurObjectsGenMask('./data/%s/Images/' % dataset, keep, exp, dataset, phase, gt, classes)

import Augmentor

p = Augmentor.Pipeline('./data/%s_%s_test%d/aug%d/' % (dataset, gt, exp, phase)) 

p.ground_truth('./data/%s_%s_test%d/aug_gt%d/' % (dataset, gt, exp, phase))

p.rotate(probability=0.4, max_left_rotation=20, max_right_rotation=20)
p.zoom(probability=0.3, min_factor=1.05, max_factor=1.1) 

p.status()

p.sample(num_samples, multi_threaded=True)


ma.genNewAnnotation(exp, dataset, phase, gt, classes)



fm = open(gt_path, 'rt')
fs = open('./results/%s_%s_result%d/gt_phase_%d.txt' % (dataset, gt, exp, phase), 'rt')
fo = open('./results/%s_%s_result%d/gt_phase_%d_combine.txt' % (dataset, gt, exp, phase), 'wt')

lines = fm.readlines()
if lines[-1][-1] != '\n':
	lines[-1] = lines[-1]+'\r\n'
lines = lines + fs.readlines()
#while lines.count(""):
#	lines.remove("")
 
#fo.writelines(lines)

#lines = fs.readlines()
#while lines.count(""):
#	lines.remove("")

fo.writelines(lines)
fo.close()

#phase = 2
keep = ma.getKeep('./results/%s_%s_result%d/gt_phase_%d_combine.txt' % (dataset, gt, exp, phase), 50, "Classes Histogram - %s dataset after augmentation"%dataset, num_cls, dataset, exp, phase, gt, 'after')

if not os.path.exists(('../Datasets/%s_%s_AUG%d'% (dataset, gt, exp))):
	os.mkdir(('../Datasets/%s_%s_AUG%d' % (dataset, gt, exp)))
	os.mkdir(('../Datasets/%s_%s_AUG%d/Annotations' % (dataset, gt, exp)))
	os.mkdir(('../Datasets/%s_%s_AUG%d/Images' % (dataset, gt, exp)))

files = [f for f in os.listdir('./data/%s_%s_test%d/aug%d/output/' % (dataset, gt, exp, phase)) if 'original' in f]
for f in files:
	os.symlink(('../../../mtsd_augmentation/data/%s_%s_test%d/aug%d/output/'%(dataset, gt, exp, phase))+f, ('../Datasets/%s_%s_AUG%d/Images/'%(dataset, gt, exp))+f)


files = [f for f in os.listdir('./data/%s/Images/' % dataset) if os.path.splitext(f)[1] != '.txt']

#TODO_DONE: symbol link the original images
for f in files:
	os.symlink(('../../%s/Images/' % dataset)+f, ('../Datasets/%s_%s_AUG%d/Images/'%(dataset, gt, exp))+f)

#TODO_DONE: symbol link the final annotation
copyfile('./results/%s_%s_result%d/gt_phase_%d_combine.txt' % (dataset, gt, exp, phase), '../Datasets/%s_%s_AUG%d/Annotations/gt_phase_%d_combine.txt' % (dataset, gt, exp, phase))

#copyfile('./main.py', './results/%s_result%d/log_code_%s_exp%d_phase%d.py' % (dataset, exp, phase))


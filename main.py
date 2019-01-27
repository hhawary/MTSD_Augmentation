import sys

if len(sys.argv) != 6:
	print "USAGE: python main.py ${DATASET} ${EXP} ${PHASE} ${NUM_CLS} ${NUM_SAMPLES}"
	exit(1)

import myAugmentation as ma
import os
from shutil import copyfile


dataset = sys.argv[1]

exp = int(sys.argv[2])
phase = int(sys.argv[3])
num_cls = int(sys.argv[4])
num_samples = int(sys.argv[5])

if not os.path.exists(('./results/%s_result%d/' % (dataset, exp)) ):
	os.mkdir(('./results/%s_result%d/' % (dataset, exp)))

#log the code
copyfile('./main.py', './results/%s_result%d/log_code_%s_exp%d_phase%d.py' % (dataset, exp, dataset, exp, phase))


#gt_path = '/home/hossam/git/mtsd_augmentation/data/MTSD/Annotations/gt_after_correction.txt'
gt_path = './data/%s/Annotations/gt.txt'%dataset
#keep = ma.getKeep(gt_path, 50, "Classes Histogram - Malaysian dataset", 66, dataset, exp, phase)
keep = ma.getKeep(gt_path, 50, "Classes Histogram - %s dataset before augmentation"%dataset, num_cls, dataset, exp, phase)


#ma.convertGT(gt_path, exp)

#ma.blurObjectsGenMask('./data/MTSD/Images/', keep, exp)

ma.convertGT(gt_path, exp, dataset, phase)

ma.blurObjectsGenMask('./data/%s/Images/' % dataset, keep, exp, dataset, phase)

import Augmentor

p = Augmentor.Pipeline('./data/%s_test%d/aug%d/' % (dataset, exp, phase)) 

p.ground_truth('./data/%s_test%d/aug_gt%d/' % (dataset, exp, phase))

p.rotate(probability=0.4, max_left_rotation=20, max_right_rotation=20)
p.zoom(probability=0.3, min_factor=1.05, max_factor=1.1) 

p.status()

p.sample(num_samples, multi_threaded=True)


ma.genNewAnnotation(exp, dataset, phase)



fm = open(gt_path, 'rt')
fs = open('./results/%s_result%d/gt_phase_%d.txt' % (dataset, exp, phase), 'rt')
fo = open('./results/%s_result%d/gt_phase_%d_combine.txt' % (dataset, exp, phase), 'wt')

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
keep = ma.getKeep('./results/%s_result%d/gt_phase_%d_combine.txt' % (dataset, exp, phase), 50, "Classes Histogram - %s dataset after augmentation"%dataset, num_cls, dataset, exp, phase, 'after')

if not os.path.exists(('../Datasets/%s_AUG%d'% (dataset, exp))):
	os.mkdir(('../Datasets/%s_AUG%d' % (dataset, exp)))
	os.mkdir(('../Datasets/%s_AUG%d/Annotations' % (dataset, exp)))
	os.mkdir(('../Datasets/%s_AUG%d/Images' % (dataset, exp)))

files = [f for f in os.listdir('./data/%s_test%d/aug%d/output/' % (dataset, exp, phase)) if 'original' in f]
for f in files:
	os.symlink(('../../../mtsd_augmentation/data/%s_test%d/aug%d/output/'%(dataset, exp, phase))+f, ('../Datasets/%s_AUG%d/Images/'%(dataset, exp))+f)


files = [f for f in os.listdir('./data/%s/Images/' % dataset) if os.path.splitext(f)[1] != '.txt']

#TODO_DONE: symbol link the original images
for f in files:
	os.symlink(('../../%s/Images/' % dataset)+f, ('../Datasets/%s_AUG%d/Images/'%(dataset, exp))+f)

#TODO_DONE: symbol link the final annotation
copyfile('./results/%s_result%d/gt_phase_%d_combine.txt' % (dataset, exp, phase), '../Datasets/%s_AUG%d/Annotations/gt_phase_%d_combine.txt' % (dataset, exp, phase))

#copyfile('./main.py', './results/%s_result%d/log_code_%s_exp%d_phase%d.py' % (dataset, exp, phase))


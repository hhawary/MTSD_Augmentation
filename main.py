import myAugmentation as ma
import os

dataset = 'MTSD'

exp = 8
phase = 1
gt_path = '/home/hossam/git/mtsd_augmentation/data/MTSD/Annotations/gt_after_correction.txt'
keep = ma.getKeep(gt_path, 50, "Classes Histogram - Malaysian dataset", 66, dataset, exp, phase)

#ma.convertGT(gt_path, exp)

#ma.blurObjectsGenMask('./data/MTSD/Images/', keep, exp)

ma.convertGT(gt_path, exp, dataset, phase)

ma.blurObjectsGenMask('./data/%s/Images/' % dataset, keep, exp, dataset, phase)

import Augmentor

p = Augmentor.Pipeline('./data/%s_test%d/aug%d/' % (dataset, exp, phase)) 

p.ground_truth('./data/%s_test%d/aug_gt%d/' % (dataset, exp, phase))

p.rotate(probability=0.4, max_left_rotation=5, max_right_rotation=5)
p.zoom(probability=0.3, min_factor=1.05, max_factor=1.1) 

p.status()

p.sample(50, multi_threaded=True)


ma.genNewAnnotation(exp, dataset, phase)



fm = open(gt_path, 'rt')
fs = open('./results/%s_result%d/gt_phase_%d.txt' % (dataset, exp, phase), 'rt')
fo = open('./results/%s_result%d/gt_phase_%d_combine.txt' % (dataset, exp, phase), 'wt')
fo.writelines(fm.readlines()) 
fo.writelines(fs.readlines())
fo.close()

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
#TODO: symbol link the final annotation



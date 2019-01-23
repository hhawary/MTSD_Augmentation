import myAugmentation as ma

exp = 7
gt_path = '/home/hossam/git/mtsd_augmentation/data/MTSD/Annotations/gt_after_correction.txt'
keep = ma.getKeep(gt_path, 50)

#ma.convertGT(gt_path, exp)

#ma.blurObjectsGenMask('./data/MTSD/Images/', keep, exp)

ma.convertGT(gt_path, exp)

ma.blurObjectsGenMask('./data/MTSD/Images/', keep, exp)

import Augmentor

p = Augmentor.Pipeline('./data/test%d/aug/' % exp) 

p.ground_truth('./data/test%d/aug_gt/' % exp)

p.rotate(probability=0.4, max_left_rotation=5, max_right_rotation=5)
p.zoom(probability=0.3, min_factor=1.05, max_factor=1.1) 

p.status()

p.sample(4400, multi_threaded=True) 

ma.genNewAnnotation(exp)

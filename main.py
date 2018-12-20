import myAugmentation as ma

keep = ma.getKeep('/home/hossam/git/mtsd_augmentation/data/MTSD/Annotations/gt_after_correction.txt',50)

ma.convertGT('/home/hossam/git/mtsd_augmentation/data/MTSD/Annotations/gt_after_correction.txt', 6)

ma.blurObjectsGenMask('./data/MTSD/Images/', keep, 6)

ma.convertGT('/home/hossam/git/mtsd_augmentation/data/MTSD/Annotations/gt_after_correction.txt', 6)

ma.blurObjectsGenMask('./data/MTSD/Images/', keep, 6)

import Augmentor

p = Augmentor.Pipeline('./data/test6/aug/') 

p.ground_truth('./data/test6/aug_gt/')

p.rotate(probability=0.4, max_left_rotation=5, max_right_rotation=5)
p.zoom(probability=0.3, min_factor=1.05, max_factor=1.1) 

p.status()

p.sample(4400, multi_threaded=True) 

ma.genNewAnnotation(6)

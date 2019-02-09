import sys
import numpy as np
import os
import pandas as pd
import progressbar

if __name__ == '__main__':
	if len(sys.argv) != 6:
		print "USAGE: python convert_aug.py ${DATASET} ${EXP} ${PHASE} ${NUM_CLS} ${GT}"
		exit(1)
	
	dataset = sys.argv[1]

	exp = int(sys.argv[2])
	phase = int(sys.argv[3])
	num_cls = int(sys.argv[4])
	gt = sys.argv[5]

	#csv_path = os.path.join("./data/%s/Annotations/"%dataset, gt.split('.')[0]+'.csv')
	#table = pd.read_csv(csv_path)[:][0:num_cls].sort_values(by='class_ind', ascending=True)

	#classes = np.array(table['class_name'].tolist())

	anno_file_path = '../Datasets/%s_%s_AUG%d/Annotations/gt_phase_%d_combine.txt'%(dataset, gt, exp, phase)
	gt_in_f = open(anno_file_path, 'r')
	lines = gt_in_f.readlines()
	dic = {}
	tmp = []

	for line in lines:
		spl = line.split(';')
		tmp.append(int(spl[5]))
		if dic.has_key(spl[0]):
			dic[spl[0]].append(line)
		else:
			dic[spl[0]] =[line]
	

	# Convert the annotation to caffe-faster-rcnn
	# 
	f_path_out = "../caffe-faster-rcnn/examples/FRCNN/dataset/%s_%s_aug%d_%d.trainval"%(dataset.lower(), gt, exp, phase)
	my_f_out = open(f_path_out, 'wt')
	for i, trains in enumerate(dic):
		my_f_out.write("# {}\n".format(str(i)))
		my_f_out.write(trains+'\n')
		my_f_out.write(str(len(dic[trains]))+'\t\n')
		for line in dic[trains]:
			split = line.split(';')
			filename = split[0]
			x1 = int(split[1])
			y1 = int(split[2])
			x2 = int(split[3])
			y2 = int(split[4])

			cls = int(split[5])
			my_f_out.write("{}\t{}\t{}\t{}\t{}\t0\n".format(str(cls+1), str(x1), str(y1), str(x2), str(y2)) )
	my_f_out.close()


	from IPython import embed; embed()

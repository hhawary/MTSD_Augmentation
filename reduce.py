import numpy as np
import os
import pandas as pd
import progressbar
import copy

classes_all = np.array([
                         'u-turn', 'keep-right', 'keep-left', 'pass-either-side',
                         'compulsory-motor-cycles-track', 'stop', 'no-left-turn', 'no-right-turn', 'no-u-turn',
                         'no-entry', 'weight-limit-sign-5T', 'weight-limit-sign-30T', 'height-limit-sign-2.-m',
                         'height-limit-sign-3.-m', 'height-limit-sign-4.-m', 'height-limit-sign-5.-m', 'height-limit-sign-6.-m',
                         'speed-limit-20', 'speed-limit-30', 'speed-limit-40', 'speed-limit-50', 'speed-limit-60', 'speed-limit-70',
                         'speed-limit-80', 'speed-limit-90','speed-limit-110', 'no-entry-for-vehicles-ex-5T-truntks-etc',
                         'heavy-vehicles-no-driving-on-right-lane', 'no-parking', 'no-stopping', 'give-way', 'wide-limit-3.-m',
                         'no-overtaking', 'road-work', 'camera-operation-zone', 'crosswind-area', 'caution-hump',
                         'hump-ahead', 'towing-zone', 'left-bend', 'slippery-road', 'pedestrain-crossing-opt1', 'pedestrain-crossing-opt2',
                         'school-childern-crossing-opt1', 'school-childern-crossing-opt2', 'caution', 'narrow-roads-on-the-left',
                         'traffic-lights-ahead', 'obstacles', 'staggered-junctions', 'crossroads-T-junction', 'crossroads-to-the-right',
                         'crossroads-to-the-left', 'exit-to-the-left', 'crossroads', 'minor-road-on-right', 'minor-road-on-left',
                         'minor-road-on-left-opt2', 'cattle-crossing', 'roundabout-ahead', 'narrow-bridge','split-way', 'two-way-road',
                         'divided-road-ending', 'curve-on-the-left', 'crossroads-Y-junction'])


if __name__ == '__main__':

	kept = np.sort([7,8,16,23,27,37,49,57,58,62])
	kept[:] = [x - 1 for x in kept]
	classes = classes_all[kept]
	for c in classes:
		print("\'{}\', ".format(c))




	freq = np.array([70.,20,56,11,1,20,11,6,20,51,7,2,9,14,39,144,1,2,26,13,54,32,4,67,30,25,12,9,42,66,85,2,2,27,35,9,72,28,41,9,2,11,37,37,9,27,8,59,235,3,3,11,18,15,13,8,87,4,6,11,6,264,1,1,4,2])
	#pre = np.array([0.6,0.424,0.622,0.909,1,0.826,0.992,1,0.729,0.725,0.686,1,0.407,0.635,0.707,0.629,0,1,0.748,0.718,0.688,0.622,1,0.643,0.671,0.605,1,0.457,0.664,0.593,0.603,1,0.45,0.497,0.55,0.604,0.763,0.729,0.837,0.815,0.833,0.227,0.69,0.753,0.544,0.689,0.715,0.782,0.8,0.167,0.833,0.548,0.763,0.775,0.962,0.557,0.752,1,0.738,0.894,0.611,0.733,1,1,1,1])
	pre = np.array([0.7258120762,0.5716843762,0.5798444057,0.9090909091,1,0.9917355372,0.8747474747,1,0.7854545455,0.7882231405,0.9338842975,1,0.461038961,0.6093280812,0.7226107226,0.6889243374,1,1,0.7338497174,0.9808857809,0.7655594589,0.7794612795,1,0.7385875562,0.859349819,0.6462022826,1,0.528649668,0.6218885281,0.6178161829,0.6172363441,1,0.2727272727,0.7157046657,0.5901185173,0.8838383838,0.766439651,0.9473445344,0.8930735931,0.9669421488,1,0.3106060606,0.8878880266,0.806550352,0.75002775,0.6472100559,0.9107142857,0.876969697,0.7652400182,0.0303030303,0.9090909091,0.6387403446,0.8352813853,0.9659090909,0.9090909091,0.6363636364,0.8785786251,1,0.7373737374,0.9924242424,0.6060606061,0.7903407398,1,1,1,1])
	freqN = freq / max(freq)	
	preInv = 1 - pre
	freqNInv = 1 - freqN

	df = pd.DataFrame({
		'class_ind'  : np.array(range(0,66))+1,
		'class_name' : classes_all,
		'freq' : freq,
		'Precision' : pre,
		'1 - Pre'   : preInv,
		'Freq/max(frq)' : freqN,
		'1 - FreqN'     : freqNInv,
		'((FREQ/264)+PRE)/2' : (freqN + pre)/2.0,
		'((FREQ/264)+(1-PRE))/2' : (freqN + preInv)/2.0,
		'((1-(FREQ/264))+PRE)/2' : (freqNInv + pre)/2.0,
		'((1-(FREQ/264))+(1-PRE))/2' : (freqNInv + preInv)/2.0
	})
	
	HFHP = df.sort_values(by=['((FREQ/264)+PRE)/2'], ascending=False)		#High Freq, High Pre
	HFLP = df.sort_values(by=['((FREQ/264)+(1-PRE))/2'], ascending=False)		#High Freq, Low Pre
	LFHP = df.sort_values(by=['((1-(FREQ/264))+PRE)/2'], ascending=False)		#Low Freq, High Pre
	LFLP = df.sort_values(by=['((1-(FREQ/264))+(1-PRE))/2'], ascending=False)	#Low Freq, Low Pre
	
	# 10 HFHP	
	
	kept2 = np.sort(HFHP['class_ind'][0:10].tolist())-1
	classes2 = classes_all[kept2]

	anno_file = open('./data/MTSD/Annotations/gt.txt', 'rt')
	anno_lines = anno_file.readlines()
	anno_file.close()

	anno_out = open('./data/MTSD/Annotations/gt_reduced_10_HFHP.txt','wt')
	HFHP.to_csv(path_or_buf='./data/MTSD/Annotations/gt_reduced_10_HFHP.csv')
	for line in anno_lines:
		spl = line.split(';')
		if int(spl[5]) in kept2:
			anno_out.write( ';'.join(spl[0:5]) + ';' + str(kept2.tolist().index(int(spl[5]))) +'\r\n')
	
	anno_out.close()

	# 10 HFLP
	kept2 = np.sort(HFLP['class_ind'][0:10].tolist())-1
	classes2 = classes_all[kept2]

	anno_out = open('./data/MTSD/Annotations/gt_reduced_10_HFLP.txt','wt')
	HFLP.to_csv(path_or_buf='./data/MTSD/Annotations/gt_reduced_10_HFLP.csv')
	for line in anno_lines:
		spl = line.split(';')
		if int(spl[5]) in kept2:
			anno_out.write( ';'.join(spl[0:5]) + ';' + str(kept2.tolist().index(int(spl[5]))) +'\r\n')
	
	anno_out.close()



	# 10 LFHP
	kept2 = np.sort(LFHP['class_ind'][0:10].tolist())-1
	classes2 = classes_all[kept2]

	anno_out = open('./data/MTSD/Annotations/gt_reduced_10_LFHP.txt','wt')
	LFHP.to_csv(path_or_buf='./data/MTSD/Annotations/gt_reduced_10_LFHP.csv')
	for line in anno_lines:
		spl = line.split(';')
		if int(spl[5]) in kept2:
			anno_out.write( ';'.join(spl[0:5]) + ';' + str(kept2.tolist().index(int(spl[5]))) +'\r\n')
	
	anno_out.close()


	# 10 LFLP
	kept2 = np.sort(LFLP['class_ind'][0:10].tolist())-1
	classes2 = classes_all[kept2]

	anno_out = open('./data/MTSD/Annotations/gt_reduced_10_LFLP.txt','wt')
	LFLP.to_csv(path_or_buf='./data/MTSD/Annotations/gt_reduced_10_LFLP.csv')
	for line in anno_lines:
		spl = line.split(';')
		if int(spl[5]) in kept2:
			anno_out.write( ';'.join(spl[0:5]) + ';' + str(kept2.tolist().index(int(spl[5]))) +'\r\n')
	
	anno_out.close()
	

	# print the classes
	print("Classes that has been kept after HFLP")
	kept2 = np.sort(HFHP['class_ind'][0:10].tolist())-1
	#kept2[:] = [x for x in kept2]
	classes = classes_all[kept2]
	for c in classes:
		print("\'{}\', ".format(c))


	# Split dataset 70% tranning and 30% testing
	anno_file_path = './data/MTSD/Annotations/gt_reduced_10_HFHP.txt'
	# (1) Read the GT into a dictionary with file name as key
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
        
	print(len(dic))

	# (2) Calculate the frequancy of each class
	_, counts = np.unique(np.array(tmp), return_counts=True)
	frq = np.array(counts)

	print(frq)
	
	# (3) Sort with key of minimum of minimum frequancy in the dictionary value
	def min_frq(vv, frq):
		min_f = frq[int(vv[0].split(';')[5].strip())]
		for v in vv:
			spl = v.split(';')
			cfrq = frq[int(spl[5].strip())]
			if cfrq < min_f:
				min_f = cfrq
		return min_f

	# (4) Calculate the 30% frequancy for each class and round the value
	frq_30 = np.round(frq*30/100.0)
	#frq_70 = np.round(frq*70/100.0)
	print(frq_30)
	#print(frq_70)


	train_keys = []
	test_keys = []

	for key, value in sorted(dic.iteritems(), key=lambda (k,v): min_frq(v, frq)): #len(v)):
		flag_test = True
		#print("%s:\t\t\t" % (key), end='')
		frq_30_copy = copy.deepcopy(frq_30)
		for val in value:
			#print (val.replace('\n','\t')," ",end='')
			cls = int(val.split(';')[5].strip())
			frq_30_copy[cls] -= 1
			if (frq_30_copy[cls] <= -1):
				flag_test = False
				break
		# (5) If at least one of the values frequancy of 30% will \
		#     be less than 0 after subtract from 1 then add key to train_set otherwise add to test_set
		if flag_test:
			## Test
			for val in value:
				cls = int(val.split(';')[5].strip())
				frq_30[cls] -= 1
			test_keys.append(key)
		else:
			train_keys.append(key)
		#print("")	
	
	print("Train_keys = ", len(train_keys),", Test_keys = ", len(test_keys), ", Total number of images = ", (len(train_keys)+len(test_keys)))
	print((len(train_keys)+len(test_keys)) == len(dic))

	file_p = open(anno_file_path.replace('.txt', '.train.txt'),'wt')
	#anno_file_path.replace('.txt', '.train.txt')

	for trains in train_keys:
		for line in dic[trains]:
			file_p.write(line)
	file_p.close()

	file_p = open(anno_file_path.replace('.txt', '.test.txt'), 'wt')
	for tests in test_keys:
		for line in dic[tests]:
			file_p.write(line)
	file_p.close()


	# Convert the annotation to caffe-faster-rcnn
	# train
	f_path_out = "../caffe-faster-rcnn/examples/FRCNN/dataset/mtsd_HFHP_reduced.train.trainval"
	my_f_out = open(f_path_out, 'wt')
	for i, trains in enumerate(train_keys):
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

	# test
	f_path_out = "../caffe-faster-rcnn/examples/FRCNN/dataset/mtsd_HFHP_reduced.test.test"
	my_f_out = open(f_path_out, 'wt')
	for i, trains in enumerate(test_keys):
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

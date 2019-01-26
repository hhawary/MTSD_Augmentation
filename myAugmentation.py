import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

import time
import progressbar

classes = np.array([
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


color_list = [(255,0,0),
             (255,255,0),
             (255,0,255),
             (255,255,255),
             (0,0,255),
             (0,255,0),
             (128,0,0),
             (128,128,0),
             (128,0,128),
             (128,128,128),
             (0,0,128),
             (0,128,0),
             (64,0,0),
             (64,64,0),
             (64,0,64),
             (64,64,64),
             (0,0,64),
             (0,64,0),
             (32,0,0),
             (32,32,0),
             (32,0,32),
             (32,32,32),
             (0,0,32),
             (0,32,0)
             ]

def convertGT(f_path, num, dataset, phase):

	aug_folder = ('./data/%s_test%d/aug%d/' % (dataset,num,phase))
	label_folder = ('./data/%s_test%d/labels/' % (dataset, num))
	mask_folder = ('./data/%s_test%d/aug_gt%d/' % (dataset, num, phase))
	if not os.path.exists(('./data/%s_test%d/' % (dataset, num)) ):
    		os.mkdir(('./data/%s_test%d/' % (dataset, num)))
		os.mkdir(aug_folder)
		os.mkdir(label_folder)
		os.mkdir(mask_folder)

	#label_folder = ('./data/test%d/labels/' % num)
	if not os.path.exists(label_folder):
		os.mkdir(label_folder)
	f = open(f_path,"rt")
	i = 0
	my_dict = {}
	lines = f.readlines()
	with progressbar.ProgressBar(maxval=len(lines)) as bar:
		for i, line in enumerate(lines):
			split = line.split(';')
			if not my_dict.has_key(split[0]):
				my_dict.update({split[0]:[]})

			filename = split[0]
			x1 = int(split[1])
			y1 = int(split[2])
			x2 = int(split[3])
			y2 = int(split[4])

			cls = int(split[5])

			#x,y,w,h = convert([my_dict_shape[split[0]][1],my_dict_shape[split[0]][0]], [x1,x2,y1,y2])
			my_obj = (cls,x1,y1,x2,y2)
			my_dict[split[0]].append(my_obj)
			#f_out.write(str("{}\n".format('/home/mjiit/Hossam/darknet/data/mtsd/MTSD/Images/'+filename)))
			#f_out.write(str("{} {} {} {} {}\n".format(cls, x, y, w, h)))
			bar.update(i)
	print ('Done...')
	#print (len(my_dict.items()), my_dict.items())
	f.close()
	for i, it in enumerate(my_dict):
		f_path_label_out = (label_folder+"{}.txt").format('.'.join(it.split('.')[0:-1]))
		f_label_out = open(f_path_label_out, 'wt')
		for it2 in my_dict[it]:
			print(str("{};{};{};{};{}\n".format(classes[it2[0]],it2[1],it2[2],it2[3],it2[4])))
			f_label_out.write(str("{} {} {} {} {}\n".format(classes[it2[0]],it2[1],it2[2],it2[3],it2[4])))
		f_label_out.close()

def getKeep(file_p, min_freq, title, num_classes):
	file = open(file_p,'rt')
	tmp = []
	for line in file:
		tmp.append( int(line.split(';')[5].replace('\r\n',''))+1)
	plt.hist(tmp,histtype='bar', rwidth=0.8,bins=range(1,num_classes+3,1), align='left')
	plt.title(title)
	plt.xlabel("Value")
	plt.ylabel("Frequency")

	plt.show()
	file.close()

	a = np.array(tmp)
	unique, counts = np.unique(a, return_counts=True)

	frq = np.array(counts)
	# print counts
	keep = np.where(frq < min_freq)

	print 'freq->\n',counts

	print 'Mean: {:.2f}\n'.format(frq.mean()),\
	    'Var: {:.2f}\n'.format(frq.var()),\
	    'Max: {:.2f}\n'.format(frq.max()),\
	    'Min: {:.2f}\n'.format(frq.min()),\
	    'Sum: {:.2f}\n'.format(frq.sum()),\
	    'STD: {:.2f}\n'.format(frq.std()),\
	    'CV: {:.2f}\n'.format(frq.std()/frq.mean())
	print 'Total number of objects:',len(tmp)
	file.close()
	return keep

def blurObjectsGenMask(im_path, keep, num, dataset, phase):
	files = [f for f in os.listdir(im_path) if os.path.isfile(os.path.join(im_path,f))]
	aug_folder = ('./data/%s_test%d/aug%d/' % (dataset, num, phase))
	label_folder = ('./data/%s_test%d/labels/' % (dataset, num))

	mask_folder = ('./data/%s_test%d/aug_gt%d/' % (dataset, num, phase))
	
	if not os.path.exists(('./data/%s_test%d/' % (dataset,num)) ):
    		#os.mkdir(('./data/test%d/' % num))
		#os.mkdir(aug_folder)
		#os.mkdir(label_folder)
		#os.mkdir(mask_folder)
		print("Convert the GT first")
		return

	for fname in files:
		if fname.split('.')[1] == 'txt':
		    continue
		#print fname
		if not os.path.isfile(os.path.join(label_folder, fname.split('.')[0]+'.txt')):
		    #fault = open('faults.txt', 'a')
		    #fault.write(fil+'\n')
		    continue
		
		store = False

		im = cv2.imread(os.path.join(im_path, fname))
		org = im.copy()
		f = open(os.path.join(label_folder, fname.split('.')[0]+'.txt'), 'r')
		
		lines = f.readlines()
		text = ''
		for line in lines:
		    spl = line.split(' ')
		    i = np.where(classes == spl[0])[0][0]
		    if i not in keep[0]:
		        x1 = int(spl[1])#+1
		        y1 = int(spl[2])#+1
		        x2 = int(spl[3])#-1
		        y2 = int(spl[4])#-1

		        #print y2-y1, x2-x1

		        #cv2.imwrite('without_aug.jpg', im)
		        #print im[y1:y2,x1:x2]

		        np.random.shuffle(im[y1:y2+1,x1:x2+1,0])
		        np.random.shuffle(im[y1:y2+1,x1:x2+1,0].T)

		        np.random.shuffle(im[y1:y2+1,x1:x2+1,1])
		        np.random.shuffle(im[y1:y2+1,x1:x2+1,1].T)

		        np.random.shuffle(im[y1:y2+1,x1:x2+1,2])
		        np.random.shuffle(im[y1:y2+1,x1:x2+1,2].T)

		        #np.random.shuffle(im[y1:y2,x1:x2])
		        #np.random.shuffle(im[y1:y2,x1:x2].T)

		        blur = cv2.blur(im, (50, 50))

		        #print lines[0] , x1, x2, y1, y2

		        #print im[y1:y2,x1:x2,0]
		        #cv2.imwrite('aug.jpg', im)

		        num =10
		        im[y1-num:y2+num,x1-num:x2+num] = blur[y1-num:y2+num,x1-num:x2+num]
		        #cv2.imwrite('aug_blur.jpg', im)

		for line in lines:
		    spl = line.split(' ')
		    i = np.where(classes == spl[0])[0][0]
		    if i in keep[0]:
		        store = True
		        
		        #spl = line.split(' ')
		        x1 = int(spl[1])#+1
		        y1 = int(spl[2])#+1
		        x2 = int(spl[3])#-1
		        y2 = int(spl[4])#-1
		        text += line
		        #im2 = cv2.imread(os.path.join(im_path, fname))
		        ynum = int(y2-y1)/14
		        xnum = int(x2-x1)/14
		        im[y1-ynum:y2+ynum,x1-xnum:x2+xnum] = org[y1-ynum:y2+ynum,x1-xnum:x2+xnum]
		
		if store:
		    cv2.imwrite(os.path.join(aug_folder,fname.split('.')[0]+('_aug%d'%phase)+'.jpg'), im)
		    f_out = open(os.path.join(label_folder, fname.split('.')[0]+('_aug%d'%phase)+'.txt'), 'w')
		    f_out.write(text)
		    f_out.close()

	files = [f for f in os.listdir(aug_folder) if os.path.isfile(os.path.join(aug_folder,f))]
	for fil in files:
		if not os.path.isfile(os.path.join(label_folder, fil.split('.')[0]+'.txt')):
			continue
		im = cv2.imread(os.path.join(aug_folder, fil))
		f = open(os.path.join(label_folder, fil.split('.')[0]+'.txt'))
		lines = f.readlines()
		im_out = np.zeros(im.shape, np.uint8)
		for i, line in enumerate(lines):
			spl = line.split(' ')
			x1 = int(spl[1])#+1
			y1 = int(spl[2])#+1
			x2 = int(spl[3])#-1
			y2 = int(spl[4])#-1
			cv2.rectangle(im_out, (x1,y1), (x2,y2),color_list[i], -1)
		cv2.imwrite(os.path.join(mask_folder, fil), im_out)

def genNewAnnotation(num, dataset, phase):

	if not os.path.exists(('./data/%s_test%d/' % (dataset, num)) ):
		print(('./data/%s_test%d/' % (dataset, num)), "Not Found")
		return
	if not os.path.exists(('./data/%s_test%d/aug%d/output/' % (dataset, num, phase)) ):
		print(('./data/%s_test%d/aug%d/output/' % (dataset, num, phase)), "Not Found")
		return
	if not os.path.exists(('./results/%s_result%d/' % (dataset, num)) ):
		os.mkdir(('./results/%s_result%d/' % (dataset, num)))

	#aug_folder = ('./data/test%d/aug/' % num)
	aug_out_folder = ('./data/%s_test%d/aug%d/output/' % (dataset, num, phase))
	label_folder = ('./data/%s_test%d/labels/' % (dataset, num))

	files = [f for f in os.listdir(aug_out_folder) if os.path.isfile(os.path.join(aug_out_folder,f))]
	gts = []
	augs = []
	for fi in files:
		if 'original' in fi:
			augs.append(fi)
		else:
			gts.append(fi)

	f_out = open(('./results/%s_result%d/gt_phase_%d.txt' % (dataset, num, phase)), 'w')
	f_out_bad = open(('./results/%s_result%d/gt_phase_%d_bad.txt' % (dataset, num, phase)), 'w')

	for aug in augs:
		aug_sp = aug.split('_')
		gt = aug.replace('aug%d_original_'%phase, '_groundtruth_(1)_aug%d_'%phase)
		############## bug fix if org file name have '_' #############
		org_file =  aug_sp[2] #+'_'+aug_sp[3]
		i=3
		while('aug%d'%phase not in aug_sp[i]):
			org_file = org_file + '_' + aug_sp[i]
			i=i+1
		org_file = org_file + '_' + aug_sp[i]
		##############################################################
		if not os.path.exists(os.path.join(label_folder, org_file.split('.')[0]+'.txt')):
			print(os.path.join(label_path, org_file.split('.')[0]+'.txt'))
			continue

		im = cv2.imread(os.path.join(aug_out_folder,gt))
		f = open(os.path.join(label_folder, org_file.split('.')[0]+'.txt'), 'r')
		lines = f.readlines()
		f.close()
		for i in range(0, len(lines)):
			line_sp = lines[i].split(' ')
		
			orgW = int(line_sp[3]) - int(line_sp[1])
			orgH = int(line_sp[4]) - int(line_sp[2])
			org_aspect = float(orgW)/(int(orgH))

			lower = np.array(color_list[i])
			upper = np.array(color_list[i])
			for k in range(0, 3):
				if lower[k] == 0:
					upper[k] = 50
				else:
					upper[k] += 100
					lower[k] -= 100
			mask = cv2.inRange(im, lower, upper)
			im3 = cv2.bitwise_and(im, im, mask=mask)

			im2 = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
			ret, thresh = cv2.threshold(im2, 0, np.max(im2)+1 ,cv2.THRESH_BINARY)
			_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			if len(contours) == 0:
				continue
			#FIXMED: in 26.jpg cannot detect contour of  keep-right 804 465 809 469 5x4
			x,y,w,h = cv2.boundingRect(contours[0])
			for cnt in contours:
				cx, cy, cw, ch = cv2.boundingRect(cnt)
				if cw*ch > w*h: # bug fix compare the area instead of the width only
					x,y,w,h = cx, cy, cw, ch
			cls_num = np.where(classes == line_sp[0])[0][0]
			new_aspect = float(w)/h
			imW = im.shape[1]
			imH = im.shape[0]
			if abs(org_aspect - new_aspect)>=0.5 and (x==0 or (x+w) >= imW or y==0 or (y+h) >= imH):
				f_out_bad.write(aug+';'+str(x)+';'+str(y)+';'+str(x+w)+';'+str(y+h)+';'+ str(cls_num) +'\n')
				continue
			
			f_out.write(aug+';'+str(x)+';'+str(y)+';'+str(x+w)+';'+str(y+h)+';'+ str(cls_num) +'\n')

	f_out.close()
	f_out_bad.close()





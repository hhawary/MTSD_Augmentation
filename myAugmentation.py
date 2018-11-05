import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

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

def getKeep(file_p, min_freq):
	file = open(file_p,'rt')
	tmp = []
	for line in file:
		tmp.append( int(line.split(';')[5].replace('\r\n',''))+1)
	plt.hist(tmp,histtype='bar', rwidth=0.8,bins=range(1,69,1), align='left')
	plt.title("Classes Histogram - Malaysian dataset")
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

def blurObjectsGenMask(im_path, keep, num):
	files = [f for f in os.listdir(im_path) if os.path.isfile(os.path.join(im_path,f))]
	aug_folder = ('./data/test%d/aug/' % num)
	label_folder = ('./data/test%d/aug_labels/' % num)

	mask_folder = ('./data/test%d/aug_gt/' % num)
	
	if not os.path.exists(('./data/test%d/' % num) ):
    	os.mkdir(('./data/test%d/' % num))
		os.mkdir(aug_folder)
		os.mkdir(label_folder)
		os.mkdir(mask_folder)

	for fname in files:
		if fname.split('.')[1] == 'txt':
		    continue
		#print fname
		if not os.path.isfile(os.path.join(label_path, fname.split('.')[0]+'.txt')):
		    #fault = open('faults.txt', 'a')
		    #fault.write(fil+'\n')
		    continue
		
		store = False

		im = cv2.imread(os.path.join(im_path, fname))
		org = im.copy()
		f = open(os.path.join(label_path, fname.split('.')[0]+'.txt'), 'r')
		
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
		    cv2.imwrite(os.path.join(aug_folder,fname.split('.')[0]+'_aug'+'.jpg'), im)
		    f_out = open(os.path.join(label_folder, fname.split('.')[0]+'_aug'+'.txt'), 'w')
		    f_out.write(text)
		    f_out.close()

	for fil in files:
		if not os.path.isfile(os.path.join(label_folder, fil.split('.')[0]+'.txt')):
			continue
		im = cv2.imread(os.path.join(im_path, fil))
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

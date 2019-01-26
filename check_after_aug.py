import cv2
import os
import numpy as np

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


def drawMask(aug, num, dataset, phase):
	label_folder = ('./data/%s_test%d/labels/' % (dataset, num))
	aug_out_folder = ('./data/%s_test%d/aug%d/output/' % (dataset, num, phase))
	aug_sp = aug.split('_')
	gt = aug.replace('aug%d_original_'%phase, '_groundtruth_(1)_aug%d_'%phase)
	org_file =  aug_sp[2] #+'_'+aug_sp[3]
	i=3
	while('aug%d'%phase not in aug_sp[i]):
		org_file = org_file + '_' + aug_sp[i]
		i=i+1
	org_file = org_file + '_' + aug_sp[i]
	if not os.path.exists(os.path.join(label_folder, org_file.split('.')[0]+'.txt')):
		print(os.path.join(label_path, org_file.split('.')[0]+'.txt'))        

	while('aug%d'%phase not in aug_sp[i]):
		org_file = org_file + '_' + aug_sp[i]
		i=i+1
	org_file = org_file + '_' + aug_sp[i]
	if not os.path.exists(os.path.join(label_folder, org_file.split('.')[0]+'.txt')):
		print(os.path.join(label_path, org_file.split('.')[0]+'.txt'))
 	im = cv2.imread(os.path.join(aug_out_folder,gt))
	f = open(os.path.join(label_folder, org_file.split('.')[0]+'.txt'), 'r')
	lines = f.readlines()
	for i in range(0, len(lines)):
		line_sp = lines[i].split(' ')
		org_aspect = float(int(line_sp[3]) - int(line_sp[1]))/(int(line_sp[4]) - int(line_sp[2]))
		lower = np.array(color_list[i])
		upper = np.array(color_list[i])
		for k in range(0, 3):
			if lower[k] == 0:
				upper[k] = 50
			else:
				upper[k] += 100
				lower[k] -= 100
		mask = cv2.inRange(im, lower, upper)
		imo = cv2.imread(os.path.join(aug_out_folder,aug))
		im3 = cv2.bitwise_and(imo, imo, mask=mask)
		imoo = cv2.resize(im3, (800,600), cv2.INTER_LINEAR)
		im2 = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
		ret, thresh = cv2.threshold(im2, 0, np.max(im2)+1 ,cv2.THRESH_BINARY)
		_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		if len(contours) == 0:
			print "No Cont"
			continue
		x,y,w,h = cv2.boundingRect(contours[0])
		for cnt in contours:
			cx, cy, cw, ch = cv2.boundingRect(cnt)
			if cw*ch > w*h:
				x,y,w,h = cx, cy, cw, ch
		cls_num = np.where(classes == lines[i].split(' ')[0])[0][0]
		after_aspect = float(w)/h
		print org_aspect - after_aspect
		cv2.imshow("test",imoo)
		key = cv2.waitKey(0)
		if key == ord('q'):
			cv2.destroyAllWindows();
			return False
	f.close()
	cv2.destroyAllWindows();
	return True


def drawMaskOld(aug, num):
	label_folder = ('./data/test%d/labels/' % num)
	aug_out_folder = ('./data/test%d/aug/output/' % num)
	aug_sp = aug.split('_')
	gt = aug.replace('aug_original_', '_groundtruth_(1)_aug_')
	org_file =  aug_sp[2] #+'_'+aug_sp[3]
	i=3
	while('aug' not in aug_sp[i]):
		org_file = org_file + '_' + aug_sp[i]
		i=i+1
	org_file = org_file + '_' + aug_sp[i]
	if not os.path.exists(os.path.join(label_folder, org_file.split('.')[0]+'.txt')):
		print(os.path.join(label_path, org_file.split('.')[0]+'.txt'))        

	while('aug' not in aug_sp[i]):
		org_file = org_file + '_' + aug_sp[i]
		i=i+1
	org_file = org_file + '_' + aug_sp[i]
	if not os.path.exists(os.path.join(label_folder, org_file.split('.')[0]+'.txt')):
		print(os.path.join(label_path, org_file.split('.')[0]+'.txt'))
 	im = cv2.imread(os.path.join(aug_out_folder,gt))
	f = open(os.path.join(label_folder, org_file.split('.')[0]+'.txt'), 'r')
	lines = f.readlines()
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
		imo = cv2.imread(os.path.join(aug_out_folder,aug))
		im3 = cv2.bitwise_and(imo, imo, mask=mask)
		imoo = cv2.resize(im3, (800,600), cv2.INTER_LINEAR)
		im2 = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
		ret, thresh = cv2.threshold(im2, 0, np.max(im2)+1 ,cv2.THRESH_BINARY)
		_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		if len(contours) == 0:
			print "No Cont", i, aug
			#imoo = cv2.resize(mask, (800,600), cv2.INTER_LINEAR)
			#cv2.imshow("test",imoo)
			#key = cv2.waitKey(0)
			#if key == ord('q'):
				#cv2.destroyAllWindows()
				#return False
			continue
		x,y,w,h = cv2.boundingRect(contours[0])
		for cnt in contours:
			cx, cy, cw, ch = cv2.boundingRect(cnt)
			if cw*ch > w*h:
				x,y,w,h = cx, cy, cw, ch
		cls_num = np.where(classes == lines[i].split(' ')[0])[0][0]

		after_aspect = float(w)/h
		
		imW = im.shape[1]
		imH = im.shape[0]
		#print orgW-w, float(max(orgW,w))/min(orgW,w), orgH-h, float(max(orgH,h))/min(orgH,h)
		if abs(org_aspect - after_aspect)>=0.5 and (x==0 or (x+w) >= imW or y==0 or (y+h) >= imH):
		#if (abs(orgW - w)>=0.3 or abs(orgH - h)>=0.3) and (x==0 or (x+w) >= imW or y==0 or (y+h) >= imH):
			print org_aspect - after_aspect
			#print orgW-w, orgH-h
			cv2.imshow("test",imoo)
			key = cv2.waitKey(0)
			if key == ord('q'):
				cv2.destroyAllWindows();
				return False
	f.close()
	cv2.destroyAllWindows();
	return True

num = 6
aug_out_folder = ('./data/test%d/aug/output/' % num)
files = [f for f in os.listdir(aug_out_folder) if os.path.isfile(os.path.join(aug_out_folder,f))]
augs = []
gts = []
for fi in files:                                                                                 
	if 'original' in fi:    
		augs.append(fi)
	else:
		gts.append(fi)

for aug in augs:                                                                                 
	if not drawMaskOld(aug, num):
		break


num = 8
dataset = 'MTSD'
phase = 1

aug_out_folder = ('./data/%s_test%d/aug%d/output/' % (dataset, num, phase))
files = [f for f in os.listdir(aug_out_folder) if os.path.isfile(os.path.join(aug_out_folder,f))]
augs = []
gts = []
for fi in files:                                                                                 
	if 'original' in fi:    
		augs.append(fi)
	else:
		gts.append(fi)

for aug in augs:                                                                                 
	if not drawMask(aug, num, dataset, phase):
		break



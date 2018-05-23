import os
import cv2
import time
import progressbar

CLASSES = ('__background__',  # always index 0
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
                         'divided-road-ending', 'curve-on-the-left', 'crossroads-Y-junction')

f_path = "./results/gt_final.txt" #"./data/MTSD/GT.txt"

f = open(f_path,"rt")
i = 0

my_dict = {}
#my_dict_shape = {}
lines = f.readlines()
with progressbar.ProgressBar(maxval=len(lines)) as bar:
    for i, line in enumerate(lines):
        #print i
        split = line.split(';')
	
        if not my_dict.has_key(split[0]):
            #img = cv2.imread(os.path.join('data/MTSD/Images/', split[0]))
            my_dict.update({split[0]:[]})
            #my_dict_shape.update({split[0]:img.shape})


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
print 'Done...'
print len(my_dict.items()), my_dict.items()
#f_out.write(str(fn+";"+x+";"+y+";"+str(int(x)+int(w))+";"+str(int(y)+int(h))+";"+cls))
f.close()

#del my_dict_shape

for i, it in enumerate(my_dict):
    #print i, it
    f_path_label_out = "./data/eval/labels/{}.txt".format('.'.join(it.split('.')[0:-1]))
    f_label_out = open(f_path_label_out, 'wt')
    for it2 in my_dict[it]:
        print(str("{};{};{};{};{}\n".format(CLASSES[it2[0]+1],it2[1],it2[2],it2[3],it2[4])))
        f_label_out.write(str("{} {} {} {} {}\n".format(CLASSES[it2[0]+1],it2[1],it2[2],it2[3],it2[4])))
    f_label_out.close()

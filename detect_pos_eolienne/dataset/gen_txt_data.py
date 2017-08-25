import json
import os
import sys
import cv2


data_dir = "data/"
h,w = 2448.0,3264.0 ## image shape

files = os.listdir(data_dir)

f = open('labelsxdata.txt','w')
for fc in files:
    if fc[-4:] == '.png':
        print(fc)
        with open(os.path.join(data_dir,fc[:-4]+"_refactored.json")) as json_data:
            d = json.load(json_data)
            f.write(
        str(d['points2D']['blade1Tip'][0]/w)+" "+
        str(d['points2D']['blade1Tip'][1]/h)+" "+
        str(d['points2D']['blade2Tip'][0]/w)+" "+
        str(d['points2D']['blade2Tip'][1]/h)+" "+
        str(d['points2D']['blade3Tip'][0]/w)+" "+
        str(d['points2D']['blade3Tip'][1]/h)+" "+
        str(d['points2D']['hubCenter'][0]/w)+" "+
        str(d['points2D']['hubCenter'][1]/h)+" "+
        str(d['points2D']['mastBottom'][0]/w)+" "+
        str(d['points2D']['mastBottom'][1]/h)+" "+
        str(d['points2D']['mastTop'][0]/w)+" "+
        str(d['points2D']['mastTop'][1]/h)+" "+
        os.path.abspath(os.path.join(data_dir,fc)+"\n"))
f.close()

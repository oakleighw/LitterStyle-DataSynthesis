import cv2
import os
from tqdm import tqdm

class dashlit:
    #takes path to images and labels (YOLO[8] format)
    def __init__(self,im_p,lab_p):
        self.im_path = im_p
        self.lab_path = lab_p

        self.litIms = []

    #gets dashlit samples based on index slice
    def getDashlit(self,strt,nd):
        batch = os.listdir(self.lab_path)[strt:nd]

        for ann_f in tqdm(batch):
            #get annotation path
            ann = os.path.join(self.lab_path,ann_f)

            #get image name from annotation file name
            im_name = ann_f.split('.')[:-1]
            im_name.append('jpg') #get image name from text name
            im_name = ".".join(im_name)

            #read image
            full_im_p = os.path.join(self.im_path,im_name)
            gv_im = cv2.cvtColor(cv2.imread(full_im_p),cv2.COLOR_BGR2RGB)

            imh = gv_im.shape[0]
            imw = gv_im.shape[1]

            #read annotation file (yolo)
            ann_txt = open(ann, 'r')
            lines = ann_txt.readlines() # read txt file
            ann_txt.close()

            #get annotation from each line
            for l in lines:
                no_nl = l.strip() # get rid of \n
                annotation = no_nl.split(' ')[1:] # split by space and remove 'class' (only one label)
                #unnormalise 
                xcen = float(annotation[0])*imw; ycen = float(annotation[1])*imh; w = float(annotation[2])*imw; h = float(annotation[3])*imh
                #convert centers to top left of bbox
                xmin = xcen-(w/2); ymin = ycen-(h/2) 
                cropped_img = gv_im[int(ymin):int(ymin+h), int(xmin):int(xmin+w),  :]
                self.litIms.append(cropped_img)
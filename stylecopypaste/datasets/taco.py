from pycocotools.coco import COCO
import cv2
import os
from tqdm import tqdm

#loads samples from taco dataset and crops them according to bounding box
class taco:
    def __init__(self,p):
        self.p = p # path to taco data
        self.litIms = []
        self.litMasks = []

    #takes taco data path and im ids as an array
    #returns cropped litter images and masks to taco object
    def getTaco(self, imIds):   
        ann_path = os.path.join(self.p, "annotations.json")
        coco = COCO(ann_path)
        
        for imId in tqdm(imIds): #for each image selected
            #get image object from coco annotations
            img = coco.imgs[imId]
            
            #get file name and join it to data path
            imp = os.path.join(self.p, img['file_name'])
            
            #get category ids
            cat_ids = coco.getCatIds()
            #get annotation ids
            anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(anns_ids)
            
            #load taco image
            im = cv2.cvtColor(cv2.imread(imp),cv2.COLOR_BGR2RGB)
            
            #for each annotation....
            
            for i, ann in enumerate (anns):
                mask = coco.annToMask(anns[i])

                #crop image and mask
                img_h,img_w,c = im.shape


                [x,y,w,h] = anns[i]['bbox']
                cropped_img = im[int(y):int(y+h), int(x):int(x+w),  :]
                cropped_mask = mask[int(y):int(y+h), int(x):int(x+w)]

                #Cutout rubbish using mask
                cut_img = cv2.bitwise_and(cropped_img,cropped_img,mask = cropped_mask)
            
            #add to array of samples if not None
            if type(cut_img) != None  and len(cropped_mask) != 0:
                self.litIms.append(cut_img)
                self.litMasks.append(cropped_mask)
                

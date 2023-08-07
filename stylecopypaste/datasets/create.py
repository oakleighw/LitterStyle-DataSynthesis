#Program which creates new yolo datasets
import os
from . import taco
from .dashlit import dashlit
import cv2
import gc
import torch
from tqdm.notebook import tqdm
import random

from ..copypaste.paste import rand_paste, points_paste
from ..points.centrepoints import points
from ..points.centrepoints import segPoints
from ..segment.roi import SAM
from ..style.transfer import t_model






"""
placement: 'random', 'points' (along training points) or 'seg' (segmented region)
style: Path to style weights. Default = None.
src: source to get images and annotations  (contains "images" and "labels" folder)
dest: place to save new images and annotations
pts: points(optional) path to annotation csv
"""

class dataset:
    def __init__(self,src, dest, placement,style = None, pts=None):
        self.placement = placement

        self.style = style

        self.images_dir = os.path.join(src,"images")
        self.labels_dir = os.path.join(src,"labels")

        #get images and labels file paths as arrays
        self.images = [os.path.join(self.images_dir,imp) for imp in os.listdir(self.images_dir)]
        self.labels = [os.path.join(self.labels_dir,labp) for labp in os.listdir(self.labels_dir)]

        self.dest = dest
        
        self.dest_images_dir = os.path.join(dest,"images")
        self.dest_labels_dir = os.path.join(dest,"labels")

        #structures to contain taco processed ims and masks
        self.litIms = []
        self.litMasks = []

        
        #if points provided
        if pts is not None:
            #load from csv path given
            self.points = points(pts)
            self.kx, self.ky = self.points.getClusters(40,0)
        
        #if style path provided
        if self.style is not None:
            #load adaIN model
            self.model = t_model(imsize= 96,weightspath= style)

    #creates new dataset
    def generate(self,litPath):
        
        #create new directories if not exist
        if not os.path.exists(self.dest):
            # Create a new directory because it does not exist
            os.makedirs(self.dest)

        #checks if directory empty
        if len(os.listdir(self.dest)) != 0:
            print("Destination directory contains items. Cannot create dataset here.")
            return

        #create sub directories ("images" & "labels")
        os.makedirs(self.dest_images_dir)
        os.makedirs(self.dest_labels_dir)

        #get taco images and masks
        if len(self.litIms) == 0 and len(self.litMasks) == 0:
            self.__load_taco__(litPath)

        #set up SAM model if segmentation-placement
        if self.placement =="seg":
            sam = SAM(checkpoint = "sam_vit_h_4b8939.pth", model_type = "vit_h") #Add "cuda" here if on gpu!

        #load dashlit data if using style
        if self.style is not None:
            print("Loading Dashlit Data...")

            #get dashlit litter samples.
            dash = dashlit(self.images_dir,self.labels_dir)

            dash.getDashlit(0,10000) # range can only be as big as litter instances in training data

        print("Generating Images (Wait while I litter)...")

        


        for f in tqdm(self.images):
            #############get random selection here###############

            taco_lit_ims = []
            taco_masks = []

            #get random selection of taco litter (between 2-7 pieces)
            amnt = random.randint(2,7)

            for t in range(amnt):
                #for each of the taco litter piece, select a random one from the dataset
                taco_choice = random.choice(range(len(self.litIms)))
                taco_lit_ims.append(self.litIms[taco_choice])
                taco_masks.append(self.litMasks[taco_choice])




            #############Perform style transfer###################
            if self.style is not None:
            ###########################
                
                ###change this when testing with complete dataset (cannot be larger than litter in images)
                
                
                lit_ims = [] #these will be the litters actually pasted if style == true
                lit_masks = []
                
                #generate new style image - random style from dashlit, random selection from 
                for i in range(amnt): 
                    #get random style from dashlit pieces
                    dashstyle_index = random.randint(0,len(dash.litIms)-1)
                    nst_im, nst_mask = self.model.generate_style_data(dash.litIms[dashstyle_index],taco_lit_ims[i],taco_masks[i])
                    lit_ims.append(nst_im)
                    lit_masks.append(nst_mask)
            
            else:
                lit_ims = taco_lit_ims #these will be the litters actually pasted if style == False
                lit_masks = taco_masks



            #############Perform image ops############

            background = cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2RGB)




            #perform image synthesis based on flag
            if self.placement =="random":
                #perform random paste
                try:
                    result,xs,ys,ws,hs = rand_paste(lit_ims,lit_masks,background,show=False, rotate=True,return_loc=True)
                except:
                    continue # if issue with taco sample


            if self.placement =="points":
                #denormalise points for image
                denormx, denormy = self.points.denorm_centres(background.shape,self.kx,self.ky)
                try:
                    result,xs,ys,ws,hs = points_paste(lit_ims,lit_masks,background,denormx,denormy,show=False, rotate=True,return_loc=True)
                except:
                    continue # if issue with taco sample

            if self.placement =="seg":
                #clear cache
                gc.collect()
                torch.cuda.empty_cache()

                #denormalise points for image
                denormx, denormy = self.points.denorm_centres(background.shape,self.kx,self.ky)
                mask = sam.getMask(background,denormx,denormy) #get mask
                #get point locations from mask
                x,y = segPoints(mask)
                try:
                    result,xs,ys,ws,hs = points_paste(lit_ims,lit_masks,background,x,y,show=False, rotate=True,return_loc=True)
                except:
                    continue # if issue with taco sample
                
            
            #create save path
            head_tail = os.path.split(f)
            new_imp = os.path.join(self.dest_images_dir,head_tail[1])

            result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR) # convert back to bgr for opencv save
            cv2.imwrite(new_imp, result)

            #############Perform label ops############
            label_fname = head_tail[1].split(".")[0] +".txt" #convert to .txt fname
            label_pname = os.path.join(self.labels_dir,label_fname) #get path of related label
            ann_txt = open(label_pname, 'r') # read label file
            lines = ann_txt.readlines() # save lines
            ann_txt.close()

            #add new synthetic annotations (class 0 as always one class 'litter')
            for i in range(len(xs)):
                lines.append(f"0 {xs[i]} {ys[i]} {ws[i]} {hs[i]}\n")

            #save to new file
            save_label_fname = os.path.join(self.dest_labels_dir, label_fname)
            with open(save_label_fname, 'w') as new_txt:
                new_txt.writelines(lines)

        print("Done!")
                



    
    #loads in all taco data into dataset
    def __load_taco__(self,litPath):
        lits = taco.taco(litPath)
        #ids = range(1500) # get all taco dataset
        ids = range(1500) #15 for testing                           ###NEED TO UPDATE THIS TO A RANDOM SELECTION PER IMAGE WITH FULL DATASET.
        print("Loading Taco Data...")
        lits.getTaco(ids)
        self.litIms = lits.litIms
        self.litMasks = lits.litMasks





    
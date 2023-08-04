#Program which creates new yolo datasets
import os
from . import taco
from . import dashlit
import cv2

from ..copypaste.paste import rand_paste

"""
placement: 'random', 'points' (along training points) or 'seg' (segmented region)
style: bool, transfer style or not. Default = False.
src: source to get images and annotations  (contains "images" and "labels" folder)
dest: place to save new images and annotations
"""

class dataset:
    def __init__(self,src, dest, placement,style = False):
        self.placement = placement
        self.style = style

        images_dir = os.path.join(src,"images")
        self.labels_dir = os.path.join(src,"labels")

        #get images and labels file paths as arrays
        self.images = [os.path.join(images_dir,imp) for imp in os.listdir(images_dir)]
        self.labels = [os.path.join(self.labels_dir,labp) for labp in os.listdir(self.labels_dir)]

        self.dest = dest
        
        self.dest_images_dir = os.path.join(dest,"images")
        self.dest_labels_dir = os.path.join(dest,"labels")


        self.litIms = []
        self.litMasks = []
    
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

        #place all randomly
        if self.placement =="random":
            for f in self.images:

                #############Perform image ops############
                background = cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2RGB)

                #perform random paste
                result,xs,ys,ws,hs = rand_paste(self.litIms,self.litMasks,background,show=False, rotate=True,return_loc=True)
                
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
                



    
    #loads in all taco data into dataset
    def __load_taco__(self,litPath):
        lits = taco.taco(litPath)
        #ids = range(1500) # get all taco dataset
        ids = range(15) #15 for testing                           ###NEED TO UPDATE THIS TO A RANDOM SELECTION PER IMAGE WITH FULL DATASET.
        print("Loading Taco Data...")
        lits.getTaco(ids)
        self.litIms = lits.litIms
        self.litMasks = lits.litMasks



    #generates new image
    def generate_im(self,imp):
        pass

    
    #generates new label
    def generate_label(self,labp):
        pass
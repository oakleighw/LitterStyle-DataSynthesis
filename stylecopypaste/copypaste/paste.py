#contains functions for resizing samples and pasting 
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
#Resizes an image (such as litter instance) - keeps ratio if only width or only height is supplied
#Returns resized sample, new width and new height

def instance_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image (e.g.litter sample) to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    
    new_w = dim[0]
    new_h = dim[1]
    
    # return the resized image
    return resized, new_w, new_h


#combine litter and subsection of an image -> x1 and y1 are top-left of litter bounding box (top left cropped sample)
def combine(litter_im, litter_mask, verge, x1, y1, rotate = False):
    #Add images together
    
    #if rotate ==True, randomly rotate (possibility of no rotation '0' degrees)
    if rotate:
        angle = random.choice([-90,0,90,180])
        if angle == -90:
            litter_im = cv2.rotate(litter_im,cv2.ROTATE_90_COUNTERCLOCKWISE)
            litter_mask = cv2.rotate(litter_mask,cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 90:
            litter_im = cv2.rotate(litter_im,cv2.ROTATE_90_CLOCKWISE)
            litter_mask = cv2.rotate(litter_mask,cv2.ROTATE_90_CLOCKWISE)
        elif angle ==180:
            litter_im = cv2.rotate(litter_im,cv2.ROTATE_180)
            litter_mask = cv2.rotate(litter_mask,cv2.ROTATE_180)
            
    #https://www.binarystudy.com/2022/09/How-to-add-subtract-different-size-images-using-OpenCV-Python.html#:~:text=Alternatively%20you%20can%20use%20cv2.addWeighted%20%28%29%20to%20add,cv2.addWeighted%20%28img11%2C%200.3%2C%20img22%2C%200.7%2C%200%29%20plt.imshow%20%28img_add%29

    # Find the minimum height and width of the two images
    min_height = litter_im.shape[0]
    min_width = litter_im.shape[1]

    # Crop images with minimum height and width (object height/width)
    foreground = litter_im
    background = verge[y1-min_height:y1,x1:x1+min_width]
    
    (Rb, Gb, Bb) = cv2.split(background)
    (Rf, Gf, Bf) = cv2.split(foreground)


    #invert mask
    inv_mask = 1-litter_mask


    #In each colour channel, cut out mask and add foreground channels
    
    Rn = cv2.bitwise_and(Rb,Rb,mask=inv_mask) + Rf
    Gn = cv2.bitwise_and(Gb,Gb,mask=inv_mask) + Gf
    Bn = cv2.bitwise_and(Bb,Bb,mask=inv_mask) + Bf

    #merge channels back together
    merged = cv2.merge([Rn,Gn,Bn])
    
    verge[y1-min_height:y1,x1:x1+min_width] = merged #paste created background/litter combination into full image
    
    if rotate: #return dimensions if rotating as may be altered
        return verge, min_width, min_height #min width and min height are returned incase of rotation (Swaps points)
    else:
        return verge

#paste samples randomly onto new image
def rand_paste(sampIms,sampMasks,background, show=False, rotate= False):
    vh = background.shape[0]; vw = background.shape[1]
    merged = background.copy()
    

    if show:
        fig, ax = plt.subplots()

    for i, litSamp in enumerate(sampIms):
        
        #resize depending on largest side
        if litSamp.shape[0] > litSamp.shape[1]:
            new_lit_height = random.randint(30,70)# height for litter resizing
            #resize litter to height
            context_cut, w , h = instance_resize(sampIms[i],height=new_lit_height) ; context_mask,_,_ = instance_resize(sampMasks[i],height=new_lit_height)
        else:
            new_lit_width = random.randint(30,70)# height for litter resizing
            #resize litter to height
            context_cut, w , h = instance_resize(sampIms[i],width=new_lit_width) ; context_mask,_,_ = instance_resize(sampMasks[i],width=new_lit_width)

        safety = h if h > w else  w # makes sure object does not fall outside of boundary when rotated

        #random locations (top-left of bounding box)
        locx = random.randint(safety, vw-safety) 
        locy = random.randint(safety, vh-safety)



        #combine with background
        if rotate:
            merged, w, h = combine(context_cut, context_mask, merged, locx , locy, rotate=True)      
        else:
            merged = combine(context_cut, context_mask, merged, locx , locy)
            

        
        if show:
            rect = patches.Rectangle((locx, locy-h), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)



    if show:
        plt.title("Randomly-placed samples")
        plt.axis('off')
        plt.imshow(merged)

    return merged


def training_paste(sampIms,sampMasks,background, xpoints, ypoints, show=False, rotate= False):
    vh = background.shape[0]; vw = background.shape[1]
    merged = background.copy()
    
    pointlen = len(xpoints) # length for generating random locations from training data

    if show:
        fig, ax = plt.subplots()

    for i, litSamp in enumerate(sampIms):
        
        #resize depending on largest side
        if litSamp.shape[0] > litSamp.shape[1]:
            new_lit_height = random.randint(30,70)# height for litter resizing
            #resize litter to height
            context_cut, w , h = instance_resize(sampIms[i],height=new_lit_height) ; context_mask,_,_ = instance_resize(sampMasks[i],height=new_lit_height)
        else:
            new_lit_width = random.randint(30,70)# height for litter resizing
            #resize litter to height
            context_cut, w , h = instance_resize(sampIms[i],width=new_lit_width) ; context_mask,_,_ = instance_resize(sampMasks[i],width=new_lit_width)

        safety = h if h > w else  w # makes sure object does not fall outside of boundary when rotated

        rand = random.randint(0,pointlen) #random points from training data
        #random locations (top-left of bounding box)

        locx = int(xpoints[rand]-(w/2))#to place in centre, need to provide top-left bbox point (-w/2, +h/2)
        locy = int(ypoints[rand]+(h/2)) 

        if locx < safety: locx = 0 # to make space for bigger items, make sure x nor y are 0 (no edge items)
        if locy < safety: locy = 0 

        


        #combine with background
        if rotate:
            merged, w, h = combine(context_cut, context_mask, merged, locx , locy, rotate=True)      
        else:
            merged = combine(context_cut, context_mask, merged, locx , locy)
            

        
        if show:
            rect = patches.Rectangle((locx, locy-h), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)



    if show:
        plt.title("Samples placed in familiar training locations")
        plt.axis('off')
        plt.imshow(merged)

    return merged

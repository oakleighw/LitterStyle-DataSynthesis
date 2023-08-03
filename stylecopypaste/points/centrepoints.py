#deals with centrepoint data seen in training
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2

class points:
    def __init__(self,p):
        self.labelCSV = pd.read_csv(p) # path to csv annotations data
        self.norm_x, self.norm_y = self.__getCentres__(self.labelCSV)

    #gets centres from bounding box coordinates
    def __getCentres__(self,csv_f):
        #x and y coords of centres
        centx = []
        centy = []

        xmin = csv_f["xmin"].values
        ymin = csv_f["ymin"].values
        xmax = csv_f["xmax"].values
        ymax = csv_f["ymax"].values
        width = csv_f["width"].values #width and height of IMAGES
        height = csv_f["height"].values

        #calculate bbox centroids from csv info
        for i, name in enumerate(xmin):
            xcen = float((xmin[i] + xmax[i])) / 2 / width[i] 
            ycen = float((ymin[i] + ymax[i])) / 2 / height[i]
            
            centx.append(xcen)
            centy.append(ycen)

        return centx,centy
    
    #returns points projected onto an image size, converts to integer
    def denorm_centres(self, imShape, pointsx=None, pointsy= None):
        if pointsx is None or pointsy is None:
            denormX = np.multiply(self.norm_x,imShape[1])
            denormY = np.multiply(self.norm_y,imShape[0])
        else:
            denormX = np.multiply(pointsx,imShape[1])
            denormY = np.multiply(pointsy,imShape[0])

        return np.rint(denormX).astype(int), np.rint(denormY).astype(int)
    
    #performs k means clustering
    def getClusters(self,k,random_state = None):
        #join points from both arrays into tuple
        p_tuple = [(self.norm_x[i], self.norm_y[i]) for i in range(0, len(self.norm_x))]

        #perform k means clustering of points
        if random_state is None:
            kmeans = KMeans(n_clusters=k).fit(p_tuple)
        else:
            kmeans = KMeans(n_clusters=k, random_state=random_state).fit(p_tuple)
        
        kCents = kmeans.cluster_centers_
        x,y = zip(*kCents)

        return x,y
    
    #show heatmap of centres
    def show_centres(self):
        heatmap, xedges, yedges = np.histogram2d(self.norm_x, self.norm_y, bins=100)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        fig, ax = plt.subplots()
        plt.clf()
        plt.imshow(heatmap.T, extent=extent)
        plt.title("Normalised Centre-Point Heatmap")
        plt.show()

    #show points overlaid on image
    def show_centres_overlay(self,image,pointsx=None, pointsy= None, title = None):
        if pointsx is None or pointsy is None:
            pointsx = self.norm_x
            pointsy = self.norm_y
        
        plotted_im = image.copy()
        for i in range(0,len(pointsx)):
            cv2.circle(plotted_im, (int(pointsx[i]),int(pointsy[i])), radius=10, color=(0, 255, 0), thickness=-1)

        plt.axis('off')
        if title is not None:
            plt.title(title)
        plt.imshow(plotted_im)

#getting points from segmented mask
def segPoints(mask):
    mpy, mpx = np.where(mask==1)
    return mpx,mpy
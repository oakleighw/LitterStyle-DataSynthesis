#deals with centrepoint data seen in training
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    def denorm_centres(self, imShape):
        denormX = np.multiply(self.norm_x,imShape[1])
        denormY = np.multiply(self.norm_y,imShape[0])
        return np.rint(denormX).astype(int), np.rint(denormY).astype(int)
    
    #show heatmap of centres
    def show_centres(self):
        heatmap, xedges, yedges = np.histogram2d(self.norm_x, self.norm_y, bins=100)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        fig, ax = plt.subplots()
        plt.clf()
        plt.imshow(heatmap.T, extent=extent)
        plt.title("Normalised Centre-Point Heatmap")
        plt.show()
        
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import cv2

class SAM:
    #set up model
    def __init__(self,checkpoint, model_type, device = None):
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)

        if device is not None:
            self.sam.to(device=device)

        self.predictor = SamPredictor(self.sam)

    #uses SAM model with points as prompt to return mask. Erodes to remove noise
    def getMask(self,image, pointsx,pointsy):
        #assign predictor to image
        self.predictor.set_image(image)

        input_point = np.array([[i,j] for i,j in zip(pointsx,pointsy)])
        input_label = np.array([1]*len(pointsx)) #all labeled with '1' (Verge)

        mask, score, logits = self.predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )
        mask =np.squeeze(mask)

        #erode to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        eroMask =cv2.erode(mask.astype('uint8'), kernel, iterations=5)
        return eroMask
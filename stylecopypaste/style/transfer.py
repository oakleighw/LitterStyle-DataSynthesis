from . import adain_model
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

class t_model:
    #initialise model with image size & weights
    def __init__(self,imsize,weightspath):
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-5)
        self.loss_fn = keras.losses.MeanSquaredError()

        self.IMAGE_SIZE = imsize #96

        self.encoder = adain_model.get_encoder(self.IMAGE_SIZE)
        self.loss_net = adain_model.get_loss_net(self.IMAGE_SIZE)
        self.decoder = adain_model.get_decoder()

        self.model = adain_model.NeuralStyleTransfer(
        encoder=self.encoder, decoder=self.decoder, loss_net=self.loss_net, style_weight=2.0
        )

        self.model.compile(optimizer=self.optimizer, loss_fn=self.loss_fn)
        self.model.load_weights(weightspath)

    #transfer style onto content
    def transferStyle(self,style,content):
        #resize accordingly
        style = adain_model.decode_and_resize(style,self.IMAGE_SIZE)
        content = adain_model.decode_and_resize(content,self.IMAGE_SIZE)

        #encode and merge using model
        style_encoded = self.encoder(style)
        content_encoded = self.encoder(content)
        t = adain_model.ada_in(style=style_encoded, content=content_encoded)
        reconstructed_image = self.decoder(t)
    
        return tf.squeeze(reconstructed_image)# removes extra dim
    

    #generates neural style transfer image and new mask
    def generate_style_data(self,style,content,mask):
        og_size = content.shape
        style = np.array(style)
        style = tf.expand_dims(tf.convert_to_tensor(style),axis=0)

        content = np.array(content)
        content = tf.expand_dims(tf.convert_to_tensor(content),axis=0)

        nst_im = self.transferStyle(style,content).numpy()

        #resize image back to normal after passing through model
        lit= cv2.resize(nst_im, (og_size[1],og_size[0]))

        lit = lit*255 #denormalise
        lit = cv2.bitwise_and(lit,lit,mask = mask)

        return lit

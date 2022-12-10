from tkinter import *
from tkinter import ttk
from tkinter.font import BOLD
from webbrowser import BackgroundBrowser 
from PIL import Image, ImageTk
import os 
from keras import utils
from tensorflow import keras
import cv2
import keras
import tensorflow as tf
from keras.models import load_model
# from keras.preprocessing.image import img_to_array
# from tensorflow.keras.utils import img_to_array
#from keras_preprocessing.image import img_to_array
# tf.keras.utils.load_img
# from tf.keras.utils import img_to_array
import numpy as np

model= load_model("Face_Mask17.h5")

img_width, img_height= 200, 200

class Face_Mask_Detection:
    def __init__(self, root):
        self.root = root 
        self.root.geometry("1285x700+0+0")
        self.root.title("Face Mask Detection System")

         # First Image 
        img= Image.open(r"images/3.jpg")
        img = img.resize((350,700), Image.ANTIALIAS)
        self.photoimg = ImageTk.PhotoImage(img)

        first_label = Label(self.root, image= self.photoimg)
        first_label.place(x=500, y= 0, width=350, height=700)

        # Second Image 
        img2= Image.open(r"images/4.jpg")
        img2 = img2.resize((500,700), Image.ANTIALIAS)
        self.photoimg2 = ImageTk.PhotoImage(img2)

        second_label = Label(self.root, image= self.photoimg2)
        second_label.place(x=0, y= 0, width=500, height=700)

        # Third Image 
        img3= Image.open(r"images/5.jpg")
        img3 = img3.resize((500,700), Image.ANTIALIAS)
        self.photoimg3 = ImageTk.PhotoImage(img3)

        third_label = Label(self.root, image= self.photoimg3)
        third_label.place(x=850, y= 0, width=450, height=700)

        # Title 
        title_label = Label(self.root, text= "Face Mask Detection",
        font= ("times new roman", 25, BOLD), bg="darkblue", fg= "red")
        title_label.place(x=0, y=0, width=1285, height=45)

        # Button
        b1_1 = Button(first_label, text= "Detect Your Face Mask", command= self.face_mask_detection,cursor= "hand2",
        font= ("times new roman", 15, BOLD), bg="red", fg= "white")
        b1_1.place(x = 70, y= 560, width= 220, height= 35)


    # Face Mask Detection

    def face_mask_detection(self):
        model= load_model("Face_Mask17.h5")

        img_width, img_height= 200, 200

        face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        cap = cv2.VideoCapture(0)

        img_count_full= 0

        while True:
            img_count_full += 1
            response, color_img= cap.read()
            
            scale= 100
            width= int(color_img.shape[1] * scale/100)
            height= int(color_img.shape[0] * scale/100)
            dim= (width, height)
            
            # Resize image
            gray_img= cv2.resize(color_img, dim, cv2.COLOR_BGR2GRAY)
            
            # Detect the faces
            faces= face_cascade.detectMultiScale(gray_img, 2)
            
            img_count= 0
            for (x,y,w,h) in faces:
                color_face= color_img[y:y+h, x:x+w] # To extract the face from its specific location
                # img= cv2.resize(color_face, (200, 200)) 
                # To save the images
                cv2.imwrite("Saved images/%d%dface.jpg"%(img_count_full, img_count), color_face)
                img= tf.keras.utils.load_img("Saved images/%d%dface.jpg"%(img_count_full, img_count), target_size= (200,200))
                img= np.array(img)/255
                img= np.expand_dims(img, axis= 0)
                pred_prob= model.predict(img)
                pred= np.argmax(pred_prob)
                
                if pred== 0:
                    img_rect= cv2.rectangle(color_img, (x,y), (x+w, y+h), (255, 0, 0), 3)
                    cv2.putText(img_rect, "Mask", (x-10,y-10), 2, 1, (0,255,0), 2, cv2.LINE_AA)
                else:
                    img_rect= cv2.rectangle(color_img, (x,y), (x+w, y+h), (255, 0, 0), 3)
                    cv2.putText(img_rect, "No Mask", (x-10,y-10), 2, 1, (0,0,255), 2, cv2.LINE_AA)
                    
                # display image
                cv2.imshow("Face Mask Detection", color_img)
                
                if cv2.waitKey(2) == 27:
                    break
                    
        cap.release()
        cv2.destroyAllWindows()
        
        


if __name__ == "__main__":
    root = Tk()
    obj= Face_Mask_Detection(root)
    root.mainloop()
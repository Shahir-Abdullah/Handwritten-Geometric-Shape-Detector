import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import time 
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter import *
from tkinter import ttk, colorchooser
# needs Python Image Library (PIL)
from PIL import Image, ImageDraw
#Import all the enhancement filter from pillow
from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
)

#model loading
model = tf.keras.models.load_model('shapedetector_model.h5')

#show the model architecture 
#shapedetector_model.summary()

class_names = ['circle', 'rectangle', 'square', 'triangle']

img_height = 28
img_width = 28



class main:
    def __init__(self,master):
        self.master = master
        self.color_fg = 'black'
        self.color_bg = 'white'
        self.old_x = None
        self.old_y = None
        self.penwidth = 5
        self.drawWidgets()
        self.c.bind('<B1-Motion>',self.paint)#drwaing the line 
        self.c.bind('<ButtonRelease-1>',self.reset)
        self.image1 = Image.new("RGB", (500, 500), self.color_bg)
        self.draw = ImageDraw.Draw(self.image1)
        self.label = " "
        self.result = tk.Label(self.master,text= self.label).pack()
    def paint(self,e):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x,self.old_y,e.x,e.y,width=self.penwidth,fill=self.color_fg,capstyle=ROUND,smooth=True)
            self.draw.line([self.old_x,self.old_y,e.x,e.y], self.color_fg)

        self.old_x = e.x
        self.old_y = e.y

    def reset(self,e):    #reseting or cleaning the canvas 
        self.old_x = None
        self.old_y = None      

    def changeW(self,e): #change Width of pen through slider
        self.penwidth = e

    def predict(self): #save the image and push it to model prediction
        # PIL image can be saved as .png .jpg .gif or .bmp file
        filename = "predictiondata/drawn.png"
        self.image1 = self.image1.resize((28, 28))
        self.image1 = self.image1.filter(EDGE_ENHANCE_MORE)
        
        self.image1.save(filename)  
        #time.sleep(1)
        #loading for prediction
        img = keras.preprocessing.image.load_img(
        filename, target_size=(img_height, img_width))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        #time.sleep(1)
        #prediction
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])  
        self.label = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
        self.result["text"] = self.label 
        #print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))  

    def clear(self):
        self.c.delete(ALL)
        os.remove("predictiondata/drawn.png")
        self.image1 = Image.new("RGB", (500, 500), self.color_bg)
        self.draw = ImageDraw.Draw(self.image1)
        self.result["text"] = " "
        

    def change_fg(self):  #changing the pen color
        self.color_fg=colorchooser.askcolor(color=self.color_fg)[1]

    def change_bg(self):  #changing the background color canvas
        self.color_bg=colorchooser.askcolor(color=self.color_bg)[1]
        self.c['bg'] = self.color_bg

    def drawWidgets(self):
        self.controls = Frame(self.master,padx = 5,pady = 5)
        Label(self.controls, text='Pen Width:',font=('arial 18')).grid(row=0,column=0)
        self.slider = ttk.Scale(self.controls,from_= 5, to = 100,command=self.changeW,orient=VERTICAL)
        self.slider.set(self.penwidth)
        self.slider.grid(row=0,column=1,ipadx=30)
        self.controls.pack(side=LEFT)
        
        self.c = Canvas(self.master,width=500,height=500,bg=self.color_bg,)
        self.c.pack(fill=BOTH,expand=True)

        # PIL create an empty image and draw object to draw on
        # memory only, not visible
        self.image1 = Image.new("RGB", (500, 500), self.color_bg)
        self.draw = ImageDraw.Draw(self.image1)

        self.label = " "
        self.result = tk.Label(self.master,text=" ").pack()

        class_image = tk.Button(self.master, text='Predict Shape',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=self.predict)
        class_image.pack(side=tk.RIGHT) 
        


        menu = Menu(self.master)
        self.master.config(menu=menu)
        filemenu = Menu(menu)
        colormenu = Menu(menu)
        menu.add_cascade(label='Colors',menu=colormenu)
        colormenu.add_command(label='Brush Color',command=self.change_fg)
        colormenu.add_command(label='Background Color',command=self.change_bg)
        optionmenu = Menu(menu)
        menu.add_cascade(label='Options',menu=optionmenu)
        optionmenu.add_command(label='Clear Canvas',command=self.clear)
        optionmenu.add_command(label='Exit',command=self.master.destroy) 
        #optionmenu.add_command(label='Predict',command=self.predict)
        
        

if __name__ == '__main__':
    root = Tk()
    main(root)
    root.title('Shape Detector')
    root.mainloop()

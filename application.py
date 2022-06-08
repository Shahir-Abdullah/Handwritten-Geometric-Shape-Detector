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

# ML section

#tensoflow model loading
model = tf.keras.models.load_model('model/shapedetector_model_4b.h5')

#show the model architecture 
#shapedetector_model.summary()

class_names = ['circle', 'rectangle', 'square', 'triangle']

img_height = 28
img_width = 28


#colors
label_color="#F1C40F"
#frame_color = "#2471A3"
frame_color = "#F1C40F"
btn_fg_color = "#FFFFFF"
#start_btn_bg = "#0BB8E2"
start_btn_bg = "#E67500"
stop_btn_bg = "#F34545"
reset_btn_bg = "#17B974"
predict_btn_color = "#21D20A"
background_color = "#D19900"
control_bg_color = "#FFFFFF"
#fonts
btn_font = "Courier 15 bold"

#border effects
border_effects = {
    "flat": tk.FLAT,
    "sunken": tk.SUNKEN,
    "raised": tk.RAISED,
    "groove": tk.GROOVE,
    "ridge": tk.RIDGE,
}

class App:
    def __init__(self, master):
        
        self.master = master
        self.color_fg = 'black'
        self.color_bg = 'white'
        self.old_x = None
        self.old_y = None
        self.penwidth = 5
        self.drawWidgets()
        self.canvas.bind('<B1-Motion>',self.paint)#drwaing the line 
        self.canvas.bind('<ButtonRelease-1>',self.reset)
        self.image1 = Image.new("RGB", (500, 500), self.color_bg)
        self.draw = ImageDraw.Draw(self.image1)

    def paint(self,e):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x,self.old_y,e.x,e.y,width=self.penwidth,fill=self.color_fg,capstyle=ROUND,smooth=True)
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
        self.image1 = self.image1.convert('LA')
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
        
        self.lbl_result["text"] = "{}".format(class_names[np.argmax(score)])
        self.lbl_confidence["text"] = "{:.2f}".format(100 * np.max(score))
        #print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))  

    def clear(self):
        self.canvas.delete(ALL)
        os.remove("predictiondata/drawn.png")
        self.image1 = Image.new("RGB", (500, 500), self.color_bg)
        self.draw = ImageDraw.Draw(self.image1)
        self.lbl_result["text"] = " "
        self.lbl_confidence["text"] = " "
        

    def change_fg(self):  #changing the pen color
        self.color_fg=colorchooser.askcolor(color=self.color_fg)[1]

    def change_bg(self):  #changing the background color canvas
        self.color_bg=colorchooser.askcolor(color=self.color_bg)[1]
        self.canvas['bg'] = self.color_bg

    def drawWidgets(self):
        #resizable and responsive window
        self.master.configure(bg=background_color)
        self.master.rowconfigure(0, minsize=500, weight=0) #weight 0 means the elements are non resizable #Take a look at line 6 more closely. The minsize parameter of .rowconfigure() is set to 800 and weight is set to 1. The first argument is 0, which sets the height of the first row to 800 pixels and makes sure that the height of the row grows proportionally to the height of the self.master. There’s only one row in the application layout, so these settings apply to the entire self.master.
        self.master.columnconfigure(0, minsize=500, weight=0) #Here, you use .columnconfigure() to set the width and weight attributes of the column with index 1 to 800 and 1, respectively:. Remember, row and column indices are zero-based, so these settings apply only to the second column. By configuring just the second column, the text box will expand and contract naturally when the self.master is resized, while the column containing the buttons will remain at a fixed width.
        self.master.resizable(False, False) # the main window is non resizable

        #drawing canvas frame
        self.canvas_frame = tk.Frame(
            master=self.master,
            relief = tk.FLAT,
            borderwidth=2,
            bg = 'white'
            
        )
        self.penwidth_frame = tk.Frame(
            master=self.master,
            relief = tk.FLAT,
            padx = 5,
            pady = 5,
            bg = frame_color,
            borderwidth=2
        )
        #label for result
        self.lbl_result_label = tk.Label(master=self.penwidth_frame, text="Result",font=btn_font, bg=label_color)
        self.lbl_result = tk.Label(master=self.penwidth_frame, text=" ",font=btn_font, bg=control_bg_color)
        #label for confidence score 
        self.lbl_confidence_label = tk.Label(master=self.penwidth_frame, text="Confidence",font=btn_font, bg=label_color)
        self.lbl_confidence = tk.Label(master=self.penwidth_frame, text=" ",font=btn_font, bg=control_bg_color)
        #label for width controller
        self.lbl_penwidth = tk.Label(master=self.penwidth_frame, text="Pen Width",font=btn_font, bg=label_color)
        # empty label for padding 
        self.lbl_empty = tk.Label(master=self.penwidth_frame, text=" ", width=5, height=4, bg=frame_color)
        self.lbl_empty1 = tk.Label(master=self.penwidth_frame, text=" ", width=5, height=1, bg=frame_color)
        self.lbl_empty2 = tk.Label(master=self.penwidth_frame, text=" ", width=5, height=1, bg=frame_color)
        #button for clearing canvas
        self.clear_button = tk.Button(
            master = self.penwidth_frame,
            text="CLEAR",
            bg = predict_btn_color,
            fg = btn_fg_color,
            font=(btn_font),
            command=self.clear,
            width=5,
            height=1
        )
        #button for prediction
        self.prediction_button = tk.Button(
            master = self.penwidth_frame,
            text="PREDICT",
            bg = start_btn_bg,
            fg = btn_fg_color,
            font=(btn_font),
            command=self.predict,
            width=5,
            height=1
        )


        #canvas for painting section
        self.canvas = Canvas(self.canvas_frame ,width=500,height=500,bg='white',)
        #slider for controlling pen width
        self.slider = ttk.Scale(self.penwidth_frame,from_= 5, to = 100,command=self.changeW,orient=VERTICAL)
        self.slider.set(self.penwidth)
        
        # PIL create an empty image and draw object to draw on
        # memory only, not visible
        self.image1 = Image.new("RGB", (500, 500), self.color_bg)
        self.draw = ImageDraw.Draw(self.image1)
           
        #geometry manager to set up the window frame
        #canvas frame
        self.canvas.grid(row=0, column=0, sticky="nwse")
        #penwidth frame
        self.lbl_penwidth.grid(row=0, column=0, sticky="nswe", padx=5, pady=7)
        self.slider.grid(row=1, column=0, sticky="nswe", padx=5, pady=7)
        self.lbl_empty1.grid(row=2, column=0, sticky="nswe", padx=5, pady=7)
        self.lbl_result_label.grid(row=3, column=0, sticky="nsew" ,padx=5, pady=7)
        self.lbl_result.grid(row=4, column=0, sticky="nsew",padx=5, pady=7)
        self.lbl_confidence_label.grid(row=5, column=0, sticky="nsew",padx=5, pady=7)
        self.lbl_confidence.grid(row=6, column=0, sticky="nsew",padx=5, pady=7)
        self.lbl_empty2.grid(row=7, column=0, sticky="nsew",padx=5, pady=7)
        self.prediction_button.grid(row=8, column=0, sticky="nswe",padx=4, pady=4)
        self.lbl_empty.grid(row=9, column=0, sticky="nsew",padx=5, pady=7)
        self.clear_button.grid(row=10, column=0, sticky="nswe",padx=4, pady=4)
        #main window
        self.canvas_frame.grid(row=0, column=0, sticky="nswe", padx=5, pady=7)
        self.penwidth_frame.grid(row=0, column=1, sticky="nswe", padx=5, pady=7)
        
        
        menu = Menu(self.master)
        self.master.config(menu=menu)
        filemenu = Menu(menu)
        colormenu = Menu(menu)
        menu.add_cascade(label='Colors',menu=colormenu)
        colormenu.add_command(label='Brush Color',command=self.change_fg)
        colormenu.add_command(label='Background Color',command=self.change_bg)
        
        '''optionmenu = Menu(menu)
        menu.add_cascade(label='Options',menu=optionmenu)
        optionmenu.add_command(label='Clear Canvas',command=self.clear)
        optionmenu.add_command(label='Exit',command=self.master.destroy) '''
        
       


# main function
if __name__ == "__main__":
    #window initialization
    window = tk.Tk()
    #title of the window
    window.title("Shape Detector")
    a = App(window)
    #without mainloop, nothing will be shown
    window.mainloop()

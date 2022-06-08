# Handwritten-Geometric-Shape-Detector

# Description :

I created a python desktop application. This application allows the user to draw any geometric shape. Then the python script takes the shape and feeds the picture into a machine learning model. I used CNN model. The model is trained under 2000 images of handwritten circle, square, rectangle, triangle. The model predicts the shape of the drawn image and tells the user the shape and how confident the model is about the detection through a prediction score.

# Video demo :

1. youtube link : https://youtu.be/Tf9NPc8TVHQ

# Main files :

1. The python script : application.py
2. The model : shapedetector_model_4b.h5

# Instructions to run the application :

1. Type 'python3 application.py' to run the application.

# Installation Requirements :

1. Need to have pip3 installed.
2. Need to have python3 installed.

# Setting up virtual environment (Optional)

1. Set up a virtual env : “python3 -m venv environmentname”
2. Activate virtual env : “source environmentname/bin/activate”
3. Deactivate env : “dectivate”
4. Duplicate environment from requirement file : “pip3 install -r requirements.txt”

Now all the modules will be installed on that environment.

# Dataset used : 

1. https://www.kaggle.com/datasets/shahirabdullah/handwritten-triangles 
2. https://www.kaggle.com/datasets/shahirabdullah/handwritten-squares 
3. https://www.kaggle.com/datasets/shahirabdullah/handwritten-rectangles 
4. https://www.kaggle.com/datasets/shahirabdullah/handwritten-circles 
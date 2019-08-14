import os
import pandas as pd 
import numpy as np
import glob 
import cv2

IMG_FOLDER = './images'
def load_img_ext(fname='img_ext.txt'):
    a = open(fname,"r")
    L_ext = a.read().split('\n') 
    a.close()
    return L_ext

def list_img_ext():
#lists all the image extensions
    return
             
def add_img_ext():
#adds a new extension to              
    return             

def del_img_ext():
    #removes an existing extension from the list
    return             

# Calling - 
img_extns = load_img_ext()
#print(img_extns) 

def get_img_list(IMG_FOLDER,img_extns = load_img_ext()):  #later initialize the imaage extension s in the class initialization
    all_images = []
    for img_ext in img_extns:
        img_path = glob.glob(IMG_FOLDER + "/*."+img_ext)
        if len(img_path) > 0 : all_images = all_images + img_path  #concatenate to the main list 
    return all_images
    


get_img_list(IMG_FOLDER)


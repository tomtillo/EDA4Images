%matplotlib inline
import matplotlib.pyplot as plt
# ---------------
import os
import pandas as pd 
import numpy as np
import glob 
# ------ Image packages 
import cv2
from skimage import io
# -------------
from sklearn.cluster import KMeans


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

def get_imgPath_list(img_folder,img_extns = load_img_ext()):  #later initialize the imaage extension s in the class initialization
    all_images = []
    for img_ext in img_extns:
        img_path = glob.glob(img_folder + "/*."+img_ext)
        if len(img_path) > 0 : all_images = all_images + img_path  #concatenate to the main list 
    return all_images
    
# Create an image extension list 
# # How many images are present  - Should filter off the other files 
# how many unique combinations of RGB values are present in the image
# Dominant 10 colors in any image ( sorted based on the number of pixels )

# scatter plot based on height and width
# eda on the aspect ratio

all_images = get_imgPath_list(IMG_FOLDER)

def get_image_details(IMG_PATH):
    im= cv2.imread(IMG_PATH)
    #image_size = im.size  # width by height
    #print(IMG_PATH)
    image_height = im.shape[0]
    image_width = im.shape[1]
    channels = im.shape[2]
    return image_height,image_width,channels

def get_imgShape_list(img_folder):
    hwList = []
    all_images = get_imgPath_list(img_folder)
    
    for one_image in all_images :
        height,width,channel = get_image_details(one_image)
        hwList.append([height,width])
    return hwList

#k=get_imgShape_list(IMG_FOLDER)

def printScatter(img_folder):
    
    hwList= get_imgShape_list(img_folder)
    len_list = len(hwList)
    hwList= np.array(hwList).flatten()
    jj=np.reshape(hwList,(len_list,2))
    x=jj[:,0]
    y=jj[:,1]
    plt.scatter(x,y)
    return 

def dominant10 (img_path,ncolors):
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h,w,c = img.shape[0],img.shape[1],img.shape[2]
    img_df = img.reshape(h*w, c)
    kmeans = KMeans(n_clusters=ncolors,max_iter=20,random_state=325) #
    model = kmeans.fit(pd.DataFrame(img_df))
    clus_centers = np.asarray(kmeans.cluster_centers_,dtype=np.uint8) 
    labels = np.asarray(model.labels_,dtype=np.uint8 )  
    img_lbldata = np.reshape(labels,(h,w))
    #Recretate image with new rgb values
    image_new = np.zeros((h,w,c),dtype=np.uint8 )
    for i in range(h):
        for j in range(w):
            image_new[i,j,:] = clus_centers[img_lbldata[i,j],:]
    #io.imsave('./images/newp1_15.jpg',image_new)    
    #return clus_centers,img_lbldata
    return image_new


# ------------ Usage ------------
get_img_list(IMG_FOLDER)
printScatter('./images')

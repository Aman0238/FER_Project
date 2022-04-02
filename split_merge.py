import cv2
import matplotlib.pyplot as plt

def splitImages(img):
    b,g,r = cv2.split(img)


    # Print the channels
    #plt.figure(figsize = [20,5])
    #plt.subplot(141);plt.imshow(r, cmap= 'gray');plt.title("The RED channel");
   # plt.subplot(142);plt.imshow(g, cmap='gray');plt.title("The GREEN channel");
    #plt.subplot(143);plt.imshow(b, cmap='gray');plt.title("The BlUE channel");
    return r,g,b
    #cv2.waitKey()
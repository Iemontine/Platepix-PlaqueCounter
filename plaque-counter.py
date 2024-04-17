import cv2
from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from matplotlib.widgets import RangeSlider, Slider

# prompt and get file
root = tk.Tk()
root.withdraw()
filepath = filedialog.askopenfilename()

# define the plot
fig,axes = plt.subplots(figsize=(9,9))

# adjust subplots so there's space for the sliders
plt.subplots_adjust(top=0.95, bottom=0.25, right=0.80)

# define axes and sliders
sides_ax  = fig.add_axes([0.375, 0.20, 0.50, 0.03], facecolor='black')
sides_slider = RangeSlider(sides_ax, "Sides", 0, 100,valinit=(10,100),valstep=1)
perimeter_slider_ax  = fig.add_axes([0.375, 0.15, 0.50, 0.03], facecolor='black')
perimeter_slider = RangeSlider(perimeter_slider_ax, "Perimeter", 0, 3000,valinit=(10,1000))
area_slider_ax  = fig.add_axes([0.375, 0.10, 0.50, 0.03], facecolor='black')
area_slider = Slider(area_slider_ax, label="MinArea", valmin=0, valmax=750, valinit=0,valstep=0.001)
mask_slider_ax = fig.add_axes([0.375, 0.05, 0.50, 0.03], facecolor='black')
mask_slider = Slider(mask_slider_ax, label="Mask", valmin=200, valmax=550, valinit=475, valstep=1)
x_slider_ax = fig.add_axes([0.85, 0.28, 0.025, 0.635], facecolor='black')
x_slider = Slider(x_slider_ax, label="Mask X", valmin=0, valmax=1000, valinit=500, orientation="vertical")
y_slider_ax = fig.add_axes([0.925, 0.28, 0.025, 0.635], facecolor='black')
y_slider = Slider(y_slider_ax, label="Mask Y", valmin=0, valmax=1000, valinit=500, orientation="vertical")

# load image and perform initial image filtering (sharpness, brightness, etc.)
original = cv2.imread(filepath)
original = cv2.resize(original, (1000, 1000), interpolation=cv2.INTER_AREA)     # resize image to 1000x1000
sharpened = cv2.filter2D(original, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))
contrast = cv2.addWeighted(sharpened, 1, sharpened, 0, 0)
gamma = 2
gamma = cv2.LUT(contrast, np.array([((i / 255) ** (1/gamma)) * 255 for i in range(256)], np.uint8))

# image transformations
gray = cv2.cvtColor(gamma, cv2.COLOR_BGR2GRAY)                           # grayscale image
blur = cv2.GaussianBlur(gray, (7, 7), 0)                                  # apply (10,10) gaussian blur
canny = cv2.Canny(blur, 30, 150, 3)                                         # convert image to canny edges
dilated = cv2.dilate(canny, (1, 1), iterations=0)                           # dilate image
image = cv2.resize(dilated, (1000, 1000), interpolation=cv2.INTER_AREA)     # resize image to 1000x1000

# display initial image
final = axes.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), interpolation='nearest', aspect='auto')

# create textboxes
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textbox1 = plt.figtext(0.05, 0.175, f"Plaque/Bacteria Count\n---",  fontsize=13, c='black', bbox=props)
textbox2 = plt.figtext(0.05, 0.09, f"Average area: \n---", fontsize=13, c='black', bbox=props)

sliders_on_changed(-1)

area_slider.on_changed(sliders_on_changed)
perimeter_slider.on_changed(sliders_on_changed)
sides_slider.on_changed(sliders_on_changed)
mask_slider.on_changed(sliders_on_changed)
y_slider.on_changed(sliders_on_changed)
x_slider.on_changed(sliders_on_changed)

axes.axis('off')
plt.show()

# function that gets called every time one of the sliders gets touched
def sliders_on_changed(val):
    global image, original, textbox
   
    originalcopy = original.copy()
    copy = image.copy()
   
    # create and apply a mask, limiting the area where contours are found
    mask = np.zeros(copy.shape[:2], dtype="uint8")
    cv2.circle(mask, (int(x_slider.val), int(y_slider.val)), mask_slider.val, 255, -1)
    copy = cv2.bitwise_and(copy, copy, mask=mask)
   
    # find contours
    contours, hierarchy = cv2.findContours(copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
   
    positives = []
    negatives = []
   
    # get slider values
    amin = area_slider.val
    pmin = perimeter_slider.val[0]
    pmax = perimeter_slider.val[1]
    smin = sides_slider.val[0]
    smax = sides_slider.val[1]
   
    #print(amin, amax, pmin, pmax, smin, smax)
   
    totalarea = []
    # finds all contours that fall under the specifications of the slider parameters
    for contour in contours:
        sides = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
        totalarea.append(area)
        perimeter = cv2.arcLength(contour, True)
        if (len(sides) > smin and len(sides) < smax) and (perimeter >= pmin and perimeter <= pmax) and (area >= amin):
            positives.append(contour)
        else:
            negatives.append(contour)
   
    # update the plot with the image with positive contours and the limiting circle
    originalcopy = cv2.resize(originalcopy, (1000, 1000), interpolation=cv2.INTER_AREA)
    originalcopy = cv2.cvtColor(originalcopy, cv2.COLOR_BGR2RGB)
    cv2.drawContours(originalcopy, positives, -1, (0,255,0), 4)
    #cv2.drawContours(originalcopy, negatives, -1, (0,0,255), 2)
    cv2.circle(originalcopy, (int(x_slider.val), int(y_slider.val)), mask_slider.val, (255,0,0), 4)
   
    textbox1.set_text(f"Plaque/Bacteria Count\n{len(positives)}")
    textbox2.set_text(f"Average area\n{round(sum(totalarea)/len(totalarea)*(92.4/2473),2)} mm^2")
   
    final.set_data(originalcopy)


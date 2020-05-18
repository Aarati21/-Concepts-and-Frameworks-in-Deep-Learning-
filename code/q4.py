import numpy as np

import skimage
import skimage.measure
#import skimage.color
#import skimage.restoration
#import skimage.filters
#import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    #convert to grayscale
    #denoise
    denoise_im = skimage.restoration.denoise_bilateral(image, multichannel=True)
    #grayscale
    grayscale = skimage.color.rgb2gray(denoise_im)
    #threshold
    threshold = skimage.filters.threshold_otsu(grayscale)
    #morphology
    morph = skimage.morphology.closing(grayscale<threshold, skimage.morphology.square(3))
    
    
    #label image
    labels = skimage.measure.label(morph)
    regions = skimage.measure.regionprops(labels)
    
    #find mean region area
    mean_region_area = sum([i.area for i in regions])/len(regions)
    
    for j in regions:
        
        if j.area > (mean_region_area/2):
            y1, x1, y2, x2 = j.bbox
            bboxes.append(np.array([y1, x1, y2, x2]))
            
    bw = (grayscale>threshold).astype(np.float)
            
    return bboxes, bw



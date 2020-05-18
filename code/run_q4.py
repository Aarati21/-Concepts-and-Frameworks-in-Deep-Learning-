import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../data_images/images'):
    
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('..\\data_images\\images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw, cmap='gray')
    
    for bbox in bboxes:
        
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
        
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    # get centroid positions and width and height of box
    y_c = [(i[2] + i[0])/2 for i in bboxes]
    x_c = [(i[3] + i[1])/2 for i in bboxes]
    y_h = [(i[2] - i[0]) for i in bboxes]
    x_w = [(i[3] - i[1]) for i in bboxes]
    mean_height = sum(y_h)/len(y_h)
    positions = list(zip(y_c, x_c, y_h, x_w))
    positions = sorted(positions, key = lambda a: (a[0], a[1]))
    temp = positions[0][0]
    row = []
    rows = []
    for p in positions:
        if p[0] < temp + mean_height:
            
            row.append(p)
        else:
            row = sorted(row, key=lambda a: a[1])
            rows.append(row)
            row = [p]
            temp = p[0]
    row = sorted(row, key=lambda a: a[1])
    rows.append(row)
   
    
    rowsd = []
    for row in rows:
        rowd = []
        for y, x, h, w in row:
            
            im = bw[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
            
            h_pad, w_pad = 0, 0
            if h > w:
                h_pad = h/20
                w_pad = (h-w)/2+h_pad
            elif h < w:
                w_pad = w/20
                h_pad = (w-h)/2+w_pad
            im = np.pad(im, ((int(h_pad), int(h_pad)), (int(w_pad), int(w_pad))), 'constant', constant_values=(1, 1))
            
            im = skimage.transform.resize(im, (32, 32))
            im = skimage.morphology.erosion(im, np.array([[0, 2, 0], [2, 2, 2], [0, 2, 0]]))
            rowd.append(np.transpose(im).flatten())
        rowsd.append(np.array(rowd))
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    
    # load the weights
    # run the crops through your neural network and print them out
    

    
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle', 'rb'))

    
    for row_data in rowsd:
        h1 = forward(row_data, params, 'layer1')
        out = forward(h1, params, 'output', softmax)
        sen = ''
        b = np.argmax(out, axis = 1)
        for i in range(len(out)):
            sen += letters[b[i]]

        print(sen)
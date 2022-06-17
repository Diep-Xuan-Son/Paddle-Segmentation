# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import cv2
import numpy as np
from PIL import Image as PILImage

from matplotlib import pyplot as plt    ##them
from sklearn.cluster import KMeans

def rgb_to_hsv(r, g, b):
    # R, G, B values are divided by 255
    # to change the range from 0..255 to 0..1:
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # h, s, v = hue, saturation, value
    cmax = max(r, g, b) # maximum of r, g, b
    cmin = min(r, g, b) # minimum of r, g, b
    diff = cmax-cmin     # diff of cmax and cmin.

    # if cmax and cmax are equal then h = 0
    if cmax == cmin:
        h = 0
    
    # if cmax equal r then compute h
    elif cmax == r:
        h = (60 * ((g - b) / diff) + 360) % 360

    # if cmax equal g then compute h
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360

    # if cmax equal b then compute h
    elif cmax == b:
        h = (60 * ((r - g) / diff) + 240) % 360

    # if cmax equal zero
    if cmax == 0:
        s = 0
    else:
        s = (diff / cmax) * 100

    # compute v
    v = cmax * 100
    return round(h), round(s), round(v)

def find_color(h,s,v):
    if(v<17):
        color = 'Black'
    elif(17<=v<42):
        if(s>21):
            if(0<=h<10 or 360>=h>340):
                if(s<50):
                    color = 'Brown'
                else:
                    color = 'Red'
            elif(40>h>=10):
                if(s<50):
                    color = 'Brown'
                else:
                    color = 'Orange'
            elif(68>h>=40):
                color = 'Yellow'
            elif(170>h>=68):
                color = 'Green'
            elif(260>h>=170):
                if(s<27):
                    color = 'Grey'
                else:
                    color = 'Blue'
            elif(300>h>=260):
                color = 'Purple'
            elif(340>h>=300):
                color = 'Pink'
        elif(5<s<=21):
            if(260>h>=170):
                color = 'Grey'
            else:
                color = 'Brown'
        else:
            color = "Grey"
    elif(42<=v<60):
        if(s>15):
            if(0<=h<10 or 360>=h>340):
                if(s<25):
                    color = 'Brown'
                else:
                    color = 'Red'
            elif(40>h>=10):
                if(s<25):
                    color = 'Brown'
                else:
                    color = 'Orange'
            elif(68>h>=40):
                color = 'Yellow'
            elif(170>h>=68):
                color = 'Green'
            elif(260>h>=170):
                if(s<25):
                    color = 'Grey'
                else:
                    color = 'Blue'
            elif(300>h>=260):
                color = 'Purple'
            elif(340>h>=300):
                color = 'Pink'
        else:
            color = "Grey"

    elif(60<=v<75):
        if(s>10):
            if(0<=h<10 or 360>=h>340):
                if(s<20):
                    color = 'Brown'
                else:
                    color = 'Red'
            elif(40>h>=10):
                if(s<20):
                    color = 'Brown'
                else:
                    color = 'Orange'
            elif(68>h>=40):
                color = 'Yellow'
            elif(170>h>=68):
                color = 'Green'
            elif(260>h>=170):
                if(s<14):
                    color = 'Grey'
                else:
                    color = 'Blue'
            elif(300>h>=260):
                color = 'Purple'
            elif(340>h>=300):
                color = 'Pink'
        else:
            color = "Grey"
    elif(75<=v<=100):
        if(s>6):
            if(0<=h<16 or 360>=h>=340):
                color = 'Red'
            elif(40>h>=16):
                color = 'Orange'
            elif(68>h>=40):
                color = 'Yellow'
            elif(170>h>=68):
                color = 'Green'
            elif(260>h>=170):
                if(s<12):
                    color = 'White'
                else:
                    color = 'Blue'
            elif(300>h>=260):
                color = 'Purple'
            elif(340>h>=300):
                color = 'Pink'
        else:
            color = "White"
    return color

def dominantcolor(im, result, label):    #them
    dict_attribute = {0:'background',1:'skin',2:'bag',3:'pant',4:'shirt',5:'shoe',6:'skirt',7:'glasses',8:'hair',9:'hat'}
    km = KMeans(n_clusters=3)
    im_attribute = im[result == label]   #them
    if(len(im_attribute) > 10000):
        im_attribute = im_attribute[np.random.choice(im_attribute.shape[0], int(len(im_attribute)/50), replace=False), :]
    elif(len(im_attribute) > 1000):
        im_attribute = im_attribute[np.random.choice(im_attribute.shape[0], int(len(im_attribute)/5), replace=False), :]
    print(len(im_attribute))
    try:
        km.fit(im_attribute)
        colors = np.asarray(km.cluster_centers_, dtype='uint8')
        # print(colors)
        # print(np.unique(km.labels_, return_counts = True)[1])
        percentage = np.asarray(np.unique(km.labels_, return_counts = True)[1], dtype='float32')
        percentage = percentage/im_attribute.shape[0]
        # print(percentage)
        colors = colors[percentage == max(percentage)]
        h,s,v = rgb_to_hsv(colors[0][0],colors[0][1],colors[0][2])
        print(h,s,v)
        name_color = find_color(h,s,v)
        print('Color of ' + dict_attribute[label] + ':' + name_color)
        return name_color
    except:
        print("Dont't have " + dict_attribute[label])
    

def visualize(image, result, color_map, save_dir=None, weight=0.6):
    """
    Convert predict result to color image, and save added image.

    Args:
        image (str): The path of origin image.
        result (np.ndarray): The predict result of image.
        color_map (list): The color used to save the prediction results.
        save_dir (str): The directory for saving visual image. Default: None.
        weight (float): The image weight of visual image, and the result weight is (1 - weight). Default: 0.6

    Returns:
        vis_result (np.ndarray): If `save_dir` is None, return the visualized result.
    """

    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    color_map = np.array(color_map).astype("uint8")
    # Use OpenCV LUT for color mapping
    c1 = cv2.LUT(result, color_map[:, 0])
    c2 = cv2.LUT(result, color_map[:, 1])
    c3 = cv2.LUT(result, color_map[:, 2])
    pseudo_img = np.dstack((c3, c2, c1))

    im = cv2.imread(image)
    #----------------------------------------------------
    print(image)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)    ##them  #chuyen sang BGR der co dang RGB dua vao tim hsv (co the do opencv bi loi, opencv ben C++ khong can convert)
    # im = cv2.resize(im, (192,128), interpolation = cv2.INTER_AREA)  #them 
    # km = KMeans(n_clusters=3)
    # im1 = im.reshape(-1,3)
    # km.fit(im1)
    # colors = np.asarray(km.cluster_centers_, dtype='uint8')
    # print(colors)
    # print(np.unique(km.labels_, return_counts = True)[1])
    # percentage = np.asarray(np.unique(km.labels_, return_counts = True)[1], dtype='float32')
    # percentage = percentage/im1.shape[0]
    # colors = colors[percentage == max(percentage)]
    # h,s,v = rgb_to_hsv(colors[0][0],colors[0][1],colors[0][2])
    # print(h,s,v)
    # exit()
    # ['background','skin','bag','pant','shirt','shoe','skirt','glasses','hair','hat']
    list_attribute = [3,4,5,6,8]

    for at in list_attribute:
        color_skin = dominantcolor(im, result, at)

    # if 'skin' in list_attribute:
    #     color_skin = dominantcolor(im, result, 1)
    # if 'bag' in list_attribute:
    #     color_bag = dominantcolor(im, result, 2)
    # if 'pant' in list_attribute:
    #     color_pant = dominantcolor(im, result, 3)
    # if 'shirt' in list_attribute:
    #     color_shirt = dominantcolor(im, result, 4)
    # if 'shoe' in list_attribute:
    #     color_shoe = dominantcolor(im, result, 5)
    # if 'skirt' in list_attribute:
    #     color_skirt = dominantcolor(im, result, 6)
    # if 'hair' in list_attribute:
    #     color_hair = dominantcolor(im, result, 8)
    # if 'hat' in list_attribute:
    #     color_hat = dominantcolor(im, result, 9)

    # colors = np.expand_dims(color, axis=0)
    # plt.figure(0)
    # for ix in range(colors.shape[0]):
    #     patch = np.ones((20, 20, 3))
    #     patch[:, :, :] = colors[ix]
    #     plt.subplot(1, colors.shape[0], ix+1)
    #     plt.axis('off')
    #     plt.imshow(patch.astype('uint8'))
    # plt.show()                              ##them
    # exit()
    #---------------------------------------------------------------------------
    vis_result = cv2.addWeighted(im, weight, pseudo_img, 1 - weight, 0)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_name = os.path.split(image)[-1]
        out_path = os.path.join(save_dir, image_name)
        cv2.imwrite(out_path, vis_result)
    else:
        return vis_result


def get_pseudo_color_map(pred, color_map=None):
    """
    Get the pseudo color image.

    Args:
        pred (numpy.ndarray): the origin predicted image.
        color_map (list, optional): the palette color map. Default: None,
            use paddleseg's default color map.

    Returns:
        (numpy.ndarray): the pseduo image.
    """
    pred_mask = PILImage.fromarray(pred.astype(np.uint8), mode='P')
    # print(pred_mask)
    # print(pred_mask.size)
    if color_map is None:
        color_map = get_color_map_list(256)
    # print(color_map)
    # color_map = [255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    pred_mask.putpalette(color_map)
    return pred_mask


def get_color_map_list(num_classes, custom_color=None):
    """
    Returns the color map for visualizing the segmentation mask,
    which can support arbitrary number of classes.

    Args:
        num_classes (int): Number of classes.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.

    Returns:
        (list). The color map.
    """

    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]

    if custom_color:
        color_map[:len(custom_color)] = custom_color
    return color_map


def paste_images(image_list):
    """
    Paste all image to a image.
    Args:
        image_list (List or Tuple): The images to be pasted and their size are the same.
    Returns:
        result_img (PIL.Image): The pasted image.
    """
    assert isinstance(image_list,
                      (list, tuple)), "image_list should be a list or tuple"
    assert len(
        image_list) > 1, "The length of image_list should be greater than 1"

    pil_img_list = []
    for img in image_list:
        if isinstance(img, str):
            assert os.path.exists(img), "The image is not existed: {}".format(
                img)
            img = PILImage.open(img)
            img = np.array(img)
        elif isinstance(img, np.ndarray):
            img = PILImage.fromarray(img)
        pil_img_list.append(img)

    sample_img = pil_img_list[0]
    size = sample_img.size
    for img in pil_img_list:
        assert size == img.size, "The image size in image_list should be the same"

    width, height = sample_img.size
    result_img = PILImage.new(sample_img.mode,
                              (width * len(pil_img_list), height))
    for i, img in enumerate(pil_img_list):
        result_img.paste(img, box=(width * i, 0))

    return result_img

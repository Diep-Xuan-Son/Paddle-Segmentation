import cv2
import os
import numpy as np

def change_class(x):    #them
    # num_of_data = x.shape[0]
    # shape1 = x.shape[1]
    # shape2 = x.shape[2]
    # # print(num_of_data)
    # x = x.reshape([num_of_data,-1])
    # result = np.zeros(x.shape)
    # classes = [0, 41, 19, 39]

    # for k in range(num_of_data):
    #     for i, v in enumerate(x[k]):
    #         if v not in classes:
    #             result[k][i] = 3 # clothes
    #         else:
    #             if v == 0:
    #                 result[k][i] = 0 # background
    #             elif v == 41:
    #                 result[k][i] = 1 # skin
    #             elif v == 19:
    #                 result[k][i] = 2 # hair
    #             else:
    #                 result[k][i] = 4 # shoes
    # return result.reshape([num_of_data, shape1, shape2])

    result = x.copy()
    
    # result[result == 0] = 0     #background
    # result[result == 41] = 1    #skin
    # result[result == 19] = 2    #hair
    # result[result == 39] = 3    #shoes
    # result[result > 3] = 4
    background = [0]
    skin = [1, 9, 15, 18, 29, 34, 41, 44, 56, 57]
    bag = [2, 33]
    pant = [3, 23, 25, 27, 30, 31, 40, 45, 53]
    shirt = [4, 5, 6, 8, 10, 11, 13, 22, 24, 26, 37, 38, 46, 48, 49, 50, 51, 52, 54, 55]
    shoe = [7, 12, 16, 21, 28, 32, 36, 39, 43, 58]
    skirt = [14, 35, 42]
    glasses = [17, 47]
    hair = [19]
    hat = [20]

    # list_label = ['background','skin','bag','pant','shirt','shoe','skirt','glasses','hair','hat']
    dict_label = {  '0':[0],
                    '1':[1, 9, 15, 18, 29, 34, 41, 44, 56, 57],
                    '2':[2, 33],
                    '3':[3, 23, 25, 27, 30, 31, 40, 45, 53],
                    '4':[4, 5, 6, 8, 10, 11, 13, 22, 24, 26, 37, 38, 46, 48, 49, 50, 51, 52, 54, 55],
                    '5':[7, 12, 16, 21, 28, 32, 36, 39, 43, 58],
                    '6':[14, 35, 42],
                    '7':[17, 47],
                    '8':[19],
                    '9':[20]}
    for k,v in dict_label.items():
        for i in v:
            result[result == i] = int(k)
    return result


if __name__ == '__main__':
    old_path = '/home/son/AI/Brief cam/PaddleSeg/data_seg_clothing/clothing/annotations/train/clothing_seg'
    new_path = '/home/son/AI/Brief cam/PaddleSeg/data_seg_clothing/clothing/annotations/train/clothing_seg_new'
    if not os.path.isdir(new_path):
        os.mkdir(new_path)
    for name in os.listdir(old_path):
        img = cv2.imread(os.path.join(old_path,name), 0)
        # print(img.shape)
        # print(type(img))
        # exit()
        new_img = change_class(img)
        print(np.unique(new_img))
        print(new_img.shape)
        cv2.imwrite(os.path.join(new_path,name),new_img)
import numpy as np
import os
import random
import pickle
from scipy.misc import imread, imresize, imsave
from random import randint
def trainvalid_test_split():
    img_dir = 'D:\workspace\common_database\钢板\clean'
    img_cls = ['Normal', 'Red', 'Wrinkle']

    #先把 训练验证（10 fold） 和 测试集先分开
    for ic in img_cls:
        img_list = os.listdir(os.path.join(img_dir, ic))
        img_files = []
        for il in img_list:
            if il[-3:]=='jpg':
                img_files.append(os.path.join(ic,il))

        if ic == 'Wrinkle':
            trainvalid_num = int(0.8*len(img_files))
            test_num = len(img_files) - trainvalid_num
        else:
            trainvalid_num = 10000
            test_num = 2000
        random.shuffle(img_files)
        trainvalid_set = img_files[:trainvalid_num]
        test_set = img_files[trainvalid_num: trainvalid_num+test_num]

        print('c', ic)
        print(len(trainvalid_set))
        print(len(test_set))

        with open(os.path.join('dataset_split',ic+'_trainvalid.pickle'), 'wb') as f:
            pickle.dump(trainvalid_set, f, protocol=-1)
        with open(os.path.join('dataset_split',ic+'_test.pickle'), 'wb') as f:
            pickle.dump(test_set, f, protocol=-1)

def compute_mean_std():
    # mean 101.125349669
    # std 54.6458422562

    X = []
    classes = ['Normal', 'Red', 'Wrinkle']
    for i in range(len(classes)):
        with open(os.path.join('dataset_split', classes[i]+'_trainvalid.pickle'), 'rb') as f:
            samples = pickle.load(f, encoding="bytes")
            X += samples

    img_dir = 'D:\workspace\common_database\钢板\clean'
    all_imgs = []
    for x in X:
        img = imread(os.path.join(img_dir, x))
        all_imgs.append(img)
    all_imgs = np.asarray(all_imgs)
    print('mean', np.mean(all_imgs))
    print('std', np.std(all_imgs))


def test_oneline_augment(img, i, isTrain):
    scale_resize = [32, 195]
    crop_size = [128-4, 780-4]
    mean = 101.125/255
    std = 54.646/255


    def random_crop(img, width, height):
        width1 = randint(0, img.shape[0] - width)
        height1 = randint(0, img.shape[1] - height)
        width2 = width1 + width
        height2 = height1 + height
        img = img[width1:width2, height1:height2]
        return img

    def center_crop(img, width, height):
        width1 = int((img.shape[0] - width)/2)
        height1 = int((img.shape[1] - height)/2)
        width2 = width1 + width
        height2 = height1 + height
        img = img[width1:width2, height1:height2]
        return img

    def random_flip_left_right(img):
        prob = randint(0, 1)
        if prob == 1:
            img = np.fliplr(img)
        return img

    if isTrain:
        new_img = random_flip_left_right(img)
        new_img = random_crop(new_img, width=crop_size[0], height=crop_size[1])
        new_img = imresize(new_img, size=scale_resize)
    else:
        new_img = center_crop(img, width=crop_size[0], height=crop_size[1])
        new_img = imresize(new_img, size=scale_resize)

    imsave(os.path.join('onelineAug_test', str(i)+'.jpg'), new_img)

if __name__ == '__main__':
    # compute_mean_std()

    img = imread('P4680030_00917499_04_srcimg_0093.jpg')
    for i in range(100):
        test_oneline_augment(img, i, isTrain=False)






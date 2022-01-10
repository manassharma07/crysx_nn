from os import listdir
from os.path import isfile, join
from PIL import Image, ImageDraw
import numpy as np
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import cv2
try:
    import image_slicer                     
except ImportError:
    print('image_slicer could not be imported!')

def downloadMNIST(url='https://github.com/manassharma07/MNIST-PLUS/archive/refs/tags/PNG.zip', extract_to='.'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)

def loadDigitDataset(mypath, digit=1):
    '''
    mypath is the path to the folder/directory containing the png images of a particular digit
    ex: mypath= 'mnist_orig_png/training/1/'
    '''
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    training_png = np.empty((len(onlyfiles),28,28))
    training_label = np.ones((len(onlyfiles),1))*digit
    i=0
    for file in onlyfiles:
        img = Image.open(mypath+'/'+file, mode='r')
        np_img = np.array(img.getdata())
        training_png[i,:,:] = np_img.reshape(28,28)
        i=i+1
    return training_png, training_label

def unison_shuffled_copies(a, b):
#     https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def loadMNIST(path_main, train=True, shuffle=True):
    '''
    path_main is the path to the folder/directory containing the 'training' and 'testing' directories of the mnist dataset
    ex: path_main= 'mnist_orig_png'
    '''
    if train:
        path = path_main +'/training/0'
    else:
        path = path_main +'/testing/0'
    data_0_png, data_0_label = loadDigitDataset(path, digit=0)
    if train:
        path = path_main +'/training/1'
    else:
        path = path_main +'/testing/1'
    data_1_png, data_1_label = loadDigitDataset(path, digit=1)
    if train:
        path = path_main +'/training/2'
    else:
        path = path_main +'/testing/2'
    data_2_png, data_2_label = loadDigitDataset(path, digit=2)
    if train:
        path = path_main +'/training/3'
    else:
        path = path_main +'/testing/3'
    data_3_png, data_3_label = loadDigitDataset(path, digit=3)
    if train:
        path = path_main +'/training/4'
    else:
        path = path_main +'/testing/4'
    data_4_png, data_4_label = loadDigitDataset(path, digit=4)
    if train:
        path = path_main +'/training/5'
    else:
        path = path_main +'/testing/5'
    data_5_png, data_5_label = loadDigitDataset(path, digit=5)
    if train:
        path = path_main +'/training/6'
    else:
        path = path_main +'/testing/6'
    data_6_png, data_6_label = loadDigitDataset(path, digit=6)
    if train:
        path = path_main +'/training/7'
    else:
        path = path_main +'/testing/7'
    data_7_png, data_7_label = loadDigitDataset(path, digit=7)
    if train:
        path = path_main +'/training/8'
    else:
        path = path_main +'/testing/8'
    data_8_png, data_8_label = loadDigitDataset(path, digit=8)
    if train:
        path = path_main +'/training/9'
    else:
        path = path_main +'/testing/9'
    data_9_png, data_9_label = loadDigitDataset(path, digit=9)
    data_png = np.concatenate((data_0_png, data_1_png, data_2_png, data_3_png, \
                                  data_4_png, data_5_png, data_6_png, data_7_png, data_8_png, data_9_png), axis=0)
    label = np.concatenate((data_0_label, data_1_label, data_2_label, data_3_label, \
                                  data_4_label, data_5_label, data_6_label, data_7_label, data_8_label, data_9_label), axis=0)
    if shuffle:
        data_png, label = unison_shuffled_copies(data_png, label)
    
    return data_png, label

def one_hot_encode(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.astype(int).reshape(-1)])

def createGrid():
    # Load the image
    height = 3000
    width = 3000
    img = Image.new(mode='L', size=(height, width), color=0)
    img.show()

    ## Draw grids
    # draw horizontal lines
    draw = ImageDraw.Draw(img, mode='L')
    for i in range(200, img.height,200):
        x_start = 0.0
        x_end = img.width
        y_start = i
        y_end = i
        line = ((x_start, y_start), (x_end, y_end))
        draw.line(line, fill=200,width=1)
    # draw vertical lines
    draw = ImageDraw.Draw(img, mode='L')
    for i in range(200, img.height,200):
        x_start = i
        x_end = i
        y_start = img.height
        y_end = 0
        line = ((x_start, y_start), (x_end, y_end))
        draw.line(line, fill=200,width=1)

    img.save('empty_grid.png')

def extractGridImages(dir_name, filnam, ext='png'):
    filename_ = dir_name+'/'+filnam
    filename = filename_+'.'+ext
    img = Image.open(filename, mode='r')



    image_slicer.slice(filename, 225)

    for i in range(1,16,1):
        for j in range(1,16,1):
            tilename = filename_+'_'+str(i).zfill(2)+'_'+str(j).zfill(2)+'.'+ext
    #         print(tilename)
            tileImg = Image.open(tilename, mode='r')
            w, h = tileImg.size
            tileImg = tileImg.crop((2, 2, w-2, h-2))
            tileImg.save(tilename)
            
            ###### Centering begin
            # Load image as grayscale and obtain bounding box coordinates
            image = cv2.imread(tilename, 0)

            height, width = image.shape
            x,y,w,h = cv2.boundingRect(image)

            # Create new blank image and shift ROI to new coordinates
            ROI = image[y:y+h, x:x+w]
            mask = np.zeros([ROI.shape[0]+10,ROI.shape[1]+10])
            width, height = mask.shape
            #     print(ROI.shape)
            #     print(mask.shape)
            x = width//2 - ROI.shape[0]//2 
            y = height//2 - ROI.shape[1]//2 
            #     print(x,y)
            mask[y:y+h, x:x+w] = ROI

            output_image = Image.fromarray(mask)
            compressed_output_image = output_image.resize((28,28))
            compressed_output_image.save(tilename)
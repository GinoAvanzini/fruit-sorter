# Implementation of k-means algorithm
# https://en.wikipedia.org/wiki/K-means_clustering

# Path to Fruits 360 dataset
path = '/home/gino/Desktop/fruits-360_dataset/'


from fruit import Fruit

from random import shuffle
from skimage import io


def img_grayscale(image):
    return io.imread(image, as_gray=True)


def main():
    
    fruit_list = []
    
    
    banana_collection = io.ImageCollection([path 
                                            + 'fruits-360/Training/Banana/*.jpg', 
                                            path + 'fruits-360/Training/Banana Lady Finger/*.jpg'],
                                            load_func=img_grayscale)
    orange_collection = io.ImageCollection(path 
                                           + 'fruits-360/Training/Orange/*.jpg',
                                           load_func=img_grayscale)
    lemon_collection = io.ImageCollection(path 
                                          + 'fruits-360/Training/Lemon/*.jpg', 
                                          load_func=img_grayscale)

    
    for i, banana in enumerate(banana_collection):
        fruit_list.append(Fruit(banana_collection.files[i]))
        
    for i, orange in enumerate(orange_collection):
        fruit_list.append(Fruit(orange_collection.files[i]))
    
    for i, lemon in enumerate(lemon_collection):
        fruit_list.append(Fruit(lemon_collection.files[i]))
    
    shuffle(fruit_list)
    
    print(len(fruit_list))
    print(fruit_list)
    
    return


if __name__ == '__main__':
    main()

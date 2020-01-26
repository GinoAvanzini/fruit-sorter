# Implementation of k-means algorithm
# https://en.wikipedia.org/wiki/K-means_clustering

# Path to Fruits 360 dataset
path = '/home/gino/Desktop/fruits-360_dataset/'

# Allow beginning k-means with reasonable means instead of random
cheats=True


from fruit import Fruit

from random import shuffle, randrange
from skimage import io


def img_grayscale(image):
    return io.imread(image, as_gray=True)

def k_means(fruit_list, cheats_on=False, init_means=None):
    pass
    
    
    


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

            
    fruit_list = [Fruit(banana_collection.files[i]) for i in range(len(banana_collection))]
    fruit_list.extend([Fruit(orange_collection.files[i]) for i in range(len(orange_collection))])
    fruit_list.extend([Fruit(lemon_collection.files[i]) for i in range(len(lemon_collection))])
    
    shuffle(fruit_list)
    
    banana_size = len(banana_collection)
    orange_size = len(orange_collection)
    lemon_size = len(lemon_collection)
    
    random_banana_position = randrange(0, banana_size - 1)
    random_orange_position = randrange(banana_size, banana_size + orange_size - 1)
    random_lemon_position = randrange(banana_size + orange_size, len(fruit_list))
    
    #cheat_means = [ [fruit_list[random_banana_position].hu
    
    #k_means(fruit_list, cheats_on=cheats, init_means=cheat_means)
    
    
    print(len(fruit_list))
    print(fruit_list)
    
    return


if __name__ == '__main__':
    main()

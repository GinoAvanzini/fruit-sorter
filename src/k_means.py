# Implementation of k-means algorithm
# https://en.wikipedia.org/wiki/K-means_clustering

# Path to Fruits 360 dataset
path = '/home/gino/Desktop/fruits-360_dataset/'

# Allow beginning k-means with reasonable means instead of random
# Initial means will be from one banana, one orange and one lemon
# Not recommended to leave False because there isn't an equal
# amount of each type of fruits
cheats_on = False


from fruit import Fruit

from random import shuffle, randrange
from skimage import io
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt



def img_grayscale(image):
    return io.imread(image, as_gray=True)

def k_means(fruit_list, init_means):
    
    print(init_means)
    
    A_centroid = init_means[0, :]
    B_centroid = init_means[1, :]
    C_centroid = init_means[2, :]
        
    k = 0
    j = 0
    assignments_changed = True
    
    while (assignments_changed):
        k += 1
        
        #print(A_centroid)
        #print(B_centroid)
        #print(C_centroid, "\n\n")

        
        assignments_changed = False
        sum_a = np.zeros(fruit_list[0].feature_size)
        sum_b = np.zeros(fruit_list[0].feature_size)
        sum_c = np.zeros(fruit_list[0].feature_size)
        count_a = 0
        count_b = 0
        count_c = 0
        
        
        for fruit in fruit_list:
            j += 1
            dist_a = np.sum(np.power(fruit.features - A_centroid, 2))
            dist_b = np.sum(np.power(fruit.features - B_centroid, 2))
            dist_c = np.sum(np.power(fruit.features - C_centroid, 2))
            
            if dist_a < dist_b and dist_a < dist_c:
                
                if fruit.guessed_label != 'A':
                    assignments_changed = True
                    fruit.guessed_label = 'A'
                
                sum_a += fruit.features
                count_a += 1
            elif dist_b < dist_a and dist_b < dist_c:
                if fruit.guessed_label != 'B':
                    assignments_changed = True
                    fruit.guessed_label = 'B'
                sum_b += fruit.features
                count_b += 1
            elif dist_c < dist_a and dist_c < dist_b:
                if fruit.guessed_label != 'C':
                    assignments_changed = True
                    fruit.guessed_label = 'C'
                sum_c += fruit.features
                count_c += 1
            else:
                print("You have incredibly bad luck. By now we ignore this")
                print("This incident will be reported\n")
                pass
        
        A_centroid = sum_a / count_a
        B_centroid = sum_b / count_b
        C_centroid = sum_c / count_c
        
    print(k)
    print(j)
    print(k*len(fruit_list))
                
    return [A_centroid, B_centroid, C_centroid]


def main(plotting=False, feature_mode='hu_only'):
    
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
            
    fruit_list = [Fruit(banana_collection.files[i], feature_mode=feature_mode) 
                  for i in range(len(banana_collection))]
    fruit_list.extend([Fruit(orange_collection.files[i], feature_mode=feature_mode) 
                       for i in range(len(orange_collection))])
    fruit_list.extend([Fruit(lemon_collection.files[i], feature_mode=feature_mode) 
                       for i in range(len(lemon_collection))])
        
    banana_size = len(banana_collection)
    orange_size = len(orange_collection)
    lemon_size = len(lemon_collection)
    
    if cheats_on:
        random_banana_position = randrange(0, banana_size - 1)
        random_orange_position = randrange(banana_size, banana_size + orange_size - 1)
        random_lemon_position = randrange(banana_size + orange_size, len(fruit_list))
    
        cheat_means = np.array([fruit_list[random_banana_position].features,
                                fruit_list[random_orange_position].features,
                                fruit_list[random_lemon_position].features
                                ])

        shuffle(fruit_list)
        means = k_means(fruit_list, cheat_means)
        
    else:
        initial_means = []
        for i in range(0, 3):
            position = randrange(0, len(fruit_list))
            initial_means.append(fruit_list[position].features)
        
        shuffle(fruit_list)
        means = k_means(fruit_list, np.array(initial_means))
        
    print(means)
    
    # Plotting results
    if plotting:
        if fruit_list[0].feature_size == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            for fruit in fruit_list:
                if fruit.guessed_label == 'A':
                    color = 'red'
                elif fruit.guessed_label == 'B':
                    color = 'green'
                elif fruit.guessed_label == 'C':
                    color = 'blue'
                else:
                    color = 'black'

                ax.scatter(fruit.features[0], fruit.features[1], fruit.features[2], c=color, alpha=.05)
            
            ax.scatter(means[0][0], means[0][1], means[0][2], marker='^', color='black', alpha=1)
            ax.scatter(means[1][0], means[1][1], means[1][2], marker='^', color='black', alpha=1)
            ax.scatter(means[2][0], means[2][1], means[2][2], marker='^', color='black', alpha=1)
            
            plt.show()
            
        elif fruit_list[0].feature_size == 2:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            
            for fruit in fruit_list:
                if fruit.guessed_label == 'A':
                    color = 'red'
                elif fruit.guessed_label == 'B':
                    color = 'green'
                elif fruit.guessed_label == 'C':
                    color = 'blue'
                else:
                    color = 'black'

                ax.scatter(x=fruit.features[0], y=fruit.features[1], c=color, alpha=.05)

            ax.scatter(x=means[0][0], y=means[0][1], marker='^', color='black', alpha=1)
            ax.scatter(x=means[1][0], y=means[1][1], marker='^', color='black', alpha=1)
            ax.scatter(x=means[2][0], y=means[2][1], marker='^', color='black', alpha=1)
            
            ax.set_xlim(left=0)
            ax.set_xlim(right=10)
            
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Hu[1] (log-normalized)")
            ax.set_ylabel("Hu[3] (log-normalized)")
            #ax.set_aspect("equal")
            
            plt.show()

    return


if __name__ == '__main__':
    
    # Plotting is self-explainatory
    # Feature mode is to toggle the features used in k-means training
    # hu_plus_ratio = [ Hu[1], Hu[3], moment_ratio ]
    # hu_only = [ Hu[1], Hu[3] ]
    # The latter allows a better clustering
    main(plotting=True, feature_mode='hu_only')

# Draft of AI 1 report

## Notes


### Thresholding
Thresholding is used to create a binary image from a grayscale image 1.

But given that our database is alredy preprocessed, the work will be simple. Background is alredy denoised and separated from the foreground fruit.

### Canny filter

https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.canny

The Canny filter is a multi-stage edge detector. It uses a filter based on the derivative of a Gaussian in order to compute the intensity of the gradients.The Gaussian reduces the effect of noise present in the image. Then, potential edges are thinned down to 1-pixel curves by removing non-maximum pixels of the gradient magnitude. Finally, edge pixels are kept or removed using hysteresis thresholding on the gradient magnitude.

The Canny has three adjustable parameters: the width of the Gaussian (the noisier the image, the greater the width), and the low and high threshold for the hysteresis thresholding.

The steps of the algorithm are as follows:

- Smooth the image using a Gaussian with sigma width.

- Apply the horizontal and vertical Sobel operators to get the gradients within the image. The edge strength is the norm of the gradient.

- Thin potential edges to 1-pixel wide curves. First, find the normal to the edge at each point. This is done by looking at the signs and the relative magnitude of the X-Sobel and Y-Sobel to sort the points into 4 categories: horizontal, vertical, diagonal and antidiagonal. Then look in the normal and reverse directions to see if the values in either of those directions are greater than the point in question. Use interpolation to get a mix of points instead of picking the one that’s the closest to the normal.

- Perform a hysteresis thresholding: first label all points above the high threshold as edges. Then recursively label any point above the low threshold that is 8-connected to a labeled point as an edge.

References
1 Canny, J., A Computational Approach To Edge Detection, IEEE Trans. Pattern Analysis and Machine Intelligence, 8:679-714, 1986 DOI:10.1109/TPAMI.1986.4767851 

### Inertia tensor for feature extraction

The relative magnitude of the eigenvalues of the tensor is thus a measure of the elongation of a (bright) object in the image

### K-means algorithm

#### A
https://towardsdatascience.com/introduction-to-image-segmentation-with-k-means-clustering-83fd0a9e2fc3
Steps in K-Means algorithm:
1. Choose the number of clusters K.
2. Select at random K points, the centroids(not necessarily from your dataset).
3. Assign each data point to the closest centroid → that forms K clusters.
4. Compute and place the new centroid of each cluster.
5. Reassign each data point to the new closest centroid. If any reassignment . took place, go to step 4, otherwise, the model is ready.


In spite of all the advantages K-Means have got but it fails sometimes due to the random choice of centroids which is called The Random Initialization Trap.

To solve this issue we have an initialization procedure for K-Means which is called K-Means++(Algorithm for choosing the initial values for K-Means clustering).

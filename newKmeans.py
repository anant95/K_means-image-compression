import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
from PIL import Image
import os
import argparse
from util import load_image, array2PIL

def get_args():
    parser = argparse.ArgumentParser(
        description='Parse Image file and number of cluster you want to create to compress that image ')
    parser.add_argument("--image", required=True,
                        help="Path of the image")
    parser.add_argument("--k", required=True,
                        help="Number of cluster")
    args = parser.parse_args()
    return args

class Kmeans:

    def kmeans_fun(n_colors,img,name):
        width, height = img.size
        # Convert to floats instead of the default 8 bits integer coding. Dividing by
        # 255 is important so that plt.imshow behaves works well on float data (need to
        # be in the range [0-1])
        orimg_arr = np.array(img, dtype=np.float64)
        img_arr = np.array(img, dtype=np.float64) / 255
        # Load Image and transform to a 2D numpy array.
        w, h, d = original_shape = tuple(img_arr.shape)
        assert d == 3
        image_array = np.reshape(img_arr, (w * h, d))
        print("Fitting model on a small sub-sample of the data")
        t0 = time()
        image_array_sample = shuffle(image_array, random_state=0)[:1000]
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
        print("done in %0.3fs." % (time() - t0))
        # Get labels for all points
        print("Predicting color indices on the full image (k-means)")
        t0 = time()
        labels = kmeans.predict(image_array)
        print("done in %0.3fs." % (time() - t0))
        """codebook_random = shuffle(image_array, random_state=0)[:n_colors]
        print("Predicting color indices on the full image (random)")
        t0 = time()
        labels_random = pairwise_distances_argmin(codebook_random,
                              image_array,
                              axis=0)
        print("done in %0.3fs." % (time() - t0))"""
        # Display all results, alongside original image
        plt.figure(1)
        plt.clf()
        plt.axis('off')
        plt.title('Original image (96,615 colors)')
        #plt.imshow(img_arr)

        plt.figure(2)
        plt.clf()
        plt.axis('off')
        plt.title('Quantized image (64 colors, K-Means)')
        plt.imsave(name,Kmeans.recreate_image(kmeans.cluster_centers_, labels, w, h))
        #plt.imshow(Kmeans.recreate_image(kmeans.cluster_centers_, labels, w, h))

        """
        plt.figure(3)
        plt.clf()
        plt.axis('off')
        plt.title('Quantized image (64 colors, Random)')
        plt.imshow(recreate_image(codebook_random, labels_random, w, h))"""
        plt.show()

    def recreate_image(codebook, labels, w, h):
        """Recreate the (compressed) image from the code book & labels"""
        d = codebook.shape[1]
        image = np.zeros((w, h, d))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                image[i][j] = codebook[labels[label_idx]]
                label_idx += 1
        return image
if __name__ == "__main__":
    args = get_args()
    k_str = args.k.split(',')
    quality_steps = [int(tt) for tt in k_str]
    img = Image.open(args.image)
    dir_path='temp_xxx_zzz'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for q in quality_steps:
        name = dir_path+'/temp_' + str(q) + '.jpg'
        Kmeans.kmeans_fun(q,img,name)

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from hough import *
from orientation_estimate import *

class bandera:
    def __init__(self, flag_name):
        self.flag = cv2.imread(flag_name)

    def colores(self):
        img = cv2.cvtColor(self.flag, cv2.COLOR_BGR2RGB)
        image = np.array(img, dtype=np.float64) / 255
        rows, cols, ch = image.shape
        assert ch == 3
        image_array = np.reshape(image, (rows * cols, ch))
        image_array_sample = shuffle(image_array, random_state=0)[:10000]
        self.inertia = []
        for n in range(1, 5, 1):
            model = KMeans(n_clusters=n, random_state=0).fit(image_array_sample)
            labels = model.predict(image_array)
            centers = model.cluster_centers_
            self.inertia.append(model.inertia_)
        colors = 0
        if self.inertia[0] <= 0.0:
            colors = 1
        elif self.inertia[1] <= 0.0:
            colors = 2
        elif self.inertia[2] <= 0.0:
            colors = 3
        elif self.inertia[3] <= 0.0:
            colors = 4

        print('Número de colores: ', colors)
        n_colors = [1, 2, 3, 4]
        plt.figure(3)
        plt.plot(n_colors, self.inertia, color="green", linewidth=1.0, )
        plt.title('Intra-cluster method=Kmeans)')
        plt.xlabel('Número de colores')
        plt.ylabel('Suma de distancia entre los datos y su cluster')
        plt.show()

    def porcentajes(self):
        img = cv2.cvtColor(self.flag, cv2.COLOR_BGR2GRAY)
        histr = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(histr)
        plt.xlim([0, 256])
        plt.show()

    def orientacion(self):
        high_thresh = 300
        bw_edges = cv2.Canny(self.flag, high_thresh * 0.3, high_thresh, L2gradient=True)

        # image_gray = cv2.cvtColor(bw_edges, cv2.COLOR_BGR2GRAY)

        # [theta_data, M] = gradient_map(image_gray)
        [theta_data, M] = orientation_map(bw_edges, 7)
        theta_data += np.pi / 2
        theta_data /= np.pi
        theta_uint8 = theta_data * 255
        theta_uint8 = np.uint8(theta_uint8)
        theta_uint8 = cv2.applyColorMap(theta_uint8, cv2.COLORMAP_JET)
        theta_view = np.zeros(theta_uint8.shape)
        theta_view = np.uint8(theta_view)
        theta_view[M > 0.2] = theta_uint8[M > 0.2]

        cv2.imshow("Image", theta_view)
        cv2.waitKey(0)

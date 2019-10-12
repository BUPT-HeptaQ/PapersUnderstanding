import numpy as np
import os
import cv2

from PIL import Image
import time

import mxnet as mx
import mxnet.ndarray as nd

from sklearn import preprocessing
# from sklearn.model_selection import KFold


# load the face vectorizer model
def load_model(prefix='./insightFace_model/model', epoch=0, batch_size=32):
    symbol, arg_params, auxiliary_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = symbol.get_internals()
    output_layer = all_layers['fc1_output']
    context = mx.gpu(0)
    model = mx.mod.Module(symbol=output_layer, context=context)
    model.bind(data_shapes=[('data', (batch_size, 3, 112, 112))])
    model.set_params(arg_params, auxiliary_params)
    return model


class FaceVectorizer(object):
    def __init__(self):
        self.batch_size = 64
        self.model = load_model(batch_size=self.batch_size)

    def get_feed_data(self, image_4d_array):
        image_list = []
        for image_3d_array in image_4d_array:
            height, width, _ = image_3d_array.shape
            if height != 112 or width != 112:
                image_3d_array = cv2.resize(image_3d_array, (112, 112))
            image_list.append(image_3d_array)

        image_4d_array_1 = np.array(image_list)
        image_4d_array_2 = np.transpose(image_4d_array_1, [0, 3, 1, 2])
        image_4d_Array = nd.array(image_list)
        image_quantity = len(image_list)
        label_1D_Array = nd.ones((image_quantity, ))
        feed_data = mx.io.DataBatch(data=(image_4d_Array, ), label=(label_1D_Array, ))

        return feed_data

    def get_embedding_vector(self, image_ndarray):
        if len(image_ndarray.shape) == 3:
            image_ndarray = np.expand_dims(image_ndarray, 0)
        assert len(image_ndarray.shape) == 4, 'image_ndarray shape length is not 4'
        image_quantity = len(image_ndarray)

        if self.batch_size != image_quantity:
            self.model = load_model(batch_size=image_quantity)
            self.batch_size = image_quantity

        feed_data = self.get_feed_data(image_ndarray)
        self.model.forward(feed_data, is_train=False)
        net_outputs = self.model.get_outputs()
        embedding_vector = net_outputs[0].asnumpy()
        embedding_vector_1 = preprocessing.normalize(embedding_vector)

        return embedding_vector_1


class FaceRecognizer(object):
    def __init__(self, face_dir_path='./face_database'):
        self.face_vectorizer = FaceVectorizer()
        self.load_database(face_dir_path)

    def load_database(self, face_dir_path='./face_database'):
        person_name_list = next(os.walk(face_dir_path))[1]
        self.person_name_list = person_name_list
        person_id_list = []

        for i, person_name in enumerate(person_name_list):
            dir_path = os.path.join(face_dir_path, person_name)
            file_name_list = next(os.walk(dir_path))[2]
            file_path_list = [os.path.join(dir_path, k) for k in file_name_list
                              if k.endswith('jpg') or k.endswith('png') or k.endswith('bmp')]
            for file_path in file_path_list:
                person_id_list.append(i)
        image_quantity = len(person_id_list)

        # load image data turn into vector
        batch_size = 20
        image_data_list = []
        count = 0
        self.embedding_matrix = np.empty((image_quantity, 512))

        for i, person_name in enumerate(person_name_list):
            dir_path = os.path.join(face_dir_path, person_name)
            file_name_list = next(os.walk(dir_path))[2]
            file_path_list = [os.path.join(dir_path, k) for k in file_name_list
                              if k.endswith('jpg') or k.endswith('png') or k.endswith('bmp')]
            for file_path in file_path_list:
                image_data = np.array(Image.open(file_path))
                image_data_list.append(image_data)
                count += 1
                if count % batch_size == 0:
                    image_ndarray = np.array(image_data_list)
                    image_data_list.clear()
                    self.embedding_matrix[count-batch_size: count] = \
                        self.face_vectorizer.get_embedding_vector(image_ndarray)

        if count % batch_size != 0:
            image_ndarray = np.array(image_data_list)
            reminder = count % batch_size
            self.embedding_matrix[count-batch_size: count] = \
                self.face_vectorizer.get_embedding_vector(image_ndarray)

        self.person_id_ndarray = np.array(person_id_list)
        print("Finish load the face data set !")
        print("in face data set has %d person, and %d face images" % (len(person_name_list), image_quantity))

    def get_person_name(self, embedding_vector):
        diff_matrix = np.subtract(self.embedding_matrix, embedding_vector)
        distance_vector = np.sum(np.square(diff_matrix), 1)
        threshold = 1.4
        bool_ndarray = np.less(distance_vector, threshold)
        person_id_ndarray = self.person_id_ndarray[bool_ndarray]

        if len(person_id_ndarray) == 0:
            return None
        else:
            count_ndarray = np.bincount(person_id_ndarray)
            max_count_index = np.argmax(count_ndarray)
            person_name = self.person_name_list[max_count_index]
            return person_name

    def get_person_name_list(self, image_ndarray):
        embedding_vectors = self.face_vectorizer.get_embedding_vector(image_ndarray)
        person_name_list = []
        for embedding_vector in embedding_vectors:
            person_name = self.get_person_name(embedding_vector)
            person_name_list.append(person_name)

        return person_name_list


face_recognizer = FaceRecognizer()

start_time = time.time()
vector = np.random.rand(1, 512)
feature_matrix = np.random.rand(2000, 512)
print(feature_matrix.shape)
diff_matrix = np.subtract(feature_matrix, vector)
sum_vector = np.sum(np.square(diff_matrix), 1)
print(sum_vector.shape)
used_time = time.time() - start_time
print("total cost %.6f seconds" % used_time)


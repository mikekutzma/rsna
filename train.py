import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import keras
import keras_preprocessing.image as KPImage
from PIL import Image
import pydicom
from keras.applications.resnet50 import ResNet50 as PTModel, preprocess_input
from keras.layers import Input

# Load data 
image_bbox_df = pd.read_csv('./input/stage_1_image_bbox_full.csv')
with open('params.json') as f:
    params = json.load(f)

# Encode labels
class_enc = LabelEncoder()
image_bbox_df['class_idx'] = class_enc.fit_transform(image_bbox_df['class'])
oh_enc = OneHotEncoder(sparse=False)
image_bbox_df['class_vec'] = oh_enc.fit_transform(image_bbox_df['class_idx']. \
                                                  values.reshape(-1, 1)).tolist()

# Create train/validation dsets
train_df, val_df = train_test_split(image_bbox_df,
                                    stratify=image_bbox_df['class_idx'])

# Balance training data
train_df = train_df.groupby('class_idx'). \
    apply(lambda x: x.sample(params['TRAIN_SIZE'] // 3)). \
    reset_index(drop=True)


# Fix Keras for DICOM
def read_dicom(infile):
    img = pydicom.read_file(infile).pixel_array
    return img / img.max()


class MedicalPIL:
    @staticmethod
    def open(infile):
        if infile.endswith('.dcm'):
            char_slice = read_dicom(infile)
            int_slice = (255 * char_slice).clip(0, 255).astype(np.uint8)
            return Image.fromarray(int_slice)
        return Image.open(infile)

    fromarray = Image.fromarray


KPImage.pil_image = MedicalPIL

# Prepare datasets
img_gen_params = dict(horizontal_flip=True,
                      height_shift_range=0.05,
                      width_shift_range=0.02,
                      rotation_range=3.0,
                      shear_range=0.01,
                      zoom_range=0.05,
                      preprocessing_function=preprocess_input
                      )
img_gen = KPImage.ImageDataGenerator(**img_gen_params)


def flow_from_dataframe(img_data_gen, in_df, path_col, y_col,
                        seed=None, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    df_gen = img_data_gen.flow_from_directory(base_dir,
                                              class_mode='sparse',
                                              seed=seed,
                                              **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values, 0)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = ''  # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen

#For training
train_gen = flow_from_dataframe(img_gen, train_df,
                                path_col='path',
                                y_col='class_vec',
                                target_size=params['IMG_SIZE'],
                                color_mode='rgb',
                                batch_size=params['BATCH_SIZE'])
#For validation
val_gen = flow_from_dataframe(img_gen, val_df,
                              path_col='path',
                              y_col='class_vec',
                              target_size=params['IMG_SIZE'],
                              color_mode='rgb',
                              batch_size=256)
#For test
valid_X, valid_Y = next(flow_from_dataframe(img_gen, val_df,
                                            path_col='path',
                                            y_col='class_vec',
                                            target_size=params['IMG_SIZE'],
                                            color_mode='rgb',
                                            batch_size=params['TEST_SIZE']))


# Build model
t_x, t_y = next(train_gen)

#Base
base_model = PTModel(input_shape=t_x.shape[1:],include_top=False)
base_model.trainable = False

#

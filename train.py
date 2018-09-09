print('Loading Modules..')
import pandas as pd
import numpy as np
import os
import json
import sys
from glob import glob
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import keras
import keras_preprocessing.image as KPImage
from PIL import Image
import pydicom
from keras.applications.resnet50 import ResNet50 as PTModel, preprocess_input
from keras.layers import (Input, BatchNormalization, GlobalAveragePooling2D,
                          Dropout, Dense)
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Load data 
print('Loading Data..')
image_bbox_df = pd.read_csv('./input/stage_1_image_bbox_full.csv')
with open('params.json') as f:
    params = json.load(f)

# Encode labels
print('Encoding labels..')
class_enc = LabelEncoder()
image_bbox_df['class_idx'] = class_enc.fit_transform(image_bbox_df['class'])
oh_enc = OneHotEncoder(sparse=False)
image_bbox_df['class_vec'] = oh_enc.fit_transform(
    image_bbox_df['class_idx'].values.reshape(-1, 1)).tolist()

# Create train/validation dsets
print('Creating train/validation sets..')
train_df, val_df = train_test_split(image_bbox_df,
                                    stratify=image_bbox_df['class_idx'])

# Balance training data
print('Balancing Data..')
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
print('Preparing Data..')
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


# For training
train_gen = flow_from_dataframe(img_gen, train_df,
                                path_col='path',
                                y_col='class_vec',
                                target_size=params['IMG_SIZE'],
                                color_mode='rgb',
                                batch_size=params['BATCH_SIZE'])
# For validation
val_gen = flow_from_dataframe(img_gen, val_df,
                              path_col='path',
                              y_col='class_vec',
                              target_size=params['IMG_SIZE'],
                              color_mode='rgb',
                              batch_size=256)
# For test
valid_X, valid_Y = next(flow_from_dataframe(img_gen, val_df,
                                            path_col='path',
                                            y_col='class_vec',
                                            target_size=params['IMG_SIZE'],
                                            color_mode='rgb',
                                            batch_size=params['TEST_SIZE']))

# Build model
print('Building Model Structure..')
t_x, t_y = next(train_gen)

# Base
base_model = PTModel(input_shape=t_x.shape[1:], include_top=False)
base_model.trainable = False

# Attentional
base_shape = base_model.get_output_shape_at(0)[1:]
layers = [Input(base_shape, name='feature_input')]
layers.append(BatchNormalization()(layers[-1]))
layers.append(GlobalAveragePooling2D()(layers[-1]))
layers.append(Dropout(params['DROPOUT'])(layers[-1]))
layers.append(Dense(params['DENSE_COUNT'], activation='elu')(layers[-1]))
layers.append(Dropout(params['DROPOUT'])(layers[-1]))
layers.append(Dense(t_y.shape[1], activation='softmax')(layers[-1]))
attn_model = Model(inputs=[layers[0]], outputs=[layers[-1]],
                   name='trained_model')

# Stitch Models Together
model = Sequential(name='combined_model')
model.add(base_model)
model.add(attn_model)
model.compile(optimizer=Adam(lr=params['LEARNING_RATE']),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# Set up training callbacks
weight_file = "lung_opacity_weights.best.hd5"
checkpoint = ModelCheckpoint(weight_file, verbose=1, save_best_only=True,
                             save_weights_only=True)
reduceLR = ReduceLROnPlateau(factor=0.8, verbose=1, cooldown=5, min_lr=0.0001)
earlystop = EarlyStopping(patience=10)
callbacks = [checkpoint, reduceLR, earlystop]

# Train model
if 'train' in sys.argv:
    print('Fitting Model')
    model.fit_generator(train_gen, validation_data=(valid_X, valid_Y),
                        epochs=20, callbacks=callbacks, workers=2)
else:
    print('Not fitting model')

# Save model
model.load_weights(weight_file)
model.save('full_model.h5')

# Make prediction
inputdir = './input'
test_dicom_dir = '/'.join([inputdir, 'stage_1_test_images'])

sub_df = pd.DataFrame({'path': glob(os.path.join(test_dicom_dir, '*.dcm'))})
sub_df['patientId'] = sub_df['path'].map(
    lambda x: os.path.splitext(os.path.basename(x))[0])

sub_gen = flow_from_dataframe(img_gen, sub_df,
                              path_col='path',
                              y_col='patientId',
                              target_size=params['IMG_SIZE'],
                              color_mode='rgb',
                              batch_size=params['BATCH_SIZE'],
                              shuffle=False)

steps = 2*sub_df.shape[0]//params['BATCH_SIZE']
out_ids, out_vec = [], []
print("Making prediction..")
for _, (t_x, t_y) in zip(tqdm(range(steps)), sub_gen):
    out_vec += [model.predict(t_x)]
    out_ids += [t_y]
out_vec = np.concatenate(out_vec, 0)
out_ids = np.concatenate(out_ids, 0)

pred_df = pd.DataFrame(out_vec, columns=class_enc.classes_)
pred_df['patientId'] = out_ids
pred_avg_df = pred_df.groupby('patientId').agg('mean').reset_index()
print("Saving submission..")
pred_avg_df['PredictionString'] = pred_avg_df['Lung Opacity'].map(
    lambda x: ('%2.2f 0 0 1024 1024' % x) if x>0.5 else '')
sub_file = 'submission.csv'
pred_avg_df[['patientId', 'PredictionString']].to_csv(sub_file, index=False)
print("Submission saves as",sub_file)
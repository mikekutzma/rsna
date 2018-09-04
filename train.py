import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# Load data 
image_bbox_df = pd.read_csv('./input/stage_1_image_bbox_full.csv')
with open('params.json') as f:
	params = json.load(f)
#params = {'TRAIN_SIZE':8000}

# Encode labels
class_enc = LabelEncoder()
image_bbox_df['class_idx'] = class_enc.fit_transform(image_bbox_df['class'])
oh_enc = OneHotEncoder(sparse=False)
image_bbox_df['class_vec'] = oh_enc.fit_transform(image_bbox_df['class_idx'].values.reshape(-1,1)).tolist()

# Create train/validation dsets
train_df, val_df = train_test_split(image_bbox_df,stratify=image_bbox_df['class_idx'])

# Balance training data
train_df = train_df.groupby('class_idx').\
	apply(lambda x: x.sample(params['TRAIN_SIZE']//3)).\
	reset_index(drop=True)

print(train_df.shape)
print(val_df.shape)

import pandas as pd
import pydicom
from glob import glob
import os

inputdir = './input'
dicom_dir = '/'.join([inputdir,'stage_1_train_images'])

bbox_df = pd.read_csv('./input/stage_1_train_labels.csv')
det_info_df= pd.read_csv('./input/stage_1_detailed_class_info.csv')
bbox_df['class'] = det_info_df['class']

image_df = pd.DataFrame({'path':glob(os.path.join(dicom_dir,'*.dcm'))})
image_df['patientId'] = image_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])

DCM_TAG_LIST = ['PatientAge', 'BodyPartExamined', 'ViewPosition', 'PatientSex']
def get_tags(in_path):
	c_dicom = pydicom.read_file(in_path, stop_before_pixels=True)
	tag_dict = {c_tag: getattr(c_dicom, c_tag, '') for c_tag in DCM_TAG_LIST}
	tag_dict['path'] = in_path
	return pd.Series(tag_dict)

image_meta_df = image_df.apply(lambda x: get_tags(x['path']), 1)
image_meta_df['PatientAge'] = image_meta_df['PatientAge'].map(int)

image_full_df = pd.merge(image_df,image_meta_df,on='path')
image_bbox_df = pd.merge(bbox_df, image_full_df, on='patientId',how='left')

outfile = '/'.join([inputdir,'stage_1_image_bbox_full.csv'])
image_bbox_df.to_csv(outfile, index=False)
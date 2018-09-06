import pandas as pd

image_bbox_df = pd.read_csv('./input/stage_1_image_bbox_full.csv')

print(image_bbox_df.sample(3))

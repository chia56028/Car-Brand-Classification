import os
import pandas as pd

root_dir = './cs-t0828-2020-hw1/'
label_path = root_dir+'training_labels.csv'

label_df = pd.read_csv(label_path, names=["label"])

sector = label_df.groupby('label')

img_dir = root_dir+'train/'
k = 0
for i in sector:
    car_label = i[0]
    print(car_label)
    if not os.path.isdir(img_dir+car_label) and car_label!='label':
        os.makedirs(img_dir+car_label)

    q = label_df.query("label == @car_label")
    q_ind = list(q.index)
    print(q_ind)

    for ind in q_ind:
        img_path = img_dir+car_label+"/"+ind+".jpg"
        if not os.path.isfile(img_path):
            os.system("cp cs-t0828-2020-hw1/training_data/training_data/" +
                      ind+".jpg"+" '"+img_path+"'")

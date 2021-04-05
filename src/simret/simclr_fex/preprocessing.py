import os
import re
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

def make_data_index(image_dir, test_size=0.5):
    image_dir = image_dir.replace("\\", "/")

    images = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if (re.search(r"([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$", f))
    ]

    df = pd.DataFrame(data={"Image_Paths": images})
    train, test = train_test_split(df, test_size=test_size, random_state=324)
    train['Subset'] = 'train'
    test['Subset'] = 'test'
    df = pd.concat((train, test)).reset_index(drop=True)
    return df

def main():
    df = make_data_index(image_dir="train/train2017",test_size=0.5)
    df.to_csv("train/test-train-split.csv")
    dr = "train/train2017"
    if not os.path.exists(dr+"/test"):
        os.mkdir(dr+"/test")

    if not os.path.exists(dr+"/train"):
        os.mkdir(dr+"/train")
    for subset in ['train','test']:
        for i, row in df.loc[lambda df: df.Subset==subset].iterrows():
            # print(row['Image_Paths'],os.path.join(os.path.join(dr,subset),os.path.basename(row['Image_Paths'])))
            shutil.move(row['Image_Paths'],os.path.join(os.path.join(dr,subset),os.path.basename(row['Image_Paths'])))

if __name__ == "__main__":
    main()

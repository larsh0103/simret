import os
import re
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial.distance import cosine
from sklearn.model_selection import train_test_split
from torchvision import transforms

from simret.simclr_fex.models.resnet_simclr import ResNetSimCLR


def make_train_test(image_dir, test_size, random_state=None):
    image_dir = image_dir.replace("\\", "/")

    images = [f for f in os.listdir(image_dir) if (re.search(r"([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$", f))]

    # labels = [f for f in os.listdir(image_dir) if f.endswith(".xml")]

    # images = [f.split(".xml")[0] + ".png" for f in labels]

    # Ensure very image has a label file
    train, test = train_test_split(images, random_state=random_state, test_size=test_size)
    # train_labels = [os.path.splitext(filename)[0] + ".xml" for filename in train]
    # test_labels = [os.path.splitext(filename)[0] + ".xml" for filename in test]

    train_df = pd.DataFrame(data={"Image_Paths": train, "subset": "train"})
    test_df = pd.DataFrame(data={"Image_Paths": test, "subset": "test"})
    # train_df["Image_Paths"] = image_dir + "/" + train_df["Image_Paths"]
    # test_df["Image_Paths"] = image_dir + "/" + test_df["Image_Paths"]

    # train_df["Label_Paths"] = image_dir + "/" + train_df["Label_Paths"]
    # test_df["Label_Paths"] = image_dir + "/" + test_df["Label_Paths"]

    df = pd.concat((train_df, test_df)).reset_index(drop=True)

    return df


def filter_duplicates(image_df: pd.DataFrame, model_path: str, threshold: float = 0.3):
    """
        Uses the simclr trained feature extractor to identify identical or nearly identical images,
        using cosine similarity on the feature vectors,
        and ensure that they do not appear in the test/validation sets.
        ---------
        Arguments:
            image_df : pandas Dataframe with paths to image files and subdived into train and test subsets.
            model_path : string with path to torch model dict file
            threshold : similarity score threshold for which scores that are smaller than this
                        will be considered too similar, and exluded from the test set.
        Returns:
            revised dataframe with all images considered duplicates assigned to the train set.
    """

    images = image_df.Image_Paths
    image_list = []
    num_images = len(image_df)
    for i in range(num_images):
        image = Image.open(images[i])
        image = np.expand_dims(image, 2)
        image = np.concatenate((image, image, image), axis=2)
        image_list.append(Image.fromarray(image))

    print("num images", num_images)

    features = compute_features(image_list, model_path=model_path)

    t = time.time()
    start = 0
    similarity = np.ones((num_images, num_images))
    for i in range(num_images):
        for j in range(start, num_images):
            if i == j:
                similarity[i, j] = 1
            else:
                similarity[i, j] = cosine(features[i, :], features[j, :])
        start += 1
    nt = time.time()
    print(f"it took {nt-t} to run")

    duplicate_images = []

    for i, j in zip(list(np.where(similarity < threshold)[0]), list(np.where(similarity < threshold)[1])):

        print(f"Image: {images[i]} is similar or identical to Image: {images[j]}")
        duplicate_images.append(images[i])
        duplicate_images.append(images[j])

    duplicate_images = list(set(duplicate_images))
    print(image_df)
    image_df[image_df.Image_Paths.isin(duplicate_images)]["subset"] = "train"
    return image_df


def compute_features(images, model_path):

    preprocess = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
    model = ResNetSimCLR(base_model="resnet18", out_dim=256)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # model = models.resnet50(pretrained=False)
    # model.load_state_dict(torch.load(model_path),strict=False)

    input_batch = torch.zeros((len(images), 3, 256, 256), dtype=torch.float32)
    for i, image, in enumerate(images):
        input_tensor = preprocess(image)
        input_tensor = input_tensor / 255
        input_batch[i, :, :, :] = input_tensor

    batch_size = 32
    features = np.zeros((1, 512))
    batches = np.arange(0, len(images))[::batch_size]

    for i in range(len(batches)):

        if torch.cuda.is_available():
            batch = input_batch[batches[i] : batches[i] + batch_size, :, :, :].to("cuda")
            model.to("cuda")

        with torch.no_grad():
            fts = model(batch)[0].detach().cpu().numpy()
        features = np.concatenate((features, fts), 0)
    return features[1:, :]


def is_image(im_path):
    return re.search(r"([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$", im_path)


def get_image_paths(dir):
    images = [os.path.join(dir, f) for f in os.listdir(dir) if is_image(f)]
    return images


def xml_2_csv(data_index_file: pd.DataFrame(), class_name="TRNSVIEW"):
    """Iterates through all .xml files (generated by labelImg) in a given dataframe and combines them in a single Pandas dataframe.

    Parameters:
    ----------
    data_index_file : {pandas dataframe}
        The dataframe containing a column named Label_Paths with paths to xml files
    Returns
    -------
    Pandas DataFrame
        The produced dataframe
    """
    xml_list = []
    i = 0
    for xml_file in data_index_file.Label_Paths:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if not root.findall("object"):
            continue

        for member in root.findall("object"):
            # root.find('path').text.split,
            value = (
                os.path.join(os.getcwd(), data_index_file.Image_Paths.to_list()[i].split("./")[-1].replace("/", "\\")),
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text),
                member[0].text,
                int(root.find("size")[0].text),
                int(root.find("size")[1].text),
            )
            if member[0].text == class_name:
                xml_list.append(value)
        i += 1
    # csv with img_file, x1, y1, x2, y2, class_name
    column_name = ["img_file", "x1", "x2", "x2", "y2", "class_name", "width", "height"]
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

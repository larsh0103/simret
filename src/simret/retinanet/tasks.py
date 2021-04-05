import datetime
import os

import luigi
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from simret.simclr_fex import tasks as simclr_fex_tasks
from simret.train import preprocessing, train, utils


def get_config():
    return luigi.configuration.get_config()


class Train(luigi.WrapperTask):
    date = luigi.DateParameter(default=datetime.date.today())

    def requires(self):
        tasks = [TrainODModel(date=self.date)]
        return tasks


class PartitionDataset(luigi.Task):

    config = get_config()
    date = luigi.DateParameter()
    imageDir = luigi.Parameter(default=config["paths"]["imageDir"])
    test_size = luigi.Parameter(default=float(config["train-test"]["test_size"]))
    random_state = luigi.Parameter(default=int(config["train-test"]["random_state"]))

    def requires(self):
        return []

    def output(self):
        fn = utils.get_build_dir(date=self.date) / str("data-index-file.parquet")
        return luigi.LocalTarget(str(fn), luigi.format.Nop)

    def load(self):
        with self.output().open("rb") as fp:
            return pd.read_parquet(fp)

    def run(self):
        df = preprocessing.make_train_test(self.imageDir, test_size=self.test_size, random_state=self.random_state)

        print(df.__len__, "numer of images found")
        with self.output().temporary_path() as fp:
            df.to_parquet(fp)


class FilterDuplicateImages(luigi.Task):

    config = get_config()
    date = luigi.DateParameter()
    imageDir = luigi.Parameter(default=config["paths"]["imageDir"])

    def requires(self):
        return {"data-index-file": PartitionDataset(date=self.date), "simclr_model": simclr_fex_tasks.TrainModel(date=self.date)}

    def output(self):
        fn = utils.get_build_dir(date=self.date) / str("data-index-file-filtered.parquet")
        return luigi.LocalTarget(str(fn), luigi.format.Nop)

    def load(self):
        with self.output().open("rb") as fp:
            return pd.read_parquet(fp)

    def run(self):
        data_index_file = self.requires()["data-index-file"].load()
        simclr_model_path = self.requires()["simclr_model"].output().fn

        df = preprocessing.filter_duplicates(data_index_file, simclr_model_path)
        with self.output().temporary_path() as fp:
            df.to_parquet(fp)


class TrainODModel(luigi.Task):

    date = luigi.DateParameter()
    ann_file = luigi.Parameter(default= get_config()['train-test']['ann_file'])
    img_dir = luigi.Parameter(default= get_config()['paths']['imageDir'])
    val_ann_file = luigi.Parameter(default= get_config()['train-test']['val_ann_file'])
    val_img_dir = luigi.Parameter(default= get_config()['paths']['val_imageDir'])
    epochs = luigi.IntParameter(default=int(get_config()["train-test"]["epochs"]))
    backbone_name = luigi.Parameter(default="resnet18")
    simclr_pretraining = luigi.BoolParameter(default=get_config()['model']['simclr_pretraining'])
    checkpoint = get_config()["model"]["checkpoint"]

    def requires(self):
        if self.simclr_pretraining:             
            return {
                "simclr_model_dict": simclr_fex_tasks.TrainModel(date=self.date),
                "data_index_file": PartitionDataset(date=self.date),
            }
        else:
            return []

    def output(self):
        # returns path to checkpoint of final epoch
        fn = utils.get_build_dir(date=self.date) / f"model-e{(self.epochs - 1):03}.pth"
        return luigi.LocalTarget(str(fn), luigi.format.Nop)

    # TODO:
    # def load(self):
    #     with self.output().open("rb") as fp:
    #         return torch.load(model_path)

    def run(self):
        if self.simclr_pretraining:
            simclr_model_dict = self.requires()["simclr_model_dict"].output().fn
        else:
            simclr_model_dict = None
        
        # class_names = {'GA':0,'NON_GA':1}
        class_names = {"POKEMON": 0}
        model_dict = train.train(
            backbone_name=self.backbone_name,
            ann_file=self.ann_file,
            val_ann_file=self.val_ann_file,
            checkpoints_folder=str(utils.get_build_dir(date=self.date)) + "/checkpoints",
            img_dir=self.img_dir,
            val_img_dir=self.val_img_dir,
            simclr_model_dict=simclr_model_dict,
            class_names=class_names,
            epochs=self.epochs,
            checkpoint=self.checkpoint,
        )
        with self.output().temporary_path() as fp:
            torch.save(model_dict, fp)


class ConvertLabelsFromXML2CSV(luigi.Task):
    subset = luigi.ChoiceParameter(choices=["train", "test"])
    date = luigi.DateParameter()

    def requires(self):
        return {"data-index-file": FilterDuplicateImages(date=self.date)}

    def output(self):
        fn = utils.get_build_dir(date=self.date) / str(f"{self.subset}.csv")
        return fn

    def load(self):
        return self.output()
        # with self.output().open('rb') as fp:
        #     return pd.read_csv(fp)

    def run(self):
        data_index_file = (
            self.requires()["data-index-file"].load().loc[lambda df: df.subset == self.subset].reset_index(drop=True)
        )
        config = get_config()
        class_name = pd.read_csv(config["train-test"]["classes_csv"]).columns[0]

        df = preprocessing.xml_2_csv(data_index_file, class_name=class_name)
        df.to_csv(self.output(), header=False, index=False)


class CrossValidate(luigi.Task):
    date = luigi.DateParameter(default=datetime.date.today())
    n_splits = luigi.IntParameter(default=5)
    epochs = luigi.IntParameter(default=int(get_config()["train-test"]["epochs"]))

    def requires(self):
        return {
            "train_csv": ConvertLabelsFromXML2CSV(date=self.date, subset="train"),
            "test_csv": ConvertLabelsFromXML2CSV(date=self.date, subset="test"),
            "simclr_model": simclr_fex_tasks.TrainModel(date=self.date),
        }

    def output(self):
        # returns path to checkpoint of final epoch
        fn = utils.get_build_dir(date=self.date) / "crossvalidation-results.csv"
        return luigi.LocalTarget(str(fn), luigi.format.Nop)

    # TODO:
    # def load(self):
    #     with self.output().open("rb") as fp:
    #         return torch.load(model_path)

    def run(self):

        kf = KFold(n_splits=self.n_splits)

        X = pd.read_csv(self.requires()["train_csv"].load(), header=None)
        val_regression_loss = []
        val_classification_loss = []
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            csv_train = os.path.join(utils.get_build_dir(date=self.date), f"kf_train_{str(i)}.csv")
            csv_val = os.path.join(utils.get_build_dir(date=self.date), f"kf_val_{str(i)}.csv")
            X.iloc[train_index].reset_index(drop=True).to_csv(csv_train, header=False, index=False)
            X.iloc[test_index].reset_index(drop=True).to_csv(csv_val, header=False, index=False)
            config = get_config()
            csv_classes = config["train-test"]["classes_csv"]
            depth = int(config["model"]["depth"])
            feature_extractor_model_path = self.requires()["simclr_model"].output().fn

            retinanet_dict = train.train_model(
                feature_extractor_model_path=feature_extractor_model_path,
                csv_train=csv_train,
                csv_classes=csv_classes,
                csv_val=csv_val,
                batch_size=6,
                depth=depth,
                epochs=self.epochs,
                checkpoints_folder=utils.get_build_dir(date=self.date) / "checkpoints",
            )

            val_regression_loss.append(retinanet_dict["regression_loss"])
            # epochs.append(np.arange(0, self.epochs))
            val_classification_loss.append(retinanet_dict["classification_loss"])

        validation_results = pd.DataFrame(
            data={"RegressionLoss": val_regression_loss, "ClassificationLoss": val_classification_loss, "Split": np.arange(5)}
        )

        with self.output().temporary_path() as fp:
            validation_results.to_csv(fp)

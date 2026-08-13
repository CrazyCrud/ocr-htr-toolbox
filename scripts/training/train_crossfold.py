import argparse
from collections import Counter
from datetime import datetime
import mlflow
import pandas as pd
from pathlib import Path
import random
import shutil
from sklearn.model_selection import KFold
import sys
from tqdm import tqdm
from ultralytics import YOLO
import yaml

tracking_uri = "http://localhost:5000"
mlflow.set_tracking_uri(tracking_uri)

script_dir = Path(__file__).parent
runs_dir = script_dir.parent / "runs"


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("params", help="Path to a standalone params.yaml", type=Path)
    parser.add_argument("section", help="Section inside params.yaml", type=str)
    parser.add_argument("--existing_fold", type=Path, default=None)
    args = parser.parse_args()

    all_sections = None
    with open(args.params, "r", encoding="utf-8") as params_file:
        all_sections = yaml.safe_load(params_file)

    section = args.section
    config = all_sections[section]

    mlflow.end_run()

    # Section fields
    experiment_config = config["experiment"]
    model_config = config["model"]
    data_config = config["data"]
    train_config = config["training"]
    dataset_path_str = script_dir.parent / data_config["dataset_path"]

    # See https://docs.ultralytics.com/guides/kfold-cross-validation
    dataset_path = Path(dataset_path_str)

    labels = sorted(dataset_path.rglob("*labels/*.txt")) 

    yaml_file = dataset_path / "yolo.yaml"
    with open(yaml_file, encoding="utf8") as y:
        classes = yaml.safe_load(y)["names"]
    print(classes)
    cls_idx = list(range(len(classes)))

    index = [label.stem for label in labels] 
    labels_df = pd.DataFrame([], columns=cls_idx, index=index)

    label_data = []
    for label in labels:
        lbl_counter = Counter()
        with open(label, 'r') as lf:
            for l in lf:
                lbl_counter[int(l.split(' ')[0])] += 1
        label_data.append({'label': label.stem, **dict(lbl_counter)})

    labels_df = pd.DataFrame(label_data).set_index('label')

    labels_df = labels_df.reindex(columns=cls_idx).fillna(0.0)

    random.seed(0)
    ksplit = 5
    kf = KFold(n_splits=ksplit, shuffle=True, random_state=20) 

    kfolds = list(kf.split(labels_df))

    folds = [f"split_{n}" for n in range(1, ksplit + 1)]
    folds_df = pd.DataFrame(index=index, columns=folds)

    for i, (train, val) in enumerate(kfolds, start=1):
        folds_df[f"split_{i}"].loc[labels_df.iloc[train].index] = "train"
        folds_df[f"split_{i}"].loc[labels_df.iloc[val].index] = "val"

    fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()

        ratio = val_totals / (train_totals + 1e-7)
        fold_lbl_distrb.loc[f"split_{n}"] = ratio

    supported_extensions = [".jpg", ".jpeg", ".png"]

    images = []

    for ext in supported_extensions:
        images.extend(sorted((dataset_path / "images").rglob(f"*{ext}")))

    timestamp = f"{datetime.now():%Y%m%d_%H%M%S}"

    if args.existing_fold:
        ds_yamls = [args.existing_fold]
    else:
        save_path = Path(dataset_path / f"{timestamp}_{ksplit}-Fold_Cross-val")
        save_path.mkdir(parents=True, exist_ok=True)
        ds_yamls = []

        for split in folds_df.columns:
            split_dir = save_path / split
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
            (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
            (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
            (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

            dataset_yaml = split_dir / f"{split}_dataset.yaml"
            ds_yamls.append(dataset_yaml)

            with open(dataset_yaml, "w") as ds_y:
                yaml.safe_dump(
                    {
                        "path": split_dir.as_posix(),
                        "train": "train",
                        "val": "val",
                        "names": classes,
                    },
                    ds_y,
                )

        for image, label in tqdm(zip(images, labels), total=len(images), desc="Copying files"):
            image_stem = image.stem

            for split in folds_df.columns: 
                k_split = folds_df.at[image_stem, split]

                img_to_path = save_path / split / k_split / "images"
                lbl_to_path = save_path / split / k_split / "labels"

                shutil.copy(image, img_to_path / image.name)
                shutil.copy(label, lbl_to_path / label.name)

    model = YOLO(model_config["architecture"])

    epochs = train_config["epochs"]
    exp_name = experiment_config["name"]
    batch = train_config["batch_size"]
    imgsize = data_config["image_size"]
    lr0 = train_config["learning_rate"]
    seed = experiment_config["seed"]
    device = train_config.get("device", "cpu")

    architecture = model_config["architecture"]
    architecture_path = Path(architecture)
    model_type = model_config["type"].lower()

    if not architecture_path.is_file():
        sys.exit(f"{architecture_path} is not valid")

    print(f"Set experiment name to {exp_name}")
    exp = mlflow.set_experiment(experiment_name=exp_name)
    print(f"Experiment_id: {exp.experiment_id}")
    print(f"Artifact Location: {exp.artifact_location}")

    run_name_base = f"{exp_name}_{timestamp}"

    if model_type == "yolo":
        for k, dataset_yaml in enumerate(ds_yamls):
            run_name = f"{run_name_base}_fold_{k + 1}"
            with mlflow.start_run(experiment_id=exp.experiment_id, run_name=run_name):
                print(f"Training fold {k + 1} / {len(ds_yamls)}")

                model = YOLO(architecture_path)
                model.train(
                    data=dataset_yaml, 
                    epochs=epochs, 
                    batch=batch, 
                    name=run_name,
                    lr0=lr0,
                    imgsz=imgsize,
                    seed=seed,
                    device=device,
                    project=run_name,
                    workers=8
                )  

            mlflow.end_run()
    elif model_type == "rf-detr":
        pass


if __name__ == "__main__":
    train()
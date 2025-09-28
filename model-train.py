from ultralytics import YOLO
import os
from roboflow import Roboflow
import glob
from pyaml_env import parse_config, BaseConfig

config = BaseConfig(parse_config('configs/project_config.yaml'))
dataset_name = config.project_settings.set_dataset
use_dataset = "datasets/{}".format(dataset_name)
use_model = config.project_settingsmodel_train # detection: yolov11n, yolov11s, yolov11m, yolov11l, yolov11x /
# segmentation: yolo11n-seg, yolo11s-seg, yolo11m-seg, yolo11l-seg, yolo11x-seg
model_file = "models/{}_{}.pt".format(dataset_name, use_model)
new_model_file = model_file
retrain_model_file = model_file
training_yaml = "configs/train_conf.yaml"

# Somewhat frequently changed model settings:
set_epochs = 500
set_image_size = 640


# Roboflow dataset download
def download_dataset():
    rf = Roboflow(api_key=config.roboflow.roboflow_api_key)
    project = rf.workspace(config.roboflow.roboflow_workspace).project(config.roboflow.project_name)
    version = project.version(4)
    dataset = version.download("yolov11")

def model_train():
    if os.path.exists("models"):
        pass
    else:
        os.mkdir("models")

    print("\n1. To train anew\n2. To resume training\n3. To retrain")
    train_choice = input("Your choice:")

    # Training anew
    if train_choice == "1":
        # Train the model from scratch
        model = YOLO("{}.yaml".format(use_model))  # build a new model from YAML

        model.train(cfg=training_yaml,
            data="{}/data.yaml".format(use_dataset),
            epochs=set_epochs, # x epochs for training
            imgsz=set_image_size, # 640 = 640x640 image sizes for training
                    )
        model.save(model_file)


    # Training resume
    elif train_choice == "2":
        # Find the newest last.pt file
        base_dir = "runs/detect"
        pattern = os.path.join(base_dir, "train*/weights/*.pt")

        files = glob.glob(pattern)

        if not files:
            print("No last.pt files found.")
        else:
            # Get the newest one by creation/modification time
            newest_file = max(files, key=os.path.getctime).replace("\\", "/")
            resume_model_file = newest_file

            # Check if the pth file exists (decide if you want to resume training)
            print(str(resume_model_file) + " is the newest .pt file, resuming training...")

            model = YOLO(resume_model_file)

            model.train(cfg=training_yaml,
                resume=True,
                data="{}/data.yaml".format(use_dataset),
                epochs=set_epochs,  # x epochs for training
                imgsz=set_image_size  # 640 = 640x640 image sizes for training
                        )

            model.save(model_file)


    # Training retrain
    elif train_choice == "3":
        try:
            print(str(model_file) + " selected, retraining...")
            model = YOLO(model_file)

            model.train(cfg=training_yaml,
                data="{}/data.yaml".format(use_dataset),
                epochs=set_epochs,  # x epochs for training
                imgsz=set_image_size  # 640 = 640x640 image sizes for training
                    )
            model.save(model_file)

        except FileNotFoundError:
            print("ERROR: Model file {} not found. Seems like it wasn't trained yet. Try again or change the use_model variable, currently set as {}".format(model_file, use_model))
            exit(1)

    else:
        exit("Invalid choice.")


def main_run():
    model_train()


if __name__ == "__main__":
    main_run()

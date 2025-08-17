# KI_Training_Pipeline

## Quick Start

0. Run `package_installer.py`. It installs all necessary packages.
1. Run `training.py`. It starts a [browser app](http://localhost:8501) for training  configuration.
2. Move the created config.yaml to folder: src>>conf
3. Depending on the task, run `training_[task].py`
4. Open [Tensorboard](http://127.0.0.1:16006/) for output.

## How to use

## Pipeline Construction

| Module | Description | Available standard options |
| --- | --- | --- |
| augmentations | Provides premade templates for augmentation. They are applied pre training, but do not create new datasets. | no_augments, augment_rotate_flip_shear_translate_brightness, augment_1 |
| conf | Contains the Config.yaml for training pipeline executions. | config.yaml |
| datasets | Contains Datasets | Cifar10, Pokemon, ImgNet, Yolo-Duckiebots-Lanes, Coco-Duckiebots-Lanes |
| model_architecture | Contains models.  | swin_transformer, cnn, ResNet50, vision_transformer |
| outputs | Temporary hydra files  |  |
| trained_models | Keeps the trained models and all according files. | Tensorboard, Model with weights, config file, summary, log file |
| training.py | Starts an app to configure the training. Sets the layout standards. |  |
| training_class.py | Executes the training | |

## training overview

| Module | Description | Available standard options |
| --- | --- | --- |
| 1. EXPERIMENT SETUP | Directorys and date-time setup | |
| 2. LOGGER SETUP | Creates and saves log for experiment in folders. | |
| 3. MODEL LOADING | Reads the selected model from the config.yaml, selects it from model_architecture folder and builds it. | |
| 4. AUGMENTATION | Adds the specified augmentation from augments folder. | no_augments, augment_rotate_flip_shear_translate_brightness, augment_1 |
| 5. DATASET TYPE LOADING | Selects the Type of the dataset. | Coco, Yolo |
| 6. MODEL TO DEVICE | Starts Cuda GPU if available. | GPU, CPU |
| 7. OPTIMIZER SETUP | Loads optimiser, learningrate and weight decay configuration. | Adam, AdamW, SGD, RMSprop, Adagrad, Adadelta |
| 8. SCHEDULER SETUP |  |  |
| 9. TENSORBOARD SETUP | Starts Tensorboard logging | training loss, validation loss, Precision, Recall, f1 |
| 10. TRAINING LOOP | Does the training and validation with metrics. Saves newest model |  |
| 11. TEST EVALUATION | Does the final Validation |  |
| 12. EXPERIMENT SUMMARY | creates a clean .yaml file from the training process. |  |
| 13. FINAL LOGGING | Summary of the training with the best scores and model. Displays options to procede from here. | result directory, TensorBoard, summary.yaml, training log, best model directory |

## Continuations

- Verwendung und Beispiele: Verwenden Sie diesen Abschnitt, um gegebenenfalls Beschreibungen und Anwendungsbeispiele für Ihr Projekt bereitzustellen.

- Abhängigkeiten: Listen Sie alle externen Bibliotheken oder Pakete auf, die zum Ausführen Ihres Projekts erforderlich sind. Dies hilft Benutzern zu verstehen, womit sie vertraut sein sollten.

- Dokumentation und Links: Stellen Sie Links zu zusätzlicher Dokumentation, der Projektwebsite oder verwandten Ressourcen bereit.

- Änderungsprotokoll: Fügen Sie einen Abschnitt hinzu, der die Änderungen, Aktualisierungen und Verbesserungen auflistet, die in jeder Version Ihres Projekts vorgenommen wurden.

- Bekannte Probleme: Listen Sie alle bekannten Probleme oder Einschränkungen mit der aktuellen Version Ihres Projekts auf. Dies kann eine Gelegenheit für Beiträge bieten, die sich mit dem Thema befassen.

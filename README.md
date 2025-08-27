# AI Training Pipeline

## Contents
- [Quick Start](#quick-start)


## Quick Start
To use tha AI Pipeline you need to have the dependencies installed

[Install dependencies](#installing-dependencies)

To start training on already implemented models:

1. [Add Your Own Dataset](#add-your-own-dataset) (Optional)
2. [Start training](#start-training)


## Installing Dependencies
All dependencies can be installed or looked up using the `package_installer.py` script located in the src folder:
```
# Navigate to the src folder
cd src/

# Run the installer script
python3 package_installer.py
```
This script will check and install all required dependencies for the AI Pipeline

You can install the necessary packages by launching the file. If you use conda just launch the file in the new env.

Otherwise if you like to use conda directly you can build the env by creating a new environment via the `environment.yml` file:
```
# Creating conda environment via the environment.yml file
conda env create -f environment.yml
```

If your on Linux it is recommended to aditionally install one of the following terminal emulators to have all functionality of the AI Pipeline
- xterm
- x-terminal-emulator
- gnome-terminal


## Start Training
To start the AI Pipeline you need to have the [dependencies](#installing-dependencies) installed.

The pipeline workflow starts with launching the `training.py` file in the src folder
```
# Navigate to the src folder
cd src/

# Run the installer script
python3 training.py
```
This will launch the browser application and allow you to configure the training settings.

Excerpt of the configuration application
![image](src/docs/images/excerpt_of_configuration_application.png)


### Start Training Workflow
```mermaid
flowchart TD
    Start([Start Application]) --> |User interface launches automatically| SelectTrainingSettings[Select your Training Settings]
    SelectTrainingSettings --> YAMLPreview[Preview YAML]
    YAMLPreview --> DownloadConfig[Download config.yaml]
    YAMLPreview --> StartTraining[Start Training]
    StartTraining --> TrainingExecution[Execute Training Process]
    TrainingExecution --> End([End])
```

At the moment only Linux with installed terminal emulators allows to start the training from the brower right away.
If your device meets that requirement you can start the training via the `Start Training` button.

If you have no terminal emulator installed or using Windows you have to start the training process manually after saving the `config.yaml` file in the `src/conf/` folder.

The manual start of the training is done by launching the file of the determied training task.

```mermaid
flowchart TD
SelectTrainingFile[Select Appropriate Training File]
    SelectTrainingFile --> |classification| ClassTraining[training_class.py]
    SelectTrainingFile --> |object detection| ObjDetTraining[training_objdet.py]
    SelectTrainingFile --> |segmentation| SegTraining[training_seg.py]
    
    ClassTraining --> ExecuteTraining[Execute training process]
    ObjDetTraining --> ExecuteTraining
    SegTraining --> ExecuteTraining
```


## View Results
After the training is completed you get an output like this:

![image](src/docs/images/completed_training_console.png)

You get a summary of where the data is stored and get the command to view the results in tensorboard
```
tensorboard --logdir=trained_models/{task}/{experiment_name}/tensorboard
```


## Add Your Own Dataset



## Implement your own Architectures
To start implementing your own models make yourself familiar with the architecture requirements

[Link zu den Anforderungen an die Architekturen]()

___
## Internal Pipeline Structure

1. Setup Phase: Creates [experiment directories](#experiment-directories), configures [logging](#logging), and loads the specified [model architecture](#model-architecture).
2. Data Preparation: Sets up data transforms/[augmentation](#augmentation) and loads datasets with appropriate preprocessing.
3. Training Environment: Moves the model to the appropriate device (GPU/CPU), configures the [optimizer](#optimizer) and learning rate [scheduler](#scheduler).
4. Training Loop: For each epoch:
    - Processes batches of training data
    - Logs model parameters and visualizations
    - Validates on the validation set
    - Evaluates model performance using [COCO metrics](#coco-metrics)
    - Updates learning rate (if scheduler is used)
    - Saves checkpoints and tracks best model
5. [Early Stopping](#early-stopping): Monitors validation performance and stops training if no improvement is seen after a set number of epochs.
6. [Evaluation](#evaluation): After training completes, the best model is evaluated on the test set to assess generalization performance.
7. [Results](#results): Creates comprehensive experiment summary including metrics, confusion matrices, and TensorBoard visualizations.


```mermaid
flowchart TD
    Start([Start]) --> ExperimentSetup[Setup Experiment Directory]
    ExperimentSetup --> LoggerSetup[Setup Logger]
    LoggerSetup --> ModelLoading[Load Model Architecture]
    ModelLoading --> DataAugmentation[Setup Data Transforms/Augmentation]
    DataAugmentation --> DatasetLoading[Load Datasets]
    DatasetLoading --> DataLoaderCreation[Create DataLoaders]
    DataLoaderCreation --> ModelToDevice[Model to Device]
    ModelToDevice --> OptimizerSetup[Setup Optimizer]
    OptimizerSetup --> SchedulerSetup[Setup LR Scheduler]
    SchedulerSetup --> TensorBoardSetup[Setup TensorBoard]
    
    TensorBoardSetup --> TrainingLoop[Start Training Loop]
    
    TrainingLoop --> |For each epoch| ClearGPUCache[Clear GPU Cache]
    ClearGPUCache --> TrainOneEpoch[Train One Epoch]
    TrainOneEpoch --> LogParameters[Log Model Parameters]
    LogParameters --> Visualizations[Log Visualizations]
    Visualizations --> ValidationPhase[Validate Model]
    ValidationPhase --> EvaluateMetrics[Evaluate COCO Metrics]
    
    EvaluateMetrics --> SchedulerStep{Use Scheduler?}
    SchedulerStep --> |Yes| PerformSchedulerStep[Perform Scheduler Step]
    SchedulerStep --> |No| LogEpochMetrics[Log Epoch Metrics]
    PerformSchedulerStep --> LogEpochMetrics
    
    LogEpochMetrics --> SaveCheckpoint[Save Checkpoint]
    SaveCheckpoint --> CheckBestModel{Is Best Model?}
    
    CheckBestModel --> |Yes| SaveBestModel[Save Best Model]
    CheckBestModel --> |No| IncrementPatience[Increment Patience Counter]
    SaveBestModel --> ResetPatience[Reset Patience Counter]
    
    ResetPatience --> CheckEpochEnd{More Epochs?}
    IncrementPatience --> CheckEarlyStop{Early Stopping?}
    
    CheckEarlyStop --> |Yes| TestEvaluation[Evaluation on Test Set]
    CheckEarlyStop --> |No| CheckEpochEnd
    
    CheckEpochEnd --> |Yes| TrainingLoop
    CheckEpochEnd --> |No| TestEvaluation
    
    TestEvaluation --> LoadBestModel[Load Best Model]
    LoadBestModel --> ComputeTestLoss[Compute Test Loss]
    ComputeTestLoss --> BuildConfusionMatrix[Build Confusion Matrix]
    BuildConfusionMatrix --> CreateSummary[Create Experiment Summary]
    CreateSummary --> CloseWriter[Close TensorBoard Writer]
    CloseWriter --> FinalLogging[Final Logging]
    FinalLogging --> End([End])
```

### Pipeline Directories
#### Dataset Directories
#### Experiment Directories


### Model Architecture
### Augmentation
### Optimizer
### Scheduler
### COCO Metrics
### Early Stopping
### Evaluation
### Results
### Logging



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


## Continuations

- Usage and Examples: Use this section to provide descriptions and usage examples for your project.

- Dependencies: List all external libraries or packages needed to run your project. This helps users understand what they should be familiar with.

- Documentation and Links: Provide links to additional documentation, the project website, or related resources.

- Changelog: Add a section listing the changes, updates, and improvements made in each version of your project.

- Known Issues: List any known issues or limitations with the current version of your project. This can provide an opportunity for contributions that address the issue.
# Prescriptive-Process-Analytics-using-Counterfactuals

### Install Requirements
Tested with Python 3.9. 

#### Install packages
```bash
pip install -r requirements.txt
```

## Notebooks
Run the `jupyter` server in the directory: `explainable-prescriptive-analytics` the project directory, it shouldn't be 
running in the `src` directory.

### 00_preprocess_event_logs.ipynb
This notebook preprocess the event-log data, so that machine learning can be applied to it. The notebook currently 
preprocesses the data for 2 KPIs "Activity Occurrence" and "Total Time".

### 02_DiCE_kpi-time.ipynb
This notebook contains code that uses VINST dataset and tries to optimize the `lead_time` aka the total
time it takes to complete a trace. DiCE algorithm is used for generating counterfactual explanations (CFEs)
and then validation of the CFEs is done as postprocessing steps.

### 05_load_data_for_experiments
It also contains code to find the best parameter combination from parameter tuning logs of the ML models. 

### 07_evaluate_recommendations
This notebook uses a different ML model (possibly trained on all the data) to evaluate the recommendations 
(activity and resource) made by the DiCE algorithm and the RAR (Alessandro's algorithm).


## CounterFactual Examples (CFEs) Scripts
### To run scripts on the server.
1. copy script E.g. `script_06...` to the Dice method name followed by some identifier (for tracking)
   e.g. name: `gradient-a01.py`
2. In the script of your choice e.g. `script_02_DiCE_kpi-time.py` change the variables:

   - `DICE_METHOD`: don't change this, this comes from file name
   - `RESULTS_FILE_PATH_N_NAME`: this also comes from file name
   - Change any other parameter you want.
   
3. Run the command: `run_script.sh gradient-a01 --first_run`.

## ML model Tuning Scripts
### To run `script_07`
if using Tensorboard setup:
```bash
cp src/script_07_pytorch_tuning.py src/script_07_exp_2.py
./run_script.sh script_07_exp_2
```
else if using wandb setup
 ```bash
source py-env/bin/activate
nohup python -m src.script_07_pytorch_tuning &
 ```

### To run `script_04`
```bash
nohup python -m src.script_04_parameter_tuning &> logs/script_04_balanced_rf.log &
```



### Gradient Based Method on Dice requires Python 3.9


## Test Code
#### src/transition_system.py
To test the transition system use the following command. Make sure to install pytest library. 

    pytest src/transition_system.py


---
## Info About Datasets
### VINST Dataset
Columns
- START_DATE - Start time of the activity. Format is Timestamp in milliseconds.
- END_DATE - End time of the activity. Format is Timestamp in milliseconds.

### Results files
File: `resutls_vf.ods` - contains the results of the experiments.
Column explanation is as follows:
- exp_name: Experiment name
- run time (days): New predicted time for completing all the cases given the best recommendations in the CFEs produced.

import dice_ml
from dice_ml import Dice
from dice_ml.utils.exception import UserConfigValidationException

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, fetch_california_housing

from src.transition_system import transition_system, indexs_for_window, list_to_str

import pandas as pd
import pickle
import os
import random
import subprocess
import joblib
from math import ceil
from wrapt_timeout_decorator import timeout
import warnings

# from multiprocessing import Process
# from timeout_decorator import timeout, TimeoutError
# import signal
from typing import Tuple, Any, List, Union
from collections import Counter
# import utils
from time import sleep


class SamplingStrategy:
    Random = 'random'
    Genetic = 'genetic'
    KdTree = 'kdtree'
    Gradient = 'gradient'


class SetupVariables:
    def __init__(self, dataset = None ):
        if dataset is None:
            raise ValueError("Pass a dataset name to the dataset parameter")

        self.dataset = dataset
        self.data_dir = "./preprocessed_datasets/"
        self.results_dir = "./results"
        self.model_dir = "./ml_models/"

        if dataset == "VINST cases incidents.csv":
            self.kpi = "total_time"
            self.case_id_name = 'SR_Number'  # The case identifier column name.
            self.start_date_name = 'Change_Date+Time'  # Maybe change to start_et (start even time)
            self.activity_column_name = "ACTIVITY"
            self.resource_column_name = "Involved_ST"
            self.role_column_name = "Involved_Org_line_3"
            self.end_date_name = None
            self.train_dataset_file = "vinst_train.csv"
            self.test_dataset_file = "vinst_test.csv"
            self.test_pickle_dataset_file = "vinst_test.pkl"
            self.outcome_name = "lead_time"
            self.columns_to_remove=["Change_Date+Time", "time_remaining"]

            self.df = pd.read_csv(f"./data/{dataset}")  # Use full dataset for transition systens
            self.df_train = pd.read_csv(os.path.join(self.data_dir, self.train_dataset_file))
            self.df_test = pd.read_csv(os.path.join(self.data_dir, self.test_dataset_file))
            # Resource Aware Recommendations
            self.df_rar = pd.read_csv(os.path.join(self.results_dir, "rar_gkpi_results_VINST_time.csv"))

        elif dataset == "bank_account_closure.csv":
            self.kpi = "activity_occurrence"
            self.case_id_name = "REQUEST_ID"
            self.start_date_name = "START_DATE"
            self.activity_column_name = "ACTIVITY"
            self.resource_column_name = "CE_UO"
            self.role_column_name = "ROLE"
            self.end_date_name = "END_DATE"
            self.activity_to_avoid = "Back-Office Adjustment Requested"
            self.train_dataset_file = "bank_acc_train.csv"
            self.test_dataset_file = "bank_acc_test.csv"
            self.test_pickle_dataset_file = "bank_acc_test.pkl"
            self.outcome_name = "Back-Office Adjustment Requested"
            self.columns_to_remove = ["START_DATE", "END_DATE", "time_remaining"]

            self.df = pd.read_csv(f"./data/{dataset}")  # concern: what is date col position is different?
            self.df[self.start_date_name] = pd.to_numeric(self.df[self.start_date_name]) // 1_000  # Convert Epochs from milliseconds to seconds
            self.df[self.end_date_name] = pd.to_numeric(self.df[self.end_date_name]) // 1_000  # Convert Epochs from milliseconds to seconds
            self.df_train = pd.read_csv(os.path.join(self.data_dir, self.train_dataset_file))
            self.df_test = pd.read_csv(os.path.join(self.data_dir, self.test_dataset_file))
            # rar: Resource Aware Recommendations
            self.df_rar = pd.read_csv( os.path.join(self.results_dir, "rar_gkpi_results_BAC_act.csv") )
        self.df = self.df.fillna("missing")
        self.test_cases = get_test_cases(None, None, load_dataset=True,
                                         path_and_filename=os.path.join(self.data_dir, self.test_pickle_dataset_file))
        self.df_rar = merge_rar_with_test_cases(self.df_rar, self.case_id_name, self.test_cases)

class StoreTestRun:
    def __init__(self, save_load_path=None):
        """
        Args:
            save_load_path: E.g. "experiment_results/name_of_file.csv"
        """
        self.configs = None
        self.run_state = {
            "cfe_before_validation": [],
            "cfe_after_validation": [],
            "cfe_not_found": [],
            "cases_includes_new_data": [],
            "cases_too_small": [],
            "cases_zero_in_y": [],
            "exceptions": [],
            "trace_time": [],
            "cases_done": 0
        }
        save_load_dir = save_load_path.split(".")[0]
        self.save_load_state_path = save_load_dir + ".pkl"

    def add_model_configs(self, configs=None):
        """
        Args:
            configs (dict): E.g. {"window_size": 3,
                                 "reduced_kpi_time": 90,
                                 "total_cfs": 50,                       # Number of CFs DiCE algorithm should produce
                                 "dice_method": extract_algo_name(RESULTS_FILE_PATH_N_NAME),
                                 "output_file_path": RESULTS_FILE_PATH_N_NAME,
                                 "train_dataset_size": 31_066,                                   # 31_066
                                 "proximity_weight": 0.2,
                                 "sparsity_weight": 0.2,
                                 "diversity_weight": 5.0}
        """
        if configs is None:
            raise ValueError("Pass a Python dict to the configs parameter")

        self.configs = configs

    def get_model_configs(self):
        return self.configs

    def add_cfe_to_results(self, res_tuple: Tuple = None):
        # E.g. ("cfe_not_found", cfe) = res_tuple
        dict_key, result_value = res_tuple
        self.run_state[dict_key].append(result_value)

    def save_state(self):
        with open(self.save_load_state_path, 'wb') as file_handle:
            self.run_state["cases_done"] += 1  # Represents a single case completion
            dictionary_store = { "run_state": self.run_state,
                                 "configs": self.configs}
            print(f"====== Start Saving the result ======")
            pickle.dump(dictionary_store, file_handle)
        print(f"====== End Saving the result For case: {self.run_state['cases_done']} ======")
        return self.run_state["cases_done"]

    def load_state(self):
        with open(self.save_load_state_path, 'rb') as file_handle:
            dictionary_store = pickle.load(file_handle)
            self.run_state = dictionary_store["run_state"]
            self.configs = dictionary_store["configs"]

    def get_save_load_path(self):
        return self.save_load_state_path

    def get_run_state_df(self):
        data = {"cfe_before_validation": [len(self.run_state['cfe_before_validation'])],
                "cfe_after_validation": [len(self.run_state["cfe_after_validation"])],
                "cfe_not_found": [len(self.run_state["cfe_not_found"])],
                "cases_includes_new_data": [len(self.run_state["cases_includes_new_data"])],
                "cases_too_small": [len(self.run_state["cases_too_small"])],
                "cases_zero_in_y": [len(self.run_state["cases_zero_in_y"])],
                "exceptions": [len(self.run_state["exceptions"])],
                "cases_done": [self.run_state["cases_done"]]
                }
        df_result = pd.DataFrame(data)
        return df_result


def extract_algo_name(save_load_path=""):
    file_name = save_load_path.split("/")[1]
    algo_name = file_name.split("-")[0]
    return algo_name

class Test_extract_algo_name:
    def test_end_exclusive_true(self):
        save_load_path = "experiment_results/kdtree-05-total_time.csv"
        algo_name = extract_algo_name(save_load_path)
        assert algo_name == "kdtree"

    # To test this function run command from project directory:
    # `pytest src/function_store.py `


def get_case_id(df, case_id_name="SR_Number") -> Union[str, int]:  # , multi=False
    return df[case_id_name].unique().item()

def cases_with_activity_to_avoid(df, case_id_name, activity_column_name, activity_to_avoid):
    """
    Returns:
        case_ids_with_activity_to_avoid, case_ids_without_activity_to_avoid
    """
    # === How much traces have `activity_to_avoid` ===========================
    case_ids_with_activity_to_avoid = []
    case_ids_without_activity_to_avoid = []
    gdf = df.groupby(case_id_name)
    for case_id, group in gdf:
        if activity_to_avoid in group[activity_column_name].to_list():
            case_ids_with_activity_to_avoid.append(case_id)
        else:
            case_ids_without_activity_to_avoid.append(case_id)
    # print(f"Cases with activity_to_avoid: {len(case_ids_with_activity_to_avoid):,}")
    # print(f"Cases without activity_to_avoid: {len(case_ids_without_activity_to_avoid):,}")
    return case_ids_with_activity_to_avoid, case_ids_without_activity_to_avoid

def variable_type_analysis(X_train, case_id_name, activity_name):
    """ Can be added to class: EventLog (in file 00_preprocess notebook).
    Args:
        X_train:
        case_id_name:
        activity_name:

    Returns:
            Tuple[List[int], List[str], List[float]]: The explanation of the lists is as:
            1st list: quantitative_attributes. Names of columns with numeric values.
            2nd List: case_attributes. Names of columns whose
                      value remains same for a single trace. Basically 1 value per trace.
            3rd List: event_attributes. Names of columns with string type.
    """
    quantitative_attributes = list()
    case_attributes = list()
    event_attributes = list()

    for col in X_train.columns:  # for col in tqdm.tqdm(X_train.columns):

        if (col not in [case_id_name, activity_name]) and (col[0] != '#'):
            if type(X_train[col][0]) != str:
                quantitative_attributes.append(col)
            else:
                trace = True
                for idx in X_train[case_id_name].unique():  # 150 has been set to be big enough
                    df = X_train[X_train[case_id_name] == idx]
                    if len(set(df[col].unique())) != 1:
                        trace = False
                if trace:
                    case_attributes.append(col)
                else:
                    event_attributes.append(col)

    return quantitative_attributes, case_attributes, event_attributes

def prepare_df_for_ml(df, case_id_name, outcome_name, columns_to_remove=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    :param str outcome_name: name of the target column.
    """
    # Before training for ml we need to remove columns that can are not needed for ML model.
    if columns_to_remove is None:
        columns_to_remove = ["Change_Date+Time", "time_remaining"]
    df = df.drop([case_id_name], axis="columns")
    df = df.drop(columns_to_remove, axis="columns")
    X = df.drop([outcome_name], axis=1)
    y = df[outcome_name]
    return X, y

def get_test_cases(df, case_id_name, load_dataset=False, path_and_filename=None):
    """
    Converts flat test dataframe to list of dataframes. Each dataframe is of a single case.
    Args:
        df:
        case_id_name:
        load_dataset (bool): Ignores arguments df and case_id_name, and loads the dataset from memory.
        path_and_filename (str): Includes Path to the pickled file and the filename.
    Returns:

    """
    if not load_dataset:
        result_lst = []
        for idx in df[case_id_name].unique():
            df_trace = df[df[case_id_name] == idx]
            # ceil enables cases with 1 row to pass through
            cut = ceil(len(df_trace) * random.uniform(0.3, 0.8) + 1)  # +1 cuz I want atleast 2 activities in the trace
            df_trace = df_trace.iloc[:cut].reset_index(drop=True)

            # df_result = pd.concat([df_result, df_trace])
            result_lst.append(df_trace.reset_index(drop=True))
            # break
        # return df_result.reset_index(drop=True)
    else:
        if path_and_filename is None:
            raise ValueError("Specify the path and filename of the pickled test dataset")
        # Unpickle the Standard test-set. To standardize the test across different parameters.
        with open( path_and_filename, 'rb') as file:
            result_lst = pickle.load(file)

    return result_lst


def get_cut_test_traces(test_cases):
    """Generate a CSV of cut Test Traces. Returns the last row from each test trace in `test_cases` """
    df_cut_test_traces = pd.DataFrame(columns=test_cases[0].columns)
    for cut_trace in test_cases:
        # display(cut_trace.iloc[-1:])
        # Create a dataframe of the above and do MAE on this
        df_cut_test_traces = pd.concat([df_cut_test_traces, cut_trace.iloc[-1:]], axis="rows")
    return df_cut_test_traces


def merge_rar_with_test_cases(df_rar, case_id_name, test_cases):
    df_test_cases_ids = pd.DataFrame( {"Case_id": [ get_case_id( trace, case_id_name ) for trace in test_cases ] })
    # Merging the rest of the test_case_ids also sorted the df according to the order of test_cases
    df_rar = pd.merge( df_rar, df_test_cases_ids, how="right", on="Case_id", sort=False)
    assert len(df_rar) == len(test_cases)  # Else the length of merged df_rar is not equal to the test_cases'
    return df_rar.rename(columns={"Case_id": case_id_name})


def activity_n_resources(df, resources_columns=None, threshold_percentage=100):
    """
    Creates a set of tuples, each tuple has elements from the columns specified through `resource_columns`.
    E.g. { ('action_1', 'rresource_1'), ... (activity_3, resource_5) }
    Args:
        df (pd.DataFrame):
        resources_columns (list): columns that contains the activity and resources.
        threshold_percentage (int): The code sorts the activity and resources combination according to their frequencies
                                It puts them in a list, `threshold_percentage` tells what fraction of that list to keep.
    Returns:
        Set of tuples. A single element contains the activity and resources of a single row from the
        dataframe.
    """
    if resources_columns is None:
        raise TypeError("Please specify the columns that have resources")

    threshold = threshold_percentage / 100

    valid_activity_n_resource = set( df[resources_columns].apply(tuple, axis='columns') )

    # combo: combination
    resource_combo_frequency = {}

    valid_resource_combo = df[resources_columns].apply(tuple, axis='columns')

    for elem in valid_resource_combo:
        if resource_combo_frequency.get(elem):
            resource_combo_frequency[elem] += 1
        else:
            resource_combo_frequency[elem] = 1
    # Creates a list of (combo, counts)
    resource_combo_counts = [ (k, v) for k, v in resource_combo_frequency.items() ]
    sorted_resource_combo_counts = sorted( resource_combo_counts, key=lambda item: item[1], reverse=True )
    sorted_combos = [combo for combo, _ in sorted_resource_combo_counts ]
    amount_of_combos_to_select = int( len(sorted_combos) * threshold ) + 1
    valid_activity_n_resources = sorted_combos[:amount_of_combos_to_select]
    return valid_activity_n_resources


def get_prefix_of_activities(expected_activity_index=None, event_log=None, df_single_trace=None, window_size=3,
                             activity_column_name=None, case_id_name=None):
    """ Retrieves the prefix of activities from the event log. So that later the next activity can be validated using the prefix.
    This function can be used for 2 different cases. 1st, passing it df_single_trace and prefix is extracted from it.
    2nd, passing different arguments allowing is to single out trace prefix from the entire event log (df_train).
    Args:
        expected_activity_index (int)
        event_log (pd.DataFrame): Dataframe containing many traces. E.g. df_train
        df_single_trace (pd.DataFrame): A dataframe that contains a single trace. E.g. a running trace or a test trace. It is expected
            that the index of this dataframe starts from 0. An assumption is that last activity/ row represents the expected activity,
            so the prefix of activities ends at the 2nd last activity. when using this parameter, query_case_id and related parameters
            are ignored.

    """
    # Error checking
    if activity_column_name is None:
        raise "Please specify activity_column_name"

    if df_single_trace is not None:  # Case 1
        # Check if indexes start from 0
        assert df_single_trace.loc[0] is not None

        # Due to assumption that last activity is the expected activity so the prefix ends at the 2nd last activity
        # If this raises an exception it means no activity has occurred.
        index_to_previous_activity = df_single_trace.index[-2]

        start_index, end_index = indexs_for_window(index_to_previous_activity, window_size=window_size, end_exclusive=False)
        prefix_of_activities = df_single_trace.loc[start_index: end_index, activity_column_name].to_list()  # loc is used to access the index values inside the dataframe
        prefix_of_activities = list_to_str(prefix_of_activities)

        return prefix_of_activities
    else:  # Case 2
        # if query_case_id is None:
        #     raise "Please specify query_case_id!"
        if event_log is None:
            raise "Please specify event_log!"
        if expected_activity_index is None:
            raise "Please specify expected_activity_index!"

        query_case_id = get_case_id( event_log[expected_activity_index: expected_activity_index+1] )

        # Isolate the query_case_id trace
        df_query = event_log[ event_log[case_id_name] == query_case_id ]

        # Prefix ends before the expected activity timestamp
        index_to_previous_activity = expected_activity_index - 1

        start_index, end_index = indexs_for_window(index_to_previous_activity, window_size=window_size, end_exclusive=False)
        prefix_of_activities = df_query.loc[start_index: end_index, activity_column_name].to_list()  # loc is used to access the index values inside the dataframe
        prefix_of_activities = list_to_str(prefix_of_activities)

        return prefix_of_activities

def validate_transition(cfe, prefix_of_activities=None, transition_graph=None, valid_resources=None,
                        activity_column_name=None, resource_columns_to_validate=None):
    """  resource_columns_to_validate=None possible future parameter
    Args:
        cfe (dice_ml.counterfactual_explanations.CounterfactualExplanations): Dice counterfactual explanations object.
        window_size (int): Size of the prefix of trace for which next activity is checked. See `index_for_window` function
                            documentation.
        expected_activity_index (int):
    Returns:
        pd.DataFrame
    """
    if cfe is None:
        raise "Please specify cfe!"
    if valid_resources is None:
        raise "Please specify valid_resources!"
    if transition_graph is None:
        raise "Please specify transition_graph"
    if prefix_of_activities is None:
        raise "Please specify prefix_of_activities"

    cf_examples_df = cfe.cf_examples_list[0].final_cfs_df.copy()  # Get the counterfactual explanations dataframe from the object

    # === Verify the next activity
    indexes_to_drop = []
    for i, suggested_next_activity in cf_examples_df[activity_column_name].items():
        if suggested_next_activity not in transition_graph[prefix_of_activities]:
            indexes_to_drop.append(i)
            # print(i, suggested_next_activity)

    cf_examples_df = cf_examples_df.drop(indexes_to_drop, axis='index').reset_index(drop=True)

    # === Verify the associated resources
    indexes_to_drop = []
    for i, row in cf_examples_df[ resource_columns_to_validate ].iterrows():
        row_tuple = tuple(row)
        if row_tuple not in valid_resources:
            # print(f"removed row had: {row_tuple}")
            indexes_to_drop.append(i)

    cf_examples_df = cf_examples_df.drop(indexes_to_drop, axis='index').reset_index(drop=True)
    return cf_examples_df

@timeout(1_200)
def generate_cfe_torch(explainer, query_instances, total_time_upper_bound=None, features_to_vary=None, total_cfs=50, kpi="",
                 proximity_weight=0.0, sparsity_weight=0.0, diversity_weight=0.0, permitted_range=None):
    """ See function generate_cfe for more details."""
    # We know kpi is activity_occurrence. This if is a hack agreed.
    # isinstance(explainer.model, dice_ml.model_interfaces.pytorch_model.PyTorchModel)  # To check ML model type.
    cfe = explainer.generate_counterfactuals(query_instances, total_CFs=total_cfs, desired_class=0,
                                             features_to_vary=features_to_vary,
                                             permitted_range=permitted_range)  # 'Back-Office Adjustment Requested'
    return cfe


@timeout(300)  # Timeout unit seconds
def generate_cfe(explainer, query_instances, total_time_upper_bound=None, features_to_vary=None, total_cfs=50, kpi="",
                 proximity_weight=0.0, sparsity_weight=0.0, diversity_weight=0.0, permitted_range=None):
    """ For ref: http://interpret.ml/DiCE/dice_ml.explainer_interfaces.html#dice_ml.explainer_interfaces.explainer_base.ExplainerBase.generate_counterfactuals
    Args:
        explainer (dice_ml.Dice):
        query_instances (pd.DataFrame):
        total_time_upper_bound (int, None): The upper value of the target (y) label.
        total_cfs (int): Number of Counterfactual examples (CFEs) to produce via `generate_counterfactuals()`
        proximity_weight (float): A positive float. Larger this weight, more close the counterfactuals are to the
                query_instance. Used by [‘genetic’, ‘gradientdescent’], ignored by [‘random’, ‘kdtree’] methods.
        sparsity_weight (float): A positive float. Larger this weight, fewer features are changed from the
                query_instance. Used by [‘genetic’, ‘kdtree’], ignored by [‘random’, ‘gradientdescent’] methods.
        diversity_weight (float): A positive float. Larger this weight, more diverse the counterfactuals are. Used by
                [‘genetic’, ‘gradientdescent’], ignored by [‘random’, ‘kdtree’] methods.
    Returns:
        cfe (dice_ml.counterfactual_explanations.CounterfactualExplanations): Dice counterfactual explanations object.
    """
    if kpi == "activity_occurrence":
        # Usually .generate_counterfactuals use desired_class="opposite" but we use 0 because we need to always want the
        # target attribute (label column) to be 0, meaning the bad activity will not occur.
        cfe = explainer.generate_counterfactuals(query_instances, total_CFs=total_cfs, desired_class=0,
                                                 features_to_vary=features_to_vary,
                                                 permitted_range = {"ACTIVITY": [
                                                     'Service closure Request with network responsibility',
                                                     'Service closure Request with BO responsibility',
                                                     'Pending Request for Reservation Closure',
                                                     'Pending Liquidation Request',
                                                     # 'Request completed with account closure',
                                                     'Request created',
                                                     'Authorization Requested',
                                                     'Evaluating Request (NO registered letter)',
                                                     'Network Adjustment Requested',
                                                     'Pending Request for acquittance of heirs',
                                                     # 'Request deleted',
                                                     'Evaluating Request (WITH registered letter)',
                                                     # 'Request completed with customer recovery',
                                                     'Pending Request for Network Information']})  # 'Back-Office Adjustment Requested'
    else:
        cfe = explainer.generate_counterfactuals(query_instances, total_CFs=total_cfs,
                                                 desired_range=[0, total_time_upper_bound],
                                                 features_to_vary=features_to_vary,
                                                 proximity_weight=proximity_weight, sparsity_weight=sparsity_weight,
                                                 diversity_weight=diversity_weight)
    return cfe


def download_remote_models(model_file_name = None, return_model=True):
    """Downloads ML models from the remote server
    No need to specify the ml_models directory
    Returns:
        model (sklearn.model_selection._search.GridSearchCV): The downloaded model if found
        else returns 0
    """
    if model_file_name is None:
        raise "Please specify the model file name!"
    if '.' not in model_file_name:
        raise ValueError("Please specify the file extension as well!")

    if not os.path.exists(f"./ml_models/{model_file_name}"):
        # Get the file from the remote server
        result = subprocess.run(['scp',
                                 f'labnum01:git_repos/explainable-prescriptive-analytics/ml_models/{model_file_name}',
                                 f'./ml_models/{model_file_name}'], capture_output=True, text=True)

        print("Return code", result.returncode)
        if result.returncode != 0:
            raise Exception(f"scp Error! Model name: {model_file_name}")

    if return_model and ".joblib" in model_file_name:
        # Return the downloaded file
        return joblib.load(f'./ml_models/{model_file_name}')
    else:
        return 0


def load_experiment(pickle_file: str = None):
    """
    Loads the experiment information from .pkl file using experiment name. If not present locally it downloads
    from the remote server
    """
    if ".pkl" not in pickle_file:
        pickle_file += ".pkl"
    if not os.path.exists( f"./experiment_results/{pickle_file}"):
        result = subprocess.run(['scp', f'labnum08:git_repos/explainable-prescriptive-analytics/experiment_results/{pickle_file}', 'experiment_results/'], capture_output=True, text=True)
        # return code of 0 means the command executed successfully
        if result.returncode != 0:
            print("There is an Error in the command")
        else:
            print("successful")
    else:
        print(f"File already exists")
    results_file_path_n_name = f"experiment_results/{pickle_file.split('.')[0]}.csv"
    # The file name passed to the function is .csv, but it doesn't read that, it reads the .pkl file'
    state_obj = StoreTestRun(save_load_path=results_file_path_n_name)
    state_obj.load_state()
    return state_obj


def get_cfe_dfs_n_count(case_id_name: str, test_cases: List[List[pd.DataFrame]], state_obj: StoreTestRun):
    # For all test_cases get CFEs (Counterfactual Examples) DataFrames from state_obj
    """
    Returns:
        List[Tuple[str, Union[str, pd.DataFrame]], int]: This tuple's 1st element is case_id and the
            2nd element is either "cfe_not_found" or DataFrame containing found CFEs. Second element
            is the number of CFEs found.
    """
    test_cases_n_cfes = []
    cfe_count = 0
    for test_trace in test_cases:

        test_trace_case_id = get_case_id(test_trace, case_id_name)
        # If a cf was found for this case id the assignment below will replace "cfe_not_found" with cfe_df
        info_tuple = [test_trace_case_id, "cfe_not_found"]
        # print(f"case id: {case_id_from_bucket}")
        for case_id, value in state_obj.run_state["cfe_after_validation"]:
            if case_id == test_trace_case_id:
                # check if CFE exists
                if isinstance(value, pd.DataFrame):
                    # change the value from "cfe_not_found" to the dataframe containing CFEs
                    info_tuple[1] = value
                    cfe_count += 1

        test_cases_n_cfes.append(info_tuple)
    return test_cases_n_cfes, cfe_count


if __name__ == '__main__':
    # Test save and load functionality
    RESULTS_FILE_PATH_N_NAME = "experiment_results/kdtree-10-total_time.csv"
    configs = {"window_size": 3,
              "reduced_kpi_time": 90,
              "total_cfs": 50,                                  # Number of CFs DiCE algorithm should produce
              "dice_method": extract_algo_name(RESULTS_FILE_PATH_N_NAME),  # genetic, kdtree, random
              "save_load_path": RESULTS_FILE_PATH_N_NAME,
              "train_dataset_size": 31_066,                                   # 31_066
              "proximity_weight": 0.2,
              "sparsity_weight": 0.2,
              "diversity_weight": 5.0}

    state_obj = StoreTestRun(save_load_path=RESULTS_FILE_PATH_N_NAME)
    save_load_path = state_obj.get_save_load_path()

    # # Test Saving ( First run as is then comment this section and see if the save data is printed )
    # state_obj.add_model_configs(configs=configs)
    # state_obj.run_state["cfe_before_validation"].append(30)
    #
    # state_obj.save_state()

    # Test Loading
    if os.path.exists(save_load_path):
        state_obj.load_state()
        cases_done = state_obj.run_state["cases_done"]
        configs = state_obj.get_model_configs()

        print(f"configs: {configs}")
        print(f"run_state: {state_obj.run_state}")



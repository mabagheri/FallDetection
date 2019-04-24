__copyright__ = "Aerial 2019"
__version__ = "1.0.0"
__license__ = "MIT"
__author__ = "MohammadAli Bagheri"

import json
import numpy as np
from core.utils import *
from core.initializations.load_filepaths_metadata import load_selected_csi_paths_metadata
from core.preprocessing import preprocess_data_v1, csi_filter_v2, samples_generation
from core.preprocessing.subcarriers_streams_reduction import main_sc_streams_reduction
from core.feature_extractions import feature_extraction, feature_reduction
from core.classification import classification
# from core.preprocessing.csi_filter import get_RssiDropFilter_newModeData
import multiprocessing as mp
import time
import random
import warnings
warnings.filterwarnings("ignore")
np.random.seed(110)
random.seed(110)


configs = json.load(open('configurations/configs_v5_LSTM.json', 'r'))


def main(configs):

    start_time = time.time()

    selected_csi_paths, selected_metadata, selected_meta_data_per_key = load_selected_csi_paths_metadata(configs)

    # --- multiprocessing begins here ---
    pool = mp.Pool(processes=16)

    results = [pool.apply_async(load_preprocess_extract_features,
                                args=(selected_csi_paths[i], selected_metadata[i], configs))
               for i in range(len(selected_csi_paths))]
    # --- ends here ---

    # -------------------------------- Get results of multi-processing step together ----------------------------------
    # Ignores short length captures. Setting the 'min_acceptable_length' to 0 will not ignore any capture (use all CSIs)
    min_acceptable_length = min(configs['preprocessing']['min_acceptable_length'], configs["preprocessing"]["chopping_length"])
    print("min_acceptable_length = ", min_acceptable_length)
    results = [p for p in results if p.get()[2] >= min_acceptable_length]

    X = [p.get()[0] for p in results]
    samples_meta_data_list = [p.get()[1] for p in results]

    samples_meta_data_per_key = {k: [] for k in selected_metadata[0].keys()}
    for m in samples_meta_data_per_key.keys():
        samples_meta_data_per_key[m] = np.array([s[m] for s in samples_meta_data_list]).flatten()

    pool.close()
    pool.join()


    #
    # ----------------------------------------------  Classification Begins ------------------------------------------------
    print(" ......... Classification Begins .........")
    X = np.array([k for l in X for k in l])
    labels = np.array(samples_meta_data_per_key['label'])
    print("Number of samples = ", X.shape[0])
    print("Number of Features = ", X.shape[1])

    X = feature_reduction.return_transformed_features(X, configs)

    acc, sum_cms, f1score, BM_score, sensitivity, model_std, acc_train = \
        classification.classify(X, labels, samples_meta_data_per_key, configs)

    # ----------------------------------------------  Classification Ends ------------------------------------------------

    print("Accuracy = ", acc)
    print("sensitivity = ", sensitivity, "\n sum_cms =  ")
    print(sum_cms)
    print("Total elapsed time (seconds) : ", time.time() - start_time)

    return acc, sum_cms, f1score, BM_score, sensitivity, acc_train


def load_preprocess_extract_features(csi_path, this_csi_meta_data, configs):

    samples_meta_data = {k: [] for k in this_csi_meta_data.keys()}

    # ---------------------------------------------- Load Each CSI File ------------------------------------------
    H, Rssi, Ntx, packet_ids = load_raw_data(csi_path, configs['preprocessing']["chopping_length"])
    n_packets = H.shape[1]
    print("CSI loaded; n_packets = ", n_packets)

    # ------ pre-processing ------
    preprocessed_data = preprocess_data_v1.main(H, Rssi, Ntx, packet_ids, configs)

    # ------ Generating more than one sample from one preprocessed CSI (usually by sliding)  ------
    samples, n_samples = samples_generation.main_samples_generation(preprocessed_data, configs)

    # ----------------------------------------------  Feature Extraction ------------------------------------------
    features = feature_extraction.main_feature_extraction(samples, configs)

    #
    for m in this_csi_meta_data.keys():
        samples_meta_data[m].extend(np.tile(this_csi_meta_data[m], n_samples).tolist())

    return features, samples_meta_data, n_packets


if __name__ == '__main__':
    main(configs)



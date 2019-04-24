from core.feature_extractions.StatisticalFeatures import StatisticalFeatures
from core.feature_extractions.StatisticalFeatures_v3 import StatisticalFeaturesV3
from core.feature_extractions.FrequencyFeatures import *
from core.feature_extractions.FrequencyFeatures import *


def main_feature_extraction(all_samples, configs):
    methods = configs['feature_extraction']['method']
    all_features = []

    #
    for i, data in enumerate(all_samples):

        for method in methods:
            if method == "None":
                features = data

            else:
                f = eval(method)(data, configs)
                features = f.extract_features()

        #
        all_features.append(features)

    all_features = np.array(all_features)
    return all_features

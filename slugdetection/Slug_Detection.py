# -*- coding: utf-8 -*-
"""
Part of slugdetection package

@author: Deirdree A Polak
github: dapolak
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import Data_Engineering


class Slug_Detection(Data_Engineering):
    """
    Classifies intervals of pressure and temperature data as interval leading to a slug flow or intervals
    leading to a normal flow.

    Parameters
    ----------
    well : Spark Data Frame
        well data frame, includes continuous pressure and temperature data

    """
    def __init__(self, well):
        Data_Engineering.__init__(self, well)

    def jump(self):
        """
        Locates gap in the continuous data in the attribute pd_df, and groups and labels the continuous sets.
        """

        assert hasattr(self, "pd_df"), "Pandas data frame pd_df attribute must exist"
        assert not self.pd_df.empty, "Pandas data frame cannot be empty"

        self.pd_df["time_diff"] = self.pd_df["ts"].diff()  # Create time difference column

        # Create 'jump' column, where the value is True if the time difference between the current and previous
        # row is larger than one minute
        self.pd_df["jump"] = np.where(self.pd_df["time_diff"] > pd.Timedelta('1 min'),
                                      True,
                                      False)

        self.pd_df['count_id'] = self.pd_df["jump"].cumsum()  # Group continuous data together by giving it a count id

    def clean_short_sub(self, min_df_size=300):
        """
        Deletes entries in pandas data frame attribute pd_df that are continuous in time for less than min_df_size

        Parameters
        ----------
        min_df_size : int (optional)
            Minimum size of sub data frame that will be considered forward in the analysis (default is 300)
        """

        assert hasattr(self, "pd_df"), "Pandas data frame pd_df attribute must exist"
        assert not self.pd_df.empty, "Pandas data frame cannot be empty"

        # Get count of individual count ids. This gives information about how many continuous sets of data exist
        # within the pd_df attribute
        counter = self.pd_df.groupby('count_id')['count_id'].count()

        for i in range(0, len(counter)):
            if counter[i] < min_df_size:
                # Drop sets of continuous data that last for less than min_df_size minutes (default 300)
                self.pd_df = self.pd_df[self.pd_df.count_id != i]

    def sub_data(self, **kwargs):
        """
        Creates a dictionary of sub data frame that are continuous on time for over the min_df_size variable

        Parameters
        ----------
        kwargs
            min_df_size : int
                For clean_short_sub method. Minimum size of sub data frame that will be considered forward in the analysis
        """
        assert hasattr(self, "pd_df"), "Pandas data frame pd_df attribute must exist"
        assert not self.pd_df.empty, "Pandas data frame cannot be empty"

        self.sub_df_dict = {}  # Create sub_df_dict attribute which will store sub data frames of continuous data

        self.jump()             # Group continuous data together
        self.clean_short_sub()  # Drop sets of continuous data that are too short (min_df_size)
        if "min_df_size" in kwargs.keys():
            self.clean_short_sub(min_df_size=kwargs["min_df_size"])

        self.jump()  # Get new, ordered count ids

        # Fill sub_df_dict attribute with appropriate size continuous data sets
        for i in range(self.pd_df["count_id"].max() + 1):
            temp = pd.DataFrame()
            mask = self.pd_df["count_id"] == i     # Mask values that equate the current count id
            temp = self.pd_df[mask].reset_index()  # Reset indices

            self.sub_df_dict[i] = temp              # Save data frame to to dictionary

        return

    def slug_check(self, slug_idx, key,
                   dict_slug_def={"time" :[60, 240], "interval" :[25, 25], "value_diff" :[2.0, 2.0]}):
        """
        From a list of indices of peaks in the data, return a list of of indices of first slug, based on
        the time since the last slug peak, the time between slug peaks and the value difference between slug peaks

        Parameters
        ----------
        slug_idx : list of int
            List of indices of all the peaks thought to be slugs occurring in the current data frame
            in the sub_df_dict attribute
        key : int
            Index of current database in sub_df_dict
        dict_slug_def : dict (optional
            Dictionary of values to define slugs, such a minimum time sine last first slug, minimum interval between
            two slug peaks, maximum WHP difference between peaks

        Returns
        -------
        first_slug_idx : list of int
            List of indices of first slug peaks for current data frame in the dictionary attribute sub_df_dict
        """
        curr_df = self.sub_df_dict[key]  # Get current data frame from sub_df_dict dictionary

        first_slug_idx = [-300]  # Create first_slug index list. Set first value to -300

        # Variables to check if slugging period has been continuous
        first = False  # First slug has not occurred yet
        last_slug = 0  # Index of last occurring slug

        # From slug indices lis, create small lists of 3 consecutive slugs indices
        slug_check = [slug_idx[i:i + 3] for i in range(0, len(slug_idx) - 2, 1)]

        # loop through list of list of three indices
        for idx in slug_check:

            # if gap of 240 minutes since last first slug AND slugs haven't been continuous, move to next condition
            if (abs(first_slug_idx[-1] - idx[0]) < dict_slug_def['time'][1]) | (
                    (abs(last_slug - idx[0]) < dict_slug_def['time'][0]) & first):
                last_slug = idx[0]
                continue  # continue to next index list

            # if less than 20 minutes between two slug, move to next condition
            elif (abs(idx[0] - idx[1]) > dict_slug_def['interval'][0]) | (
                    abs(idx[1] - idx[0]) > dict_slug_def['interval'][1]):
                continue  # continue to next index list

            # if less than 2 bar difference between slugs (we want similar flow pattern), move to next condition
            elif (abs(curr_df["WH_P"].iloc[idx[0]] - curr_df["WH_P"].iloc[idx[1]]) > dict_slug_def['value_diff'][0]) | (
                    abs(curr_df["WH_P"].iloc[idx[1]] - curr_df["WH_P"].iloc[idx[2]]) > dict_slug_def['value_diff'][1]):
                continue  # continue to next index list

            else:
                if len(first_slug_idx) == 1:
                    first = True  # Set first to True, when the first slug of the sub data frame occurs
                first_slug_idx.append(idx[0])  # Store first_slug index value

        first_slug_idx.pop(0)  # Drop dummy value set in the first slug index list

        return first_slug_idx

    def label_slugs(self, slug_diff=3, **kwargs):
        """
        Finds slug peaks in each data frame in the sub_df_dict attribute and creates the list of indices of slugs.
        Uses the slug_check method to then compute the list of indices of first slugs occurences, per data frame in
        the sub_df_dict attribute.

        Parameters
        ----------
        slug_diff : float (optional)
            Minimum differential value above which an increase in WHP is assumed to be a slug peak (default is 3.0).
        kwargs
            dict_slug_def : dict
                For slug_check method

        Returns
        -------
        first_slug : list of list of int or list of list of datetime
            List containing the sub list of first slug indices as computed by the method slug_check per sub data frames
            in the dictionary attribute
        slugs : list of list of int or list of list of datetime
            List containing the sub list of slug peaks indices per sub data frames in the dictionary attribute
        """

        assert hasattr(self, "sub_df_dict"), "Sub_df_dict attribute must have been created"

        first_slug = []  # Create list to store first slug indices
        slugs = []  # Create list to store slug peaks indices

        for key in self.sub_df_dict:
            curr_df = self.sub_df_dict[key].copy()  # Get copy of current df out of

            # Compute WHP gradient in new column
            curr_df["WH_P_diff"] = curr_df["WH_P"].diff()
            # Label increasing trends as True and decreasing trends as False
            curr_df["trend"] = np.where(curr_df["WH_P_diff"] > 0, True, False)
            # Give the consecutive sets of WHP trend an id
            curr_df["categories"] = (curr_df["trend"] != curr_df["trend"].shift()).cumsum()

            # Calculate the cumulative sum of the WHP trend at each id
            curr_df["WH_P_diff_total"] = curr_df.groupby("categories")["WH_P_diff"].cumsum()

            # Label slug peaks as large increases (> slug_diff) immediately followed by a decrease
            curr_df["point_label"] = np.where((curr_df["WH_P_diff_total"] > slug_diff) &
                                              (curr_df["WH_P_diff_total"].shift(periods=-1) < 0),
                                              True,
                                              False)

            # Store indices of slug peaks to list
            slug_index = curr_df.index[curr_df["point_label"] is True].tolist()

            # From slug_check method, compute first slug indices
            first = self.slug_check(slug_index, key)
            if 'dict_slug_def' in kwargs.keys():
                first = self.slug_check(slug_index, key, dict_slug_def=kwargs['dict_slug_def'])

            first_slug.append(first)  # Store first slug indices for current df to list
            slugs.append(slug_index)  # Store slug peaks indices for current df to list

        return first_slug, slugs

    def format_data(self, first_slug, size_list=240, max_clean_count=10000):
        """
        Formats data for classification algorithms. Splits down the sub_df_dict attribute's data frames into size_list
        sized data frame. Data frames containing first slugs, as labelled in the method label_slugs, are split right
        before the occurrence of a first slug.

        Parameters
        ----------
        first_slug : list of list of int
            List containing the sub list of first slug indices as computed by the method slug_check per sub data frames
            in the dictionary attribute
        size_list : int (optional)
            Size of data frame to use for classification (default is 240)
        max_clean_count : int (optional)
            Maximum number of data frames to create that lead to a normal, clean, flow. This is known by whether the
            first slug list is empty (default is 10000)
        """

        self.label = np.array([])  # Create label array to store labels of data frames
        self.df_list = []  # Create data frame list to store size_list sized data frames

        cnt = 0  # Counter, for max_clean_count

        for df, f in zip(self.sub_df_dict.values(), first_slug):

            if not f:  # If first slug list is empty (no slugs occurring in current data frame)
                if len(df) >= size_list:  # Check that data frame has enough data points
                    if cnt < max_clean_count:

                        # Drop last hour of data. It is not known if it would have led to a slug or normal flow
                        df = df[:-60]

                        # Compute number of splits that can be performed on the sub data frame
                        n_splits = int(np.ceil(len(df) / size_list))
                        if n_splits > 1:
                            # Compute the overlap value if any
                            overlap = int(np.ceil(((size_list * (n_splits - 1)) - (len(df) - size_list)) / (
                                    n_splits - 1)))
                        else:
                            overlap = 0

                        # Add data frames of size size_list to df_list
                        self.df_list.extend([df[i: i + size_list] for i in range(0, (len(df) - (size_list - overlap)), (
                                size_list - overlap))])

                        # Append corresponding labels to label array
                        for i in range(0, (len(df) - (size_list - overlap)), (size_list - overlap)):
                            self.label = np.append(self.label, [0])
                            cnt += 1  # Count number of clean/normal intervals added

            else:  # If first slugs are present in current data frame
                for first in f:
                    if first - size_list >= 0:  # Check there's sufficient number of points before first slug
                        self.df_list.append(df[first - size_list: first])  # Add data frames to df_list
                        self.label = np.append(self.label, [1])  # Append corresponding labels to label array

        return

    def feature_vector(self, split_num=5, time_predict=60, percentage_significance=0.1, standardise=True):
        """
        Transform classification data into feature vectors.

        Parameters
        ----------
        split_num : int (optional)
            Number of time to split the data frame (default is 5)
        time_predict : int (optional)
            Number of minutes to ignore at the end of the interval. This number will also be the number of minutes the
            classifier is trained to recognize a slug before it occurs (default is 60)
        percentage_significance : float (optional)
            For the significant feature, percentage value for which to state an increase/decrease in the data was
            significant compare to the original value. Must be below 1 (default is 0.10)
        standardise : bool (optional)
            Whether to standardise the data or not (default is True)
        """

        assert percentage_significance <= 1, "percentage_significance must be a decimal/percentage"
        assert time_predict < len(self.df_list[0]), "Time to prediction before must be less than data frame size"

        for k in range(len(self.df_list)):
            clean_df = self.df_list[k].copy()  # Get copy of current current data frame

            if standardise:  # standardise the data
                clean_df = self.standardise(clean_df)

            clean_df = clean_df[:-time_predict]  # Ignore data within prediction time, default 1 hour
            clean_df = clean_df.drop("WH_choke", axis=1)  # Drop choke data (always opened)
            self.features = ["WH_P", "DH_P", "WH_T", "DH_T"]  # Update features list attribute

            interval = int(len(clean_df) / split_num)  # Get window size

            clean = []  # Create empty list to store current data frames feature
            if k == 0:
                header = []  # Create empty list to store features' names

            for i in range(split_num):
                low = i * interval  # Lower bound of interval
                high = (i + 1) * interval  # Upper bound of interval

                for f in self.features:
                    if k == 0:
                        header.append(("mean_" + f + "_" + str(i)))  # Store mean feature header name
                        header.append(("std_" + f + "_" + str(i)))  # Sotre std feature header name

                    clean.append(clean_df[f][low:high].mean())  # Append interval's mean feature value
                    clean.append(clean_df[f][low:high].std())  # Append interval's std feature value

            if k == 0:
                self.X = pd.DataFrame([clean], columns=[*header])  # Create new data frame X to store feature vectors
            else:
                dic = dict(zip(header, clean))
                self.X = self.X.append(dic, ignore_index=True)  # Append data from current data frame to X attribute

        for i in range(split_num - 1):
            for f in self.features:
                # Get delta mean feature
                self.X["diff_mean_" + f + "_" + str(i) + "_" + str(i + 1)] = self.X["mean_" + f + "_" + str(i + 1)] - \
                                                                             self.X["mean_" + f + "_" + str(i)]
                # Get delta std feature
                self.X["diff_std_" + f + "_" + str(i) + "_" + str(i + 1)] = self.X["std_" + f + "_" + str(i + 1)] - \
                                                                            self.X["std_" + f + "_" + str(i)]

                # Get mean trend feature (1 increase, 0 decrease)
                self.X["diff_mean_trend_" + f + "_" + str(i) + "_" + str(i + 1)] = np.where(
                    self.X["diff_mean_" + f + "_" + str(i) + "_" + str(i + 1)] > 0, 1, 0)
                # Get std trend feature
                self.X["diff_std_trend_" + f + "_" + str(i) + "_" + str(i + 1)] = np.where(
                    self.X["diff_std_" + f + "_" + str(i) + "_" + str(i + 1)] > 0, 1, 0)

                # Get significance delta feature
                # Significant delat is a difference of more than percentage_significance, default 10%
                self.X["diff_mean_signif_" + f + "_" + str(i) + "_" + str(i + 1)] = np.where(
                    self.X["diff_mean_" + f + "_" + str(i) + "_" + str(i + 1)] > self.X[
                        ("mean_" + f + "_" + str(i))] * percentage_significance, 1, 0)
                self.X["diff_std_signif_" + f + "_" + str(i) + "_" + str(i + 1)] = np.where(
                    self.X["diff_std_" + f + "_" + str(i) + "_" + str(i + 1)] > self.X[
                        ("std_" + f + "_" + str(i))] * percentage_significance, 1, 0)

        # Get count of significant of increases and count of significant decreases features
        for f in self.features:
            binary_mean_diff_col_names = []
            binary_mean_trend_col_names = []
            binary_std_diff_col_names = []
            binary_std_trend_col_names = []
            for i in range(split_num - 1):
                binary_mean_trend_col_names.append("diff_mean_trend_" + f + "_" + str(i) + "_" + str(i + 1))
                binary_mean_diff_col_names.append("diff_mean_signif_" + f + "_" + str(i) + "_" + str(i + 1))
                binary_std_trend_col_names.append("diff_std_trend_" + f + "_" + str(i) + "_" + str(i + 1))
                binary_std_diff_col_names.append("diff_std_signif_" + f + "_" + str(i) + "_" + str(i + 1))

            # Count of significant increases
            self.X["num_mean_" + f + "_sign_incr"] = self.X[binary_mean_diff_col_names].sum(axis=1).where(
                self.X[binary_mean_trend_col_names] == 1, 0)
            self.X["num_std_" + f + "_sign_incr"] = self.X[binary_std_diff_col_names].sum(axis=1).where(
                self.X[binary_std_trend_col_names] == 1, 0)

            # Count of significant decreases
            self.X["num_mean_" + f + "_sign_decr"] = self.X[binary_mean_diff_col_names].sum(axis=1).where(
                self.X[binary_mean_trend_col_names] == 0, 0)
            self.X["num_std_" + f + "_sign_decr"] = self.X[binary_std_diff_col_names].sum(axis=1).where(
                self.X[binary_std_trend_col_names] == 0, 0)

        return

    def split_data(self, test_size=0.3):
        """
        Split the data into a training and testing set

        Parameters
        ----------
        test_size : float (optional)
            Percentage of the data to include in the training set (default is 0.3)
        """

        assert (test_size <= 1), "Test size must be a percentage, less than 1"

        sss = StratifiedShuffleSplit(n_splits=2, test_size=test_size, random_state=0)  # Instantiate Shuffle Split
        sss.get_n_splits(self.X, self.label)

        train_index, test_index = sss.split(self.X, self.label)  # Split data

        # Store train and test data into respective X and y attributes
        self.X_train, self.X_test = self.X.iloc[train_index[0]], self.X.iloc[test_index[1]]
        self.y_train, self.y_test = self.label[train_index[0]], self.label[test_index[1]]

        return

    def RF_train(self, n_estimators=15, max_depth=None, bootstrap=True, **kwargs):
        """
        Train the Random Forest model

        Parameters
        ----------
        n_estimators : int
            Number of Decision Trees in Random Forest model (optional, default is 15)
        max_depth : int
            Maximum depth of the Decisions Trees in Random Forest model (optional, default is None
        bootstrap : bool
            Whether bootstrap samples are used when building decision trees. If False, the whole datset is used to
            build each tree in the Random Forest (optional, default is True)
        kwargs
            test_size : float
                For split_data method, percentage repartition of testing data
        """

        test_size = 0.3
        if "test_size" in kwargs.keys():
            test_size = kwargs["test_size"]
        self.split_data(test_size=test_size)

        # Instantiate RF model
        self.rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
        self.rf.fit(self.X_train, self.y_train)  # Fit model to RF model

        pred_features = self.rf.feature_importances_  # Get features scores

        return pred_features

    def feature_selection(self, feature_score, top_n=15):
        """
        Selects the top_n most important features from the feature_score list

        Parameters
        ----------
        feature_score : list of floats
            List of feature scores, in order. As produced by the RF_train method
        top_n : int
            Number of features to return

        Returns
        -------
        top_feature : list of str
            List containing the names of the top_n features

        """

        assert (len(feature_score) == len(self.X.columns)), "Feature numbers must be the same in both lists"

        from collections import namedtuple

        feature_s = []  # Empty list to store feature score and feature name
        Feature = namedtuple('Feature', 'name score')  # New empty tuple format

        for f, s in zip(self.X.columns, feature_score):
            feature_s.append(Feature(f, s))  # Store feature score with feature names

        feature_ranked = sorted(feature_s, key=lambda x: getattr(x, 'score'), reverse=True)  # Sort features by score

        top_feature = []
        for i in range(top_n):
            top_feature.append(feature_ranked[i][0])  # Get list of names of top top_n features

        return top_feature

    def RF_predict(self, true_label=True, **kwargs):
        """
        Predict data labels from trained Random Forest model

        Parameters
        ----------
        true_label : bool (optional)
            Whether the data used has true labels (default is True)
        kwargs
            X_test : Pandas DataFrame
                For user, if additional data needs to be predicted. Data frame needs to be feature vector format
            y-test : Pandas Series or Numpy array
                For user, labels of inputted X_test data

        Returns
        -------
        pred : array
            prediction labels for the test data
        score : float
            Accuracy score of the predictions, if true_label = True
        cf : array
            Confusion matrix, as created by Scikit Learn's Confusion matrix on the accuracy of the results, if
            true_label = True
        """

        # If new data is passed to be classified
        if "X_test" in kwargs.keys():
            self.X_test = kwargs["X_test"]

        pred = self.rf.predict(self.X_test)  # Predict labels of data

        if true_label:  # If the true labels are known

            if "y_test" in kwargs.keys():
                self.y_test = kwargs["y_test"]  # if new true labels are passed

            assert (len(self.y_test) == len(self.X_test)), "X and y must be the same length"

            # Compute prediction score and confusion matrix
            score = self.rf.score(self.X_test, self.y_test)
            cf = confusion_matrix(self.y_test, pred)

            return pred, score, cf

        else:
            return pred

    def LogReg_train(self, top_features=[], C=0.01, max_iter=50, split=True, **kwargs):
        """
        Train Logistics Regression model

        Parameters
        ----------
        top_features : list of str (optional)
            List of the top features names. If list not empty, the Logistics Regression model will only be trained using
            the listed features. (default is [] meaning all features)
        C : float (optional)
            Regularization parameter for Logitics Regression (default is 0.01)
        max_iter : int (optional)
            Maximum iteration parameter for Logistics Regression (default is 50)
        split : bool
            True for new split, False to use same split as RF model was trained on
        kwargs
            test_size : float
                For split_data method, percentage repartition of testing data
        """

        # Split data into a training and testing set if new split is required
        if split:
            test_size = 0.3
            if "test_size" in kwargs.keys():
                test_size = kwargs["test_size"]
            self.split_data(test_size=test_size)

        if len(top_features) != 0:  # If top features selection are passed
            self.logreg_features = top_features  # Store top features as an attribute
            self.X_train = self.X_train[[*self.logreg_features]]  # Update feature vector selection

        # Instantiate and fit Logistics Regression model to training data
        self.log = LogisticRegression(C=C, max_iter=max_iter)
        self.log.fit(self.X_train, self.y_train)

        return

    def LogReg_pred(self, true_label=True, **kwargs):
        """
        Predict data labels from trained Logistics Regression model

        Parameters
        ----------
        true_label : bool (optional)
            Whether the data used has true labels (default is True)
        kwargs
            X_test : Pandas DataFrame
                For user, if additional data needs to be predicted. Data frame needs to be feature vector format
            y-test : Pandas Series or Numpy array
                For user, labels of inputted X_test data

        Returns
        -------
        pred : array
            prediction labels for the test data
        proba : array
            prediction probababilities for the test data
        score : float
            Accuracy score of the predictions, if true_label = True
        cf : array
            Confusion matrix, as created by Scikit Learn's Confusion matrix on the accuracy of the results, if
            true_label = True

        """

        # If new data is passed to be classified
        if "X_test" in kwargs.keys():
            self.X_test = kwargs["X_test"]

        # If only top features are used
        if hasattr(self, "logreg_features"):
            self.X_test = self.X_test[[*self.logreg_features]]

        # Predict labels and probabilities of data
        log_pred = self.log.predict(self.X_test)
        proba = self.log.predict_proba(self.X_test)

        if true_label:  # If the true labels are known
            if "y_test" in kwargs.keys():
                self.y_test = kwargs["y_test"]  # if new true labels are passed

            # Compute prediction score and confusion matrix
            score = self.log.score(self.X_test, self.y_test)
            cf = confusion_matrix(self.y_test, log_pred)

            return log_pred, proba, score, cf

        else:
            return log_pred, proba

    def data_prep(self, **kwargs):
        """
        Quick data preparation from raw pandas data frame to feature vectors

        Parameters
        ----------
        kwargs
            slug_diff : float
                 Argument of label_slugs method. Minimum WHP differential value to be assumed a slug peak
            size_list : int
                Argument of format_data method. Size of data frame to use for classification
            max_clean_count : int
                Argument of format_data method. Maximum number of data frames to create that lead to a normal flow.
            split_num : int
                Argument of feature_vector method. Number of splits the data frame
            time_predict : int
                Argument of feature_vector method. Number of minutes before slug flow.
            percentage_significance : float
                Argument of feature_vector method. For the significant feature, percentage value for which to state
                an increase/decrease in the data was significant compare to the original value.
            standardise : bool
                Argument of feature_vector method. Whether to standardise the data or not.
        """

        assert hasattr(self, "pd_df"), "Pandas data frame pd_df attribute must exist"
        assert not self.pd_df.empty, "Pandas data frame cannot be empty"

        self.sub_data(**kwargs)  # Split original data frame into smaller continuous data frames

        slug_diff = 3
        if "slug_diff" in kwargs.keys():
            slug_diff = kwargs["slug_diff"]

        first_idx, slug_idx = self.label_slugs(slug_diff=slug_diff)  # Get first slug indices list

        size_list = 300
        if "size_list" in kwargs.keys():
            size_list = kwargs["size_list"]
        max_clean_count = 10000
        if "max_clean_count" in kwargs.keys():
            max_clean_count = kwargs["max_clean_count"]

        # Format data into size_list (default 300) long data frames, with label list
        self.format_data(first_idx, size_list=size_list, max_clean_count=max_clean_count)

        # check for kwargs
        split_num = 5
        if "split_num" in kwargs.keys():
            split_num = kwargs["window_size"]
        time_predict = 60
        if "time_predict" in kwargs.keys():
            time_predict = kwargs["time_predict"]
        percentage_significance = 0.1
        if "percentage_significance" in kwargs.keys():
            percentage_significance = kwargs["percentage_significance"]
        standardise = True
        if "standardise" in kwargs.keys():
            standardise = kwargs["standardise"]

        # Create data feature vectors
        self.feature_vector(split_num=split_num, time_predict=time_predict,
                            percentage_significance=percentage_significance, standardise=standardise)

        return

    def plot_raw_slugs(self, variables=["WH_P", "DH_P"], scaled=True, n_examples=10, first_sample=10, **kwargs):
        """
        Plotting functions to plot a set of n_examples raw sub_df_dict data frame, showing the slugs peaks and first
        slugs value.

        Parameters
        ----------
        variables : list of str (optional)
            Names of variables to be plotted (default is ["WH_P", "DH_P"])
        scaled : bool (optional)
            Whether to scale variables to WH_P (default is True)
        n_examples : int (optional)
            Number of examples to plot (default is 10)
        first_sample : int (optional)
            Index of the sub_df_dict to start plotting from (default is 10)
        kwargs
            slug_diff : float
                 Argument of label_slugs method. Minimum WHP differential value to be assumed a slug peak
        """

        assert hasattr(self, "pd_df"), "Pandas data frame pd_df attribute must exist"
        assert not self.pd_df.empty, "Pandas data frame cannot be empty"

        for v in variables:
            assert (v in self.pd_df.columns)  # Assert variables name exist

        x_list = [x for x in range(first_sample, first_sample + n_examples)]  # Create list of data frames to plot

        self.sub_data(**kwargs)  # Split data frame into smaller continuous data frames

        # check for kwargs
        slug_diff = 3
        if "slug_diff" in kwargs.keys():
            slug_diff = kwargs["slug_diff"]

        f, s = self.label_slugs(slug_diff=slug_diff)  # Get first slug indexes

        fig, ax = plt.subplots(n_examples, 1, figsize=(20, 20))  # Create plot with the n_examples
        plt.tight_layout()

        for i, x in enumerate(x_list):

            for v in variables:
                if (v != "WH_P") & scaled:  # If variables are to be scaled to WHP
                    ax[i].plot(self.sub_df_dict[x]["ts"], self.sub_df_dict[x][v] - self.sub_df_dict[x][v][0] + 20, "-",
                               label=str(v))
                else:
                    ax[i].plot(self.sub_df_dict[x]["ts"], self.sub_df_dict[x][v], "-", label=str(v))

            # Plot slug peaks in red
            ax[i].plot(self.sub_df_dict[x]["ts"][s[x]], self.sub_df_dict[x]["WH_P"][s[x]], "ro", label="Slug peaks")
            # Plot first slug in magenta
            ax[i].plot(self.sub_df_dict[x]["ts"][f[x]], self.sub_df_dict[x]["WH_P"][f[x]], "m*", markersize=20,
                       label="First slug")
            # Plot start of interval in cyan
            ax[i].plot(self.sub_df_dict[x]["ts"][f[x]] - pd.Timedelta('5 h'), self.sub_df_dict[x]["WH_P"][f[x]], "c*",
                       markersize=20, label="Start interval")
            ax[i].set_xlabel("Time")
            ax[i].grid(True, which='both')
            ax[i].set_ylabel("Pressure in BarG")
            ax[i].legend()

        display(fig)

    def plot_X(self, start=60, variables=["WH_P", "DH_P"], scaled=True, n_examples=3):
        """
        Plot n_examples each of the set size_list intervals leading to normal or slug flow

        Parameters
        ----------
        start : int (optional)
            Index of the df_list to start plotting from (default is 60)
        variables : list of str (optional)
            Names of variables to be plotted (default is ["WH_P", "DH_P"])
        scaled : bool (optional)
            Whether to scale variables to WH_P (default is True)
        n_examples : int  (optional)
            Number of examples to plot (default is 3)

        """

        assert hasattr(self, "df_list"), "df_list attribute must exist"
        assert len(self.df_list) != 0, "df_list attribute cannot be empty"

        fig, ax = plt.subplots(int(n_examples), 2, figsize=(15, 7))
        plt.tight_layout()

        data1, data2 = True, True   # Create bool variables. If False, n_examples have been plotted already
        plot1, plot2 = 0, 0     # Plot axis number for each class. Increases when a class example has been plotted.
        for idx, val in enumerate(self.label):
            if idx > start:

                # Plot the first n_examples examples of intervals leading to a slug flow class
                if (val == 1) & data1:
                    for v in variables:
                        if (v != "WH_P") & scaled:  # If variables are to be scaled to WHP
                            ax[plot1][0].plot(self.df_list[idx]["ts"],
                                              self.df_list[idx][v] - self.df_list[idx][v].iloc[0] + 20, "-",
                                              label=str(v))
                        else:
                            ax[plot1][0].plot(self.df_list[idx]["ts"], self.df_list[idx][v], "-", label=str(v))
                    ax[plot1][0].set_xlabel("Time")
                    ax[plot1][0].grid(True, which='both')
                    ax[plot1][0].set_ylabel("Pressure in barG")
                    ax[0][0].set_title("Interval leading to slug flow")
                    ax[plot1][0].legend()
                    plot1 += 1
                    if plot1 == n_examples:
                        data1 = False

                # Plot the first n_examples examples of intervals leading to a normal flow class
                elif (val == 0) & (data2):
                    for v in variables:
                        if (v != "WH_P") & (scaled):
                            ax[plot2][1].plot(self.df_list[idx]["ts"],
                                              self.df_list[idx][v] - self.df_list[idx][v].iloc[0] + 20, "-",
                                              label=str(v))
                        else:
                            ax[plot2][1].plot(self.df_list[idx]["ts"], self.df_list[idx][v], "-", label=str(v))
                    ax[plot2][1].set_xlabel("Time")
                    ax[plot2][1].grid(True, which='both')
                    ax[plot2][1].set_ylabel("Pressure in barG")
                    ax[0][1].set_title("Interval leading to normal flow")
                    ax[plot2][1].legend()
                    plot2 += 1
                    if plot2 == n_examples:
                        data2 = False

        display(fig)


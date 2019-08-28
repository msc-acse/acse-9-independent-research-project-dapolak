# -*- coding: utf-8 -*-
"""
Part of slugdetection package

@author: Deirdree A Polak
github: dapolak
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix

import slugdetection.Data_Engineering as Data_Engineering


class Flow_Recognition(Data_Engineering):
    """
    Classifies different flow types within an oil well using feature vectors and classification algorithms

    Parameters
    ----------
    well : Spark data frame
        Spark data frame containing pressure and temperature data.
    """

    def __init__(self, well):
        super().__init__(well)

    def label_slugs(self, slug_diff=3, choke_diff=1, pre_slug_period=40):
        """
        Method labels the data in five different categories: slugs, first slugs, pre-slug, normal and ignore

        Parameters
        ----------
        slug_diff : int
            differential value for WHP above which a flow is considered to be a slug
        choke_diff : int
            differential value for WH choke below which the choke is considered opened and constant
        pre_slug_period : int
            number of minutes before first slug that is considered to be in "pre-slug" phase

        """

        assert hasattr(self, "pd_df"), "Pandas data frame pd_df attribute must exist"
        assert not self.pd_df.empty, "Pandas data frame cannot be empty"

        self.pd_df["WH_choke_diff"] = self.pd_df["WH_choke"].diff()  # Calculate gradient of WH choke
        self.pd_df["WH_P_diff"] = self.pd_df["WH_P"].diff()  # Calculate gradient of WHP

        self.pd_df["label"] = ""  # Create label column, keep empty

        # Label slug flow points where the WH choke is constant and the WHP is increasing
        mask = (self.pd_df["WH_choke_diff"] < choke_diff) & (self.pd_df["WH_P_diff"] > slug_diff)
        self.pd_df["label"].loc[mask] = "slug"

        # Label normal flow points when the WH choke is constant and the gradient is low
        mask = (self.pd_df["WH_choke_diff"] < choke_diff) & (self.pd_df["WH_P_diff"] <= slug_diff)
        self.pd_df["label"].loc[mask] = "normal"

        # Label all other points as choked / irregular flow -> ignore
        mask = (self.pd_df["label"] == "")
        self.pd_df["label"].loc[mask] = "ignore"

        # Label the first slug of the slug flow and the pre_slugs points
        slug_index = self.pd_df.index[self.pd_df["label"] == "slug"].tolist()  # Get slugs indices
        first_index = []  # Create empty list to store indices of first slugs
        pre_slug = []  # Create empty list to store indices of pre_slugs

        for i in range(1, len(slug_index)):  # for every slug index
            if abs(slug_index[i] - slug_index[i - 1]) > 50:  # If no slugs has occured in the past 50 minutes
                first_index.append(slug_index[i])  # Save slug point as first slug
                # Save indices of all the points within the pre_slug period
                pre_slug.extend([i for i in range(slug_index[i] - pre_slug_period, slug_index[i])])

        # Label first slugs
        self.pd_df["label"].loc[first_index] = "first_slug"

        # Label pre slugs
        self.pd_df["label"].loc[pre_slug] = "pre_slug"

        # Drop gradient columns
        self.pd_df = self.pd_df.drop(["WH_P_diff", "WH_choke_diff"], axis=1)

        return

    def window_label(self, label_list):
        """
        Defines the overall label of a feature vector/window of data points

        Parameters
        ----------
        label_list : list of str
            List of the labels of each data point within the window

        Returns
        -------
        label : str
            Label of feature vector/data window
        """

        if "first_slug" in label_list:
            return 0  # "first_slug"
        if "pre_slug" in label_list:
            return 2  # "pre_slug"
        if label_list.count("slug") > 2:
            return 1  # "slug_flow"
        if label_list.count("normal") > 10:
            return 3  # "normal"
        if label_list.count("ignore") > 5:
            return 4  # "ignore"
        else:
            return 4  # "ignore"

    def feature_vector(self, window_size, step, standardise=True):
        """
        Converts the data list into feature vector of size window_size

        Parameters
        ----------
        window_size : int
            Window size of feature vector
        step : int
            Time step to take between the start of two windows
        standardise : bool
            Whether to standardise the data

        """

        assert hasattr(self, "pd_df"), "Pandas data frame pd_df attribute must exist"
        assert not self.pd_df.empty, "Pandas data frame cannot be empty"
        assert "label" in self.pd_df.columns, "Data must have been labelled"

        self.window_size = window_size  # Create window_size attribute

        self.feature_vec = self.pd_df.copy()  # Create feature vector data frame
        self.features.append("label")  # Add label to the feature list attribute

        labels = []  # Empty list to store label column names

        # Standardise data
        if standardise:
            self.feature_vec = self.standardise(self.feature_vec)

        # Create feature vector which takes into account the values of all the features within the window size
        for i in range(1, self.window_size + 1):
            new_features = []  # New list to keep track of the new column names at current time step
            for feature_ in self.features:
                new_features.append(feature_ + "_" + str(i))
                if feature_ == "label":
                    labels.append(new_features[-1])  # Save lagged label names
            # For every minute step, create a new lagged column for each features
            self.feature_vec[new_features] = self.feature_vec[self.features].shift(periods=-i).fillna(0)

        # Create a new column which regroups all of the labels into one vector
        self.feature_vec["label_total"] = self.feature_vec[labels].values.tolist()
        self.feature_vec = self.feature_vec.drop(labels, axis=1)  # Drop all the other lagged label columns

        self.feature_vec["window_label"] = ""  # Create new column for the label of the whole window
        # Compute the label of the window by passing through the vector of labels of the individual points
        self.feature_vec["window_label"].loc[:] = self.feature_vec["label_total"].apply(self.window_label)
        self.feature_vec = self.feature_vec.drop("label_total", axis=1)  # Drop all the vector label column

        self.feature_vec = self.feature_vec[self.window_size:]  # get rid of 0 values cause by lag
        self.feature_vec = self.feature_vec[::step]  # Keep only every step-th row

        self.feature_vec = self.feature_vec.drop(["label", "ts"], axis=1)  # Drop all none features
        self.features.remove("label")  # Remove label from the features attribute
        return

    def split_data(self, test_size=0.3):
        """
        Split the data into a training and testing set

        Parameters
        ----------
        test_size : float (optional)
            Percentage of the data to include in the testing set (default is 0.3)
        """
        assert (test_size <= 1), "Test size must be a percentage"
        assert hasattr(self, "feature_vec"), "Feature vector attribute must have been created"

        X = self.feature_vec.drop("window_label", axis=1).copy()  # Data X
        y = self.feature_vec["window_label"].copy()  # Labels y

        # Split data and labels into test and training set
        sss = StratifiedShuffleSplit(n_splits=2, test_size=test_size, random_state=0)
        sss.get_n_splits(X, y)
        train_index, test_index = sss.split(X, y)

        # Create attributes of the training and testing data
        self.y_train, self.y_test = y.iloc[train_index[0]], y.iloc[test_index[1]]
        self.X_train, self.X_test = X.iloc[train_index[0]], X.iloc[test_index[1]]

        return

    def LogReg_train(self, **kwargs):
        """
        Train Logistics Regression model

        Parameters
        ----------
        kwargs:
            test_size : float
                Percentage of the data to include in the testing set for the split_data method
        """

        # Split data into a training and testing set
        self.split_data()
        if "test_size" in kwargs.keys():
            self.split_data(test_size=kwargs["test_size"])

        # Instantiate and fit Logistics Regression model to training data
        self.log = LogisticRegression()
        self.log.fit(self.X_train, self.y_train)

        return

    def LogReg_pred(self, true_label=True, **kwargs):
        """
        Predict data labels from trained Logistics Regression model

        Parameters
        ----------
        true_label (bool)
            Whether the data used has true labels (optional, default is True)
        kwargs
            X_test : Pandas DataFrame
                For user, if additional data needs to be predicted. Data frame needs to be feature vector format
            y-test : Pandas Series or Numpy array
                For user, labels of inputted X_test data

        Returns
        -------
        pred : numpy array
            prediction labels for the test data
        proba : numpy array
            prediction probababilities for the test data
        score : float
            Accuracy score of the predictions, if true_label = True
        cf : numpy array
            Confusion matrix, as created by Scikit Learn's Confusion matrix on the accuracy of the results, if
            true_label = True

        """

        # If new data is passed to be classified
        if "X_test" in kwargs.keys():
            self.X_test = kwargs["X_test"]

        # Predict labels and probabilities of labels
        log_pred = self.log.predict(self.X_test)
        proba = self.log.predict_proba(self.X_test)

        # If the labels are known
        if true_label:

            # if new true labels are passed
            if "y_test" in kwargs.keys():
                self.y_test = kwargs["y_test"]

            assert (len(self.y_test) == len(self.X_test)), "X and y must be the same length"

            # Compute prediction score and confusion matrix
            score = self.log.score(self.X_test, self.y_test)
            cf = confusion_matrix(self.y_test, log_pred, labels=[0, 1, 2, 3, 4])

            return log_pred, proba, score, cf

        else:
            return log_pred, proba

    def SVM_train(self, krnl='rbf', **kwargs):
        """
        Train Support Vector Classifier model

        Parameters
        ----------
        krnl : str
            kernel to be used for SVC model
        kwargs:
            test_size : float
                Percentage of the data to include in the testing set for the split_data method
        """

        # Split data into a training and testing set
        self.split_data()
        if "test_size" in kwargs.keys():
            self.split_data(test_size=kwargs["test_size"])

        # Instantiate and fit Logistics Regression model to training data
        self.svm = SVC(kernel=krnl)
        self.svm.fit(self.X_train, self.y_train)

        return

    def SVM_pred(self, true_label=True, **kwargs):
        """
        Predict data labels from trained Support Vector Classifier model

        Parameters
        ----------
        true_label (bool)
            Whether the data used has true labels (optional, default is True)
        kwargs
            X_test : Pandas DataFrame
                For user, if additional data needs to be predicted. Data frame needs to be feature vector format
            y-test : Pandas Series or Numpy array
                For user, labels of inputted X_test data

        Returns
        -------
        pred : numpy array
            prediction labels for the test data
        score : float
            Accuracy score of the predictions, if true_label = True
        cf : numpy array
            Confusion matrix, as created by Scikit Learn's Confusion matrix on the accuracy of the results, if
            true_label = True

        """

        # If new data is passed to be classified
        if "X_test" in kwargs.keys():
            self.X_test = kwargs["X_test"]

        svm_pred = self.svm.predict(self.X_test)  # Predict labels

        if true_label:  # If true labels of data are known

            # if new true labels are passed
            if "y_test" in kwargs.keys():
                self.y_test = kwargs["y_test"]

            assert (len(self.y_test) == len(self.X_test)), "X and y must be the same length"

            # Compute prediction score and confusion matrix
            score = self.svm.score(self.X_test, self.y_test)
            cm = confusion_matrix(self.y_test, svm_pred, labels=[0, 1, 2, 3, 4])

            return svm_pred, score, cm

        else:
            return svm_pred

# -*- coding: utf-8 -*-
"""
Part of slugdetection package

@author: Deirdree A Polak
github: dapolak
"""


import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import Data_Engineering
import Slug_Detection

import unittest


class Test_Slug_Detection(unittest.TestCase):
    """
    Unitest class for the Slug Detection class
    """
    def test_create_class(self, spark_data):
        """
        Unit test for class creation

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        test_class = Slug_Detection(spark_data)
        assert hasattr(test_class, "well_df"), "Assert well_df attribute is created"
        assert len(test_class.well_df.head(1)) != 0, \
            "well_df attribute not empty"  # Pyspark has no clear empty attribute

    def test_jump(self, spark_data):
        """
        Unit test for jump method

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard Data Engineering
        test_class = Slug_Detection(spark_data)
        test_class.timeframe(start="12-SEP-16 09:09",
                             end="18-SEP-16 09:09")  # known interval that has 3 section of data over 99% choke
        test_class.data_range()
        test_class.clean_choke(method="99")
        sd_df = test_class.df_toPandas()

        test_class.jump()
        assert 'count_id' in test_class.pd_df.columns, "Assert new count_id column was created"
        assert test_class.pd_df['count_id'].nunique() >= 3, \
            "For this example, assert that there are three continuous sets of data"

    def test_clean_short_sub(self, spark_data):
        """
        Unit test for clean_short_sub method

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        test_class = Slug_Detection(spark_data)
        test_class.timeframe(start="12-SEP-16 09:09",
                             end="18-SEP-16 09:09")  # known interval that has 3 section of data over 99% choke
        test_class.data_range()
        test_class.clean_choke(method="99")
        sd_df = test_class.df_toPandas()
        test_class.jump()

        a = len(test_class.pd_df)  # Store length of pd_df data frame
        test_class.clean_short_sub(min_df_size=200)  # Apply clean_short_sub method
        b = len(test_class.pd_df)  # Store length of pd_df data frame

        assert a > b, "For this example, the post clean_short_sub pd_df attribute should be shorter"

    def test_sub_data(self, spark_data):
        """
        Unit test for clean_short_sub method

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard Data Engineering
        test_class = Slug_Detection(spark_data)
        test_class.timeframe(start="12-SEP-16 09:09",
                             end="18-SEP-16 09:09")  # known interval that has 3 section of data over 99% choke
        test_class.data_range()
        test_class.clean_choke(method="99")
        sd_df = test_class.df_toPandas()

        test_class.sub_data(min_df_size=200)

        assert hasattr(test_class, "sub_df_dict"), "New attribute must have been created"

        a = test_class.pd_df["count_id"].nunique()
        assert a == len(test_class.sub_df_dict), "Number of unique count ids must be the same as number of data " \
                                                 "frames in sub_df_dict dictionary"

        a = test_class.sub_df_dict[0]  # Get first element of the dictionary
        assert isinstance(a, pd.DataFrame), "sub_df_dict elements are pandas data frames"

        for f in test_class.features:
            assert f in a.columns, "data frame must contain all features"

    def test_slug_check(self, spark_data):
        """
        Unit test for slug_check method

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard Data Engineering steps
        test_class = Slug_Detection(spark_data)
        test_class.timeframe(start="18-SEP-16 01:09", end="18-SEP-16 09:09")  # example interval
        test_class.data_range()
        test_class.clean_choke(method="99")
        sd_df = test_class.df_toPandas()
        test_class.sub_data()

        ## Test 1 : Test that slug_check returns right value
        ##
        # Create fake dataframe
        datetime_format = '%d-%b-%y %H:%M'  # datetime date format
        base = datetime.strptime("01-JAN-16 09:09", datetime_format)  # Create datetime type timestamp
        date_list = [[base + timedelta(minutes=x)] for x in range(1000)]  # Create list of timestamps

        x = np.linspace(0, 100 * np.pi, 1000)  # Get evenly spaced x array
        whp_list = (np.sin(x) * 3) + 10  # Create sin wave array (slug-like)

        fake_df = pd.DataFrame(data=date_list, columns=["ts"], dtype=str)  # Create data frame with timestamp
        fake_df["ts"] = pd.to_datetime(fake_df["ts"])  # Ensure timestamp are datetime type
        fake_df["WH_P"] = whp_list  # Add sine wave as WHP data

        test_class.sub_df_dict = {
            1: fake_df
        }  # Override sub_df_dict attribute with fake data frame

        slug_idx = pd.Series(whp_list)[whp_list > 12.90].index.tolist()  # Create list of slug peaks for fake slugs

        first = test_class.slug_check(slug_idx, 1)  # Get results from slug_check method

        assert len(first) == 1, "First slug index list should only contain one value in this example"

        ## Test 2 : Test that slug_check returns right value
        ##
        # Create fake data frame
        datetime_format = '%d-%b-%y %H:%M'  # datetime date format
        base = datetime.strptime("01-JAN-16 09:09", datetime_format)  # Create datetime type timestamp
        date_list = [[base + timedelta(minutes=x)] for x in range(2300)]  # Create list of timestamps

        x = np.linspace(0, 100 * np.pi, 1000)  # Get evenly spaced x array
        whp_list = (np.sin(x) * 3) + 10  # Create sin wave array (slug-like)
        whp_list = np.append(whp_list, [10 for i in range(300)])  # Add flat flow to simulate normal flow
        whp_list = np.append(whp_list, (np.sin(x) * 3) + 10)  # Add more slugs

        fake_df = pd.DataFrame(data=date_list, columns=["ts"], dtype=str)  # Create data frame with timestamp
        fake_df["ts"] = pd.to_datetime(fake_df["ts"])  # Ensure timestamp are datetime type
        fake_df["WH_P"] = whp_list  # Add fake whp data

        slug_idx = pd.Series(whp_list)[whp_list > 12.90].index.tolist()  # Create list of slug peaks

        test_class.sub_df_dict = {
            1: fake_df
        }  # Override sub_df_dict attribute with fake data frame

        first = test_class.slug_check(slug_idx, 1)  # Get results from slug_check method

        assert first, "First slug index list should not be empty"
        assert len(first) == 2, "First slug index list should only contain two value in this example"
        assert first[1] == 1305, "In this example, the second first slug of the data set occurs at minutes = 1305"

    def test_label_slugs(self, spark_data):
        """
        Unit test for label_slugs method

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard Data Engineering steps
        test_class = Slug_Detection(spark_data)
        test_class.timeframe(start="18-SEP-16 01:09", end="30-SEP-16 09:09")  # example interval
        test_class.data_range()
        test_class.clean_choke(method="99")
        sd_df = test_class.df_toPandas()

        try:
            f, s = test_class.label_slugs()
            print("Sub df dict attribute has not been created")
            raise ValueError
        except AssertionError:
            pass

        test_class.sub_data()  # Create sub df dict

        # create fake data set
        datetime_format = '%d-%b-%y %H:%M'
        base = datetime.strptime("01-JAN-16 09:09", datetime_format)
        date_list = [[base + timedelta(minutes=x)] for x in range(1000)]  # Creat time, one minute appart

        x = np.linspace(0, 100 * np.pi, 1000)
        whp_list = (np.sin(x) * 3) + 10  # create sin wave

        fake_df = pd.DataFrame(data=date_list, columns=["ts"], dtype=str)
        fake_df["ts"] = pd.to_datetime(fake_df["ts"])
        fake_df["WH_P"] = whp_list

        # overide
        test_class.sub_df_dict = {
            1: fake_df,
            2: pd.DataFrame(data=[[0, 0], [0, 0]], columns=["ts", "WH_P"])
        }

        # This should create
        f, s = test_class.label_slugs()
        assert s, "Assert slug index list is not empty"
        assert f, "Assert first slug index list not empty"
        assert len(s[0]) == 49, "In this example, there should be 50 slug peaks"
        assert len(s) == 2, "In this example, there should be one list of slug peaks"

    def test_format_data(self, spark_data):
        """
        Unit test for format_data method

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard Data Engineering steps
        test_class = Slug_Detection(spark_data)
        test_class.timeframe(start="18-SEP-16 01:09", end="18-SEP-16 09:09")  # example interval
        test_class.data_range()
        test_class.clean_choke(method="99")
        sd_df = test_class.df_toPandas()

        try:
            f, s = test_class.label_slugs()
            print("Sub df dict attribute has not been created")
            raise ValueError
        except AssertionError:
            pass

        test_class.sub_data()  # Create sub df dict

        ## Example 1
        ##
        # create fake data set
        datetime_format = '%d-%b-%y %H:%M'  # datetime date format
        base = datetime.strptime("01-JAN-16 09:09", datetime_format)  # Create datetime type timestamp
        date_list = [[base + timedelta(minutes=x)] for x in range(2600)]  # Create list of timestamps

        x = np.linspace(0, 100 * np.pi, 1000)  # Get evenly spaced x array
        whp_list = np.array([10 for i in range(300)])  # Create whp list with normal flow behaviour
        whp_list = np.append(whp_list, (np.sin(x) * 3) + 10)  # Add sin wave array (slug-like)
        whp_list = np.append(whp_list, [10 for i in range(300)])  # Add flat flow to simulate normal flow
        whp_list = np.append(whp_list, (np.sin(x) * 3) + 10)  # Add more slugs

        fake_df = pd.DataFrame(data=date_list, columns=["ts"], dtype=str)  # Create data frame with timestamp
        fake_df["ts"] = pd.to_datetime(fake_df["ts"])  # Ensure timestamp are datetime type
        fake_df["WH_P"] = whp_list  # Add fake whp data

        test_class.sub_df_dict = {
            1: fake_df,
            2: pd.DataFrame(data=[[0, 0], [0, 0]], columns=["ts", "WH_P"])
        }

        f, s = test_class.label_slugs()  # Get first slugs indices list
        test_class.format_data(f)  # Format data

        assert len(test_class.df_list) == 2, \
            "For this example, only two first slug with sufficient amount of time prior"
        assert len(test_class.df_list[0]) == 240, "Created list should be 240"
        assert len(test_class.df_list[1]) == 240, "Created list should be 240"

        ## Example 2
        ##
        # Create fake data frame
        datetime_format = '%d-%b-%y %H:%M'  # datetime date format
        base = datetime.strptime("01-JAN-16 09:09", datetime_format)  # Create datetime type timestamp
        date_list = [[base + timedelta(minutes=x)] for x in range(2300)]  # Create list of timestamps

        x = np.linspace(0, 100 * np.pi, 1000)  # Get evenly spaced x array
        whp_list = (np.sin(x) * 3) + 10  # Create sin wave array (slug-like)
        whp_list = np.append(whp_list, [10 for i in range(300)])  # Add flat flow to simulate normal flow
        whp_list = np.append(whp_list, (np.sin(x) * 3) + 10)  # Add more slugs

        fake_df = pd.DataFrame(data=date_list, columns=["ts"], dtype=str)  # Create data frame with timestamp
        fake_df["ts"] = pd.to_datetime(fake_df["ts"])  # Ensure timestamp are datetime type
        fake_df["WH_P"] = whp_list  # Add fake whp data

        # Override sub_df_dict values
        test_class.sub_df_dict = {
            1: fake_df,
            2: pd.DataFrame(data=[[0, 0], [0, 0]], columns=["ts", "WH_P"])
        }

        f, s = test_class.label_slugs()  # Get first slugs indices list
        test_class.format_data(f)  # Format data

        assert len(test_class.df_list) == 1, \
            "For this example, only one first slug with sufficient amount of time prior"

        ## Example 3
        ##
        # Create fake data frame
        datetime_format = '%d-%b-%y %H:%M'  # datetime date format
        base = datetime.strptime("01-JAN-16 09:09", datetime_format)  # Create datetime type timestamp
        date_list = [[base + timedelta(minutes=x)] for x in range(600)]  # Create list of timestamps

        whp_list = [10 for i in range(600)]  # Normal flow

        fake_df = pd.DataFrame(data=date_list, columns=["ts"], dtype=str)  # Create data frame with timestamp
        fake_df["ts"] = pd.to_datetime(fake_df["ts"])  # Ensure timestamp are datetime type
        fake_df["WH_P"] = whp_list  # Add fake whp data

        # Override sub_df_dict
        test_class.sub_df_dict = {
            1: fake_df,
            2: pd.DataFrame(data=[[0, 0], [0, 0]], columns=["ts", "WH_P"])
        }

        f, s = test_class.label_slugs()  # Get first slugs indices list
        test_class.format_data(f)  # Format data
        assert len(test_class.df_list) == 3, "Based on the example, df_list should equal to 3"

        test_class.format_data(f, size_list=300)  # Format data
        assert len(test_class.df_list) == 2, "Based on the example, df_list should equal to 2"

        ## Example 4
        ##
        # Override sub_df_dict
        test_class.sub_df_dict = {
            1: fake_df,
            2: fake_df
        }

        f, s = test_class.label_slugs()
        test_class.format_data(f, size_list=30, max_clean_count=15)
        assert len(test_class.df_list) == 17, "Based on the example, df_list should equal to 17"

    def test_feature_vector(self, spark_data):
        """
        Unit test for feature_vector method

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard Data Engineering steps
        test_class = Slug_Detection(spark_data)
        test_class.timeframe(start="18-SEP-16 01:09", end="18-SEP-16 09:09")  # large data sample
        test_class.data_range()
        test_class.clean_choke(method="99")
        sd_df = test_class.df_toPandas()
        test_class.sub_data()
        f, s = test_class.label_slugs()
        test_class.format_data(f)

        test_class.feature_vector() # Run method

        assert len(test_class.df_list) == len(test_class.X), "Same number of feature vectors as data frames"
        assert len(test_class.X.columns) == 152, "For this example, number of features should be 152"

        try:
            test_class.feature_vector(percentage_significance=10)
            print("percentage_significance must be a decimal/percentage")
            raise ValueError
        except AssertionError:
            pass

        try:
            test_class.feature_vector(time_predict=1000)
            print("time to predict before must be shorter than size list")
            raise ValueError
        except AssertionError:
            pass

    def test_data_prep(self, spark_data):
        """
        Unit test for data_prep method

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard Data Engineering steps up to data_prep method
        test_class = Slug_Detection(spark_data)
        test_class.timeframe(start="18-SEP-16 01:09", end="18-SEP-16 09:19")  # example interval
        test_class.data_range()
        test_class.clean_choke(method="99")
        sd_df = test_class.df_toPandas()

        test_class.data_prep()

        # enough to just test for X, if this has been created, then all other attributes have to
        assert hasattr(test_class, "X"), "Sub_df_dict has been created "
        assert len(test_class.df_list) == len(test_class.X), "Same number of feature vectors as data frames"
        assert len(test_class.X.columns) == 152, "For this example, number of features should be 152"

    def test_split_data(self, spark_data):
        """
        Unit test for split_data method

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard Data Engineering steps up to split_data method
        test_class = Slug_Detection(spark_data)
        test_class.timeframe(start="18-SEP-16 01:09", end="18-OCT-16 09:10")  # example interval
        test_class.data_range()
        test_class.clean_choke(method="99")
        sd_df = test_class.df_toPandas()
        test_class.data_prep()

        test_class.split_data()

        # assert variables exist
        assert hasattr(test_class, 'X_train'), "X_train attribute data frame should have been created"
        assert hasattr(test_class, 'X_test'), "X_test attribute data frame should have been created"
        assert hasattr(test_class, 'y_train'), "y_train attribute data frame should have been created"
        assert hasattr(test_class, 'y_test'), "y_test attribute data frame should have been created"

        # assert not empty
        assert not test_class.X_train.empty, "X_train attribute must not be empty"
        assert not test_class.X_test.empty, "X_test attribute must not be empty"
        assert test_class.y_train.size != 0, "y_train attribute must not be empty"
        assert test_class.y_test.size != 0, "y_test attribute must not be empty"

        # assert dimensions
        assert test_class.y_test.ndim == 1, "y_test attribute must be 1-D (pandas series)"
        assert test_class.y_train.ndim == 1, "y_train attribute must be 1-D (pandas series)"
        assert len(test_class.X_test.columns) == len(test_class.X.columns), "X_test attribute must same size as X"
        assert len(test_class.X_train.columns) == len(test_class.X.columns), "X_train attribute must same size as X"

        # Test test_size parameter
        try:
            test_class.split_data(test_size = 6)
            print("test_size must be less than 1")
            raise ValueError
        except AssertionError:
            pass

    def test_RF_train(self, spark_data):
        """
        Unit test for RF_train method

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard Data Engineering steps up to RF_train method
        test_class = Slug_Detection(spark_data)
        test_class.timeframe(start="18-SEP-16 01:09", end="18-OCT-16 09:10")  # example interval
        test_class.data_range()
        test_class.clean_choke(method="99")
        sd_df = test_class.df_toPandas()
        test_class.data_prep()
        test_class.split_data()

        # Test method
        pred_features = test_class.RF_train()

        assert hasattr(test_class, 'rf')  # check if correct format
        assert len(pred_features) == len(test_class.X.columns), \
            "There must be as many features as in the original X attribute"

    def test_feature_selection(self, spark_data):
        """
        Unit test for feature_selection method

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard Data Engineering steps
        test_class = Slug_Detection(spark_data)
        test_class.timeframe(start="30-SEP-16 01:09", end="18-OCT-16 09:09")  # example interval
        test_class.data_range()
        test_class.clean_choke(method="99")
        sd_df = test_class.df_toPandas()
        test_class.data_prep()  # need to create X

        # Create example list of features scores
        feature_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 1, 1, 1, 0.1, 0.1]
        feature_scores.extend((152 - len(feature_scores)) * [0])
        # equivalent feature names
        features_names = ['mean_WH_P_0', 'std_WH_P_0', 'mean_DH_P_0', 'std_DH_P_0', 'mean_WH_T_0', 'std_WH_T_0',
                          'mean_DH_T_0', 'mean_WH_P_1', 'std_WH_P_1', 'mean_DH_P_1', 'std_DH_P_1', 'mean_WH_T_1',
                          'std_WH_T_1', 'mean_DH_T_1']

        assert len(test_class.feature_selection(feature_scores)) == 15, "It must be top_n sized, here 15"
        assert test_class.feature_selection(feature_scores) == ['std_WH_P_1', 'mean_WH_P_1', 'std_DH_T_0',
                                                                'mean_DH_T_0', 'std_WH_T_0', 'mean_WH_T_0',
                                                                'std_DH_P_0', 'mean_DH_P_0', 'std_WH_P_0',
                                                                'mean_WH_P_0', 'mean_DH_P_1', 'std_DH_P_1',
                                                                'mean_WH_T_1', 'std_WH_T_1', 'mean_DH_T_1'], \
            "In this example, the following list og feature names is expected"

        assert test_class.feature_selection(feature_scores, top_n=3) == ['std_WH_P_1', 'mean_WH_P_1', 'std_DH_T_0'], \
            "In this example, the following list og feature names is expected"

    def test_RF_predict(self, spark_data):
        """
        Unit test for RF_predict method

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard Data Engineering steps
        test_class = Slug_Detection(spark_data)
        test_class.timeframe(start="18-SEP-16 01:09", end="18-OCT-16 09:09")  # example interval
        test_class.data_range()
        test_class.clean_choke(method="99")
        sd_df = test_class.df_toPandas()
        test_class.data_prep()
        test_class.RF_train()

        p, s, cm = test_class.RF_predict()

        assert len(p) == len(test_class.y_test), "Prediction list must be same size as y_test attribute"

        assert len(test_class.RF_predict(true_label=True)) == 3, "In this example, three objects must be returned"
        assert len(test_class.RF_predict(true_label=False)) == len(p), "In this example, only predictions are returned"

    def test_LogReg_train(self, spark_data):
        """
        Unit test for LogReg_train method

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard Data Engineering steps
        test_class = Slug_Detection(spark_data)
        test_class.timeframe(start="18-SEP-16 01:09", end="18-OCT-16 09:09")  # example interval
        test_class.data_range()
        test_class.clean_choke(method="99")
        sd_df = test_class.df_toPandas()
        test_class.data_prep()

        test_class.LogReg_train()
        assert hasattr(test_class, 'log'), "log attribute must have been created"

        pred_features = test_class.RF_train()
        top_features = test_class.feature_selection(pred_features)
        test_class.LogReg_train(top_features=top_features)
        assert hasattr(test_class, 'logreg_features'), "For this example, logreg_features must have been created"

    def test_LogReg_pred(self, spark_data):
        """
        Unit test for LogReg_pred method

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard Data Engineering steps
        test_class = Slug_Detection(spark_data)
        test_class.timeframe(start="18-SEP-16 01:09", end="18-OCT-16 09:09")  # example interval
        test_class.data_range()
        test_class.clean_choke(method="99")
        sd_df = test_class.df_toPandas()
        test_class.data_prep()

        test_class.LogReg_train()
        pred, prob, s, cm = test_class.LogReg_pred()
        assert len(pred) == len(test_class.y_test), "Prediction list must be same size as y_test attribute"

        assert len(test_class.LogReg_pred(true_label=True)) == 4, "In this example, four objects must be returned"
        assert len(test_class.LogReg_pred(true_label=False)) == 2, "In this example, two objects must be returned"

        pred_features = test_class.RF_train()
        top_features = test_class.feature_selection(pred_features)
        test_class.LogReg_train(top_features=top_features)
        pred, prob, s, cm = test_class.LogReg_pred()

        assert len(test_class.X_test.columns) == len(top_features), "Top features selection must have been performed"
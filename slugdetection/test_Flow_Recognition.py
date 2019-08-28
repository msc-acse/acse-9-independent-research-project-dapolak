
# -*- coding: utf-8 -*-
"""
Part of slugdetection package

@author: Deirdree A Polak
github: dapolak
"""

import unittest
import numpy as np
import pandas as pd

from slugdetection.Flow_Recognition import Flow_Recognition

class Test_Flow_Recognition(unittest.TestCase):
    """
    Unit tests class for Flow Recognition class
    """

    def test_create_class(self, spark_data):
        """
        Unit test for class creation
        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        test_class = Flow_Recognition(spark_data)
        assert hasattr(test_class, "well_df"), "Attribute Spark data frame must exist"
        assert len(test_class.well_df.head(1)) != 0, "Attribute Spark data frame cannot be empty"
        # Pyspark has no clear empty attribute

    def test_label_slugs(self, spark_data):
        """
        Unit test for label_slugs method

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard Flow Recognition Data Engineering
        test_class = Flow_Recognition(spark_data)
        test_class.timeframe(start="08-DEC-14 01:09",
                             end="09-DEC-14 12:09")  # known interval displaying normal flow, slugs and choked flow
        test_class.data_range(verbose=False)
        fr_df = test_class.df_toPandas()

        # Test method
        test_class.label_slugs()

        assert "label" in test_class.pd_df.columns, \
            "New column 'label' should be created in the attribute pandas data frame"
        assert test_class.pd_df["label"].nunique() <= 5, \
            "As per labelling defintion, there must be five or less labels"

    def test_window_label(self, spark_data):
        """
        Unit test for window_label method

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Instantiate Flow Recognition class
        test_class = Flow_Recognition(spark_data)

        ## Examples lists and expected results
        #  first slug example
        list_label = ["normal", "normal", "normal", "normal", "normal", "normal", "normal", "normal", "normal",
                      "normal", "first_slug"]
        assert test_class.window_label(list_label) == 0, "Test list should output 0"

        # first_slug example
        list_label = ["pre_slug", "normal", "normal", "normal", "normal", "normal", "normal", "normal", "normal",
                      "normal", "first_slug"]
        assert test_class.window_label(list_label) == 0, "Test list should output 0"

        # pre-slug example
        list_label = ["pre_slug", "normal", "normal", "normal", "normal", "normal", "normal", "normal", "normal",
                      "normal", "pre_slug"]
        assert test_class.window_label(list_label) == 2, "Test list should output 2"

        # slug flow example
        list_label = ["slug", "slug", "normal", "normal", "normal", "normal", "normal", "normal", "slug",
                      "normal", "slug"]
        assert test_class.window_label(list_label) == 1, "Test list should output 1"

        # normal example
        list_label = ["normal", "normal", "normal", "normal", "normal", "normal", "normal", "normal", "normal",
                      "normal", "normal"]
        assert test_class.window_label(list_label) == 3, "Test list should output 3"

        # ignore example
        list_label = ["ignore", "ignore", "normal", "ignore", "ignore", "ignore", "ignore", "ignore", "normal",
                      "normal", "normal"]
        assert test_class.window_label(list_label) == 4, "Test list should output 4"

        # ignore example
        list_label = ["normal", "normal", "normal"]
        assert test_class.window_label(list_label) == 4, "Test list should output 4"

    def test_feature_vector(self, spark_data):
        """
        Unit test for feature_vector method
        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard Flow Recognition Data Engineering
        test_class = Flow_Recognition(spark_data)
        test_class.timeframe(start="08-DEC-14 01:09",
                             end="09-DEC-14 12:09")  # known interval displaying normal flow, slugs and choked flow
        test_class.data_range(verbose=False)
        fr_df = test_class.df_toPandas()
        test_class.label_slugs()

        # Test method
        window_size = 20
        step = 5
        test_class.feature_vector(window_size=window_size, step=step)

        ### Test whether new column headers are created
        new_features = []  # Create new features column headers
        for i in range(1, window_size + 1):
            for feature_ in test_class.pd_df.columns:
                if (feature_ != 'ts') & (feature_ != "label"):
                    new_features.append(feature_ + "_" + str(i))  # Get all column headers as expected in feature_vec

        for f in new_features:
            assert f in test_class.feature_vec.columns, f + "must be created"

        ### Test only step-th rows are included
        step = 10
        test_class.feature_vector(window_size=40, step=step)

        for idx in test_class.feature_vec.index[1:]:  # do not include index 0 in test
            assert idx % step == 0, "Only step-th rows must be included"

        ### Test window label has been added correctly
        assert "window_label" in test_class.feature_vec.columns, "New column 'window_label' should have been created"
        label_list = test_class.pd_df["label"][:20].values.tolist()
        assert np.array_equal(test_class.feature_vec["window_label"].iloc[0], test_class.window_label(label_list))

    def test_split_data(self, spark_data):
        """
        Unit test for split_data method
        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard Flow Recognition Data Engineering
        test_class = Flow_Recognition(spark_data)
        test_class.timeframe(start="08-DEC-14 01:09",
                             end="09-DEC-14 12:09")  # example interval
        test_class.data_range(verbose=False)
        fr_df = test_class.df_toPandas()
        test_class.label_slugs()
        test_class.feature_vector(window_size=20, step=5)

        # Test method
        test_class.split_data()

        # assert variables exist
        assert hasattr(test_class, 'X_train'), "X_train attribute data frame should have been created"
        assert hasattr(test_class, 'X_test'), "X_test attribute data frame should have been created"
        assert hasattr(test_class, 'y_train'), "y_train attribute data frame should have been created"
        assert hasattr(test_class, 'y_test'), "y_test attribute data frame should have been created"

        # assert dimensions
        assert isinstance(test_class.y_test, pd.Series), "y_test attribute must be 1-D (pandas series)"
        assert isinstance(test_class.y_train, pd.Series), "y_train attribute must be 1-D (pandas series)"
        assert len(test_class.X_test.columns) == (len(test_class.feature_vec.columns) - 1), \
            "X_test attribute must same size as feature vector"
        assert len(test_class.X_train.columns) == (len(test_class.feature_vec.columns) - 1), \
            "X_train attribute must same size as feature vector"

        # assert changing test_size works
        test_class.split_data(test_size=0.5)
        assert len(test_class.y_test) == len(test_class.y_train), \
            "In this example, y_test and y_train must be the same length"
        assert len(test_class.X_test) == len(test_class.X_train), \
            "In this example, X_test and X_train must be the same length"

    def test_LogReg_train(self, spark_data):
        """
        Unit test for LogReg_train method
        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard Flow Recognition Data Engineering
        test_class = Flow_Recognition(spark_data)
        test_class.timeframe(start="08-DEC-14 01:09",
                             end="15-DEC-14 12:09")   # example interval
        test_class.data_range(verbose=False)
        fr_df = test_class.df_toPandas()
        test_class.label_slugs()
        test_class.feature_vector(window_size=20, step=5)
        # Test method
        test_class.LogReg_train()

        assert hasattr(test_class, 'log'), 'Logistics Regression model attribute must have been created'


    def test_LogReg_pred(self, spark_data):
        """
        Unit test for LogReg_pred method
        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard Flow Recognition Data Engineering
        test_class = Flow_Recognition(spark_data)
        test_class.timeframe(start="08-DEC-14 01:09",
                             end="30-DEC-14 12:09")  # example interval
        test_class.data_range(verbose=False)
        fr_df = test_class.df_toPandas()
        test_class.label_slugs()
        test_class.feature_vector(window_size=20, step=5)
        test_class.LogReg_train()

        # Test method
        log_pred, proba, score, cf = test_class.LogReg_pred()

        # test dimensions
        assert len(log_pred) == len(test_class.X_test), 'Number of predictions must be the same as number of points'
        assert len(proba) == len(test_class.y_test),  'Number of predictions must be the same as number of points'

        ## if True is unknown (False) only return two variables
        assert len(test_class.LogReg_pred(true_label=False)) == 2, 'In this example, two variables must be returned'
        assert len(test_class.LogReg_pred(true_label=True)) == 4, 'In this example, four variables must be returned'

    def test_SVM_train(self, spark_data):
        """
        Unit test for SVM_train method
        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard Flow Recognition Data Engineering
        test_class = Flow_Recognition(spark_data)
        test_class.timeframe(start="08-DEC-14 01:09",
                             end="30-DEC-14 12:09")  # example interval
        test_class.data_range(verbose=False)
        fr_df = test_class.df_toPandas()
        test_class.label_slugs()
        test_class.feature_vector(window_size=20, step=5)

        # Test model
        test_class.SVM_train()

        assert hasattr(test_class, 'svm'), "SVM model attribute has been created"

    def test_SVM_pred(self, spark_data):
        """
        Unit test for SVM_pred method
        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard Flow Recognition Data Engineering
        test_class = Flow_Recognition(spark_data)
        test_class.timeframe(start="08-DEC-14 01:09",
                             end="30-DEC-14 12:09")  # example interval
        test_class.data_range(verbose=False)
        fr_df = test_class.df_toPandas()
        test_class.label_slug()
        test_class.feature_vector(window_size=20, step=5)
        test_class.SVM_train()

        # Test method
        svm_pred, score, cf = test_class.SVM_pred()

        # check dimensions
        assert len(svm_pred) == len(test_class.y_test), 'Number of predictions must be the same as number of points'

        # If true_label is unknown (False) only return one variables
        assert len(test_class.SVM_pred(true_label=False)) == len(svm_pred), \
            'In this example, one variable must be returned, which is svm_pred'
        assert len(test_class.SVM_pred(true_label=True)) == 3, 'In this example, three variables must be returned'

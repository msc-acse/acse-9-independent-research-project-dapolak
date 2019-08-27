# -*- coding: utf-8 -*-
"""
Part of slugdetection package

@author: Deirdree A Polak
github: dapolak
"""
import unittest
import numpy as np

from slugdetection.Slug_Labelling import Slug_Labelling


class Test_Slug_Labelling(unittest.TestCase):
    """
    Unitest class for the Slug Labelling class
    """

    def test_create_class(self, spark_data):
        """
        Tests that the class object gets created properly

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        test_class = Slug_Labelling(spark_data)
        assert hasattr(test_class, "well_df"), "Well data frame must be created"
        assert len(test_class.well_df.head(1)) != 0, "well attribute not empty"  # Pyspark has no clear empty attribute

    def test_feature_vector(self, spark_data):
        """
        Unit test for the feature_vector method

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard slug labelling data engineering
        test_class = Slug_Labelling(spark_data)
        test_class.timeframe(start="01-JAN-14 09:09", end="02-JAN-14 09:09")  # Small interval, for efficiency purposes
        test_class.data_range()
        test_class.df_toPandas()

        window_size = 5
        test_class.feature_vector(window_size=window_size, step=1)

        ### Test new column headers are created
        new_features = []  # Create new features column headers
        for i in range(1, window_size + 1):
            for feature_ in test_class.pd_df.columns:
                if feature_ != 'ts':
                    new_features.append(feature_ + "_" + str(i))

        for f in new_features:
            assert f in test_class.feature_vec.columns, f + "must be created"

        ### Test only step-th rows are included
        step = 10
        test_class.feature_vector(window_size=40, step=step)

        for idx in test_class.feature_vec.index[1:]:  # do not include index 0 in test
            assert idx % step == 0, "Only step-th rows must be included"

        ### Test keep ts variable
        test_class.feature_vector(window_size=10, step=5, keep_ts=False)
        assert 'ts' not in test_class.feature_vec.columns, "Timestamp column must not be in feature vector"

        test_class.feature_vector(window_size=40, step=step, keep_ts=True)
        assert 'ts' in test_class.feature_vec.columns, "Timestamp column must be included in feature vector"

    def test_Kmean_classification(self, spark_data):
        """
        Unit test for the Kmean_classification method

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard slug labelling data engineering up to KMeans_classification method
        test_class = Slug_Labelling(spark_data)
        test_class.timeframe(start="01-JAN-14 09:09", end="02-JAN-14 09:09")
        test_class.data_range()
        test_class.df_toPandas()
        test_class.feature_vector(window_size=10, step=5)

        ### Test five clusters can be created
        test_class.Kmean_classification(n_labels=5)
        assert len(test_class.labels) == len(test_class.feature_vec), "There must be as many labels as feature vectors"
        assert len(np.unique(test_class.labels)) <= 5, "There should be five or less labels present"

        ### Test two clusters can be created
        test_class.Kmean_classification(n_labels=2)
        assert len(test_class.labels) == len(test_class.feature_vec), "There must be as many labels as feature vectors"
        assert len(np.unique(test_class.labels)) == 2, "There should be two or less labels present"

    def test_get_labels(self, spark_data):
        """
        Unit test for the get_labels method

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard slug labelling data engineering up to KMeans_classification method
        test_class = Slug_Labelling(spark_data)
        test_class.timeframe(start="01-JAN-14 09:09", end="02-JAN-14 09:09")
        test_class.data_range()
        test_class.df_toPandas()
        test_class.feature_vector(window_size=10, step=5)
        test_class.Kmean_classification(n_labels=2)

        assert len(test_class.get_labels()) == len(test_class.labels)
        assert all(test_class.get_labels() == test_class.labels)

    def test_label_plot(self, spark_data):
        """
        Unit test for the label_plot method

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard slug labelling data engineering up to label_plot method
        test_class = Slug_Labelling(spark_data)
        test_class.timeframe(start="01-JAN-14 09:09", end="02-JAN-14 09:09")
        test_class.data_range()
        test_class.df_toPandas()
        test_class.feature_vector(window_size=10, step=5)
        test_class.Kmean_classification(n_labels=2)

        try:
            test_class.label_plot(["WH_P", "DH_P"])  # need 3 variables
            print("label_plot method requires three variables")
            raise ValueError
        except AssertionError:
            pass

        try:
            test_class.label_plot(["WH_P", "random_variable", "DH_P"])  # variables need to exist in features
            print("label_plot method requires all three variables to exist in class attribute features")
            raise ValueError
        except AssertionError:
            pass

        try:
            test_class.labels = [1, 2, 3, 1, 3, 1, 4]
            test_class.label_plot(["WH_P", "WH_T", "DH_P"])  # labels list needs to be the same size as feature vec list
            print("label_plot method requires the label list to be the same size as the feature vector attribute")
            raise ValueError
        except AssertionError:
            pass

    def test_unpack_feature_vector(self, spark_data):
        """
        Unit test for the label_plot method

        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard slug labelling data engineering up to unpack_feature_vector method
        test_class = Slug_Labelling(spark_data)
        test_class.timeframe(start="01-JAN-14 09:09", end="02-JAN-14 09:09")  # one day worth of data
        test_class.data_range()
        test_class.df_toPandas()
        test_class.feature_vector(window_size=10, step=5)
        test_class.Kmean_classification(n_labels=2)

        unpacked_list = test_class.unpack_feature_vector

        assert len(unpacked_list) == 2, "There are two clusters, there must be two data frames in unpack list"
        for var in test_class.features:
            assert var in unpacked_list[0].columns, var + "must be a column header of the unpacked vectors data frames"

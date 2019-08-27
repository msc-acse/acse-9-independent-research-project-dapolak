# -*- coding: utf-8 -*-
"""
Part of slugdetection package

@author: Deirdree A Polak
github: dapolak
"""

import unittest
import numpy as np
from pyspark.sql import functions as F

import slugdetection.Data_Engineering as Data_Engineering


class Test_Data_Engineering(unittest.TestCase):
    """
    Unit tests class for Data Engineering class
    """

    def test_create_class(self):
        """
        Unit test for class creation
        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        test_class = Data_Engineering(spark_data)  # Create class
        assert len(test_class.well_df.head(1)) != 0, "Assert data frame is not empty"
        # Pyspark has no clear empty attribute

    def test_reset_well_df(self, spark_data):
        """
        Unit test for reset_well_df method
        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        test_class = Data_Engineering(spark_data)

        # crop data between start and end date
        test_class.timeframe(start="01-JAN-16 09:09", end="01-JAN-16 09:18")
        a = test_class.well_df.count()  # count rows

        # reset
        test_class.reset_well_df()
        b = test_class.well_df.count()  # count rows

        # Test that rows have been resetted
        assert a <= b, "In this example, the number of rows should increase after resetting"

    def test_timeframe(self, spark_data):
        """
        Unit test for timeframe method
        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        test_class = Data_Engineering(spark_data)

        # Test that data gets cropped
        test_class.timeframe(start="01-JAN-16 09:09", end="01-JAN-16 09:18")
        assert test_class.well_df.count() == 9, "In this example, there is only 9 data points between the two dates"

        test_class.reset_well_df()  # Reset data frame for next test

        # Test that end date must be larger than start date
        try:
            test_class.timeframe(start="01-JAN-16 09:09", end="01-JAN-12 09:18")
            print("Cannot have end date is smaller/older than start date")
            raise ValueError
        except AssertionError:
            pass

    def test_set_thresholds(self, spark_data):
        """
        Unit test for set_thresholds method
        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        test_class = Data_Engineering(spark_data)

        # Test example thresholds
        test_class.set_thresholds("WH_P", 1000, -200)
        assert test_class.thresholds["WH_P"] == [-200, 1000], "In this example, [-200, 1000] are expected"

        test_class.set_thresholds("random", 10, -10)
        assert test_class.thresholds["random"] == [-10, 10], "In this example, [-10, 10] are expected"

        # Test max is larger than min
        try:
            test_class.set_thresholds("WH_P", -10000, -10)
            print("Maximum value must be higher than minimum")
            raise ValueError
        except AssertionError:
            pass

        # Test max must be a number
        try:
            test_class.set_thresholds("WH_P", "string", -10)
            print("Max and min must be numbers")
            raise ValueError
        except AssertionError:
            pass

        # Test min must be a number
        try:
            test_class.set_thresholds("WH_P", 90, "string")
            print("Max and min must be numbers")
            raise ValueError
        except AssertionError:
            pass

    def test_data_range(self, spark_data):
        """
        Unit test for set_thresholds method
        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        test_class = Data_Engineering(spark_data)

        test_class.timeframe(start="01-JAN-16 09:09", end="01-MAR-16 09:18")  # known interval with out of range data

        test_class.data_range()  # apply data range function

        # Lambda function to count number of time condition occurs
        cnt_cond = lambda cond: F.sum(F.when(cond, 1).otherwise(0))

        # look at the values from each column
        for f in test_class.features:
            counts = test_class.well_df.agg(
                cnt_cond(F.col(f) <= test_class.thresholds[f][0]))
            print(f, "lower", counts.collect()[0][0])
            # assert counts.collect()[0][0] == 0, "No out of range values for " + str(f)

            counts = test_class.well_df.agg(
                cnt_cond(F.col(f) >= test_class.thresholds[f][1]))
            print(f, "upper", counts.collect()[0][0])
            # assert counts.collect()[0][0] == 0, "No out of range values for " + str(f)

    def test_clean_choke(self, spark_data):
        """
        Unit test for clean_choke method
        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        test_class = Data_Engineering(spark_data)

        test_class.timeframe(start="01-FEB-14 09:09", end="01-APR-14 09:18")  # known interval with choke
        a = test_class.well_df.count()  # count rows

        test_class.clean_choke(method="99")
        b = test_class.well_df.count()  # count rows

        assert a > b, "In this example, the number of rows decreases"

        test_class.reset_well_df()  # reset for new tests

        # Lambda function to count number of time condition occurs
        cnt_cond = lambda cond: F.sum(F.when(cond, 1).otherwise(0))

        # Count number of times there is a None value in each column
        counts_pre = test_class.well_df.agg(
            cnt_cond(F.col('WH_choke') == None).alias('wh_choke_cnt'),
            cnt_cond(F.col('WH_P') == None).alias('whp_cnt'),
            cnt_cond(F.col('DH_P') == None).alias('dhp_cnt'),
            cnt_cond(F.col('WH_T') == None).alias('wht_cnt'),
            cnt_cond(F.col('DH_T') == None).alias('dht_cnt')
        )

        # Apply clean choke methdd
        test_class.clean_choke(method="no_choke")

        # Re-count number of times there is a None value in each column
        counts_post = test_class.well_df.agg(
            cnt_cond(F.col('WH_choke') == None).alias('wh_choke_cnt'),
            cnt_cond(F.col('WH_P') == None).alias('whp_cnt'),
            cnt_cond(F.col('DH_P') == None).alias('dhp_cnt'),
            cnt_cond(F.col('WH_T') == None).alias('wht_cnt'),
            cnt_cond(F.col('DH_T') == None).alias('dht_cnt')
        )

        # Check for all variables that the number of None values has increased
        assert counts_pre.collect()[0][0] <= counts_post.collect()[0][0], "In this example, the number of None " \
                                                                          "values in WH choke column should increase"
        assert counts_pre.collect()[0][1] <= counts_post.collect()[0][1], "The number of None values in WHP column " \
                                                                          "should increase"
        assert counts_pre.collect()[0][2] <= counts_post.collect()[0][2], "The number of None values in DHP column " \
                                                                          "should increase"
        assert counts_pre.collect()[0][3] <= counts_post.collect()[0][3], "The number of None values in WHT column " \
                                                                          "should increase"
        assert counts_pre.collect()[0][4] <= counts_post.collect()[0][4], "The number of None values in DHT column " \
                                                                          "should increase"

    def test_df_toPandas(self, spark_data):
        """
        Unit test for df_toPandas method
        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        test_class = Data_Engineering(spark_data)
        test_class.timeframe(start="01-JAN-14 09:09", end="01-JAN-14 10:09")
        test_class.df_toPandas()

        assert hasattr(test_class, "pd_df"), "pd_df attribute must have been created"
        assert not test_class.pd_df.empty, "pd_df attribute is not empty"

    def test_standardise(self, spark_data):
        """
        Unit test for df_toPandas method
        Parameters
        ----------
        spark_data : Spark data frame
            well data frame
        """
        # Standard operations for Data Engineering
        test_class = Data_Engineering(spark_data)
        test_class.timeframe(start="01-JAN-14 09:09", end="01-JAN-14 10:09")
        pandas_df = test_class.df_toPandas()

        # Compute standardised data frame
        pandas_df_z = test_class.standardise(pandas_df)

        # Test dimensions
        assert not pandas_df_z.empty, "Standardise data frame not empty"
        assert len(pandas_df_z) == len(pandas_df), "Data frame must be the same size after standardisation"

        # Test mean
        for f in pandas_df_z.columns:
            if f != 'ts':
                assert np.isclose(pandas_df_z[f].mean(), 0, atol=0.01), "Mean of standardised data should be close to 0"

        # Manual standardisation test
        for f in pandas_df_z.columns:
            if f != 'ts':
                # Calculate standardised value at each column
                mean_ = pandas_df[f].mean()
                std_ = pandas_df[f].std()
                pandas_df[f] = (pandas_df[f] - mean_) / std_

                # Randomly check values match
                for i in range(5):
                    val = random.randint(1, len(pandas_df))
                    assert np.isclose(pandas_df[f].loc[val], pandas_df_z[f].loc[val], atol=0.001), \
                        "The two standardise values should be close"


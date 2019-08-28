# -*- coding: utf-8 -*-
"""
Part of slugdetection package

@author: Deirdree A Polak
github: dapolak
"""


import numpy as np
import pandas as pd
import math

import unittest

from slugdetection.Slug_Forecasting import Slug_Forecasting


class Test_Slug_Forecasting(unittest.TestCase):
    """
    Unit tests class for Slug Forecasting class
    """

    def test_create_class(self, whp_pandas):
        """
        Unit test for class creation
        Parameters
        ----------
        whp_pandas : Pandas DataFrame
            whp and ts data frame
        """
        test_class = Slug_Forecasting(whp_pandas.copy()) # Instantiate class

        assert hasattr(test_class, "slug_df"), "slug_df attribute must be created"
        assert isinstance(test_class.slug_df.index, pd.DatetimeIndex), "slug_df has DateTimeIndex"

        whp_pandas_short = whp_pandas[:60].copy() # crop data frame

        # Test that class does not get created if whp_pandas is too short
        try:
            test_class = Slug_Forecasting(whp_pandas_short)
            print("pandas data frame is too short")
            raise ValueError
        except AssertionError:
            pass

        whp_pandas_nowhp = whp_pandas.copy()
        whp_pandas_nowhp = whp_pandas_nowhp.drop("WH_P", axis=1)

        # Test that class does not get created if whp_pandas does not contain WHP column
        try:
            test_class = Slug_Forecasting(whp_pandas_nowhp)
            print("pandas data frame does not contain WH_P column")
            raise ValueError
        except AssertionError:
            pass

        whp_pandas_nots = whp_pandas.copy()
        whp_pandas_nots = whp_pandas_nots.drop("ts", axis=1)

        # Test that class does not get created if whp_pandas does not contain timestamp column
        try:
            test_class = Slug_Forecasting(whp_pandas_nots)
            print("pandas data frame does not contain ts column")
            raise ValueError
        except AssertionError:
            pass

        # Test that other column in whp_pandas get ignored and dropped from slug_df attribute
        whp_pandas_extravar = whp_pandas.copy()
        whp_pandas_extravar["random"] = whp_pandas_extravar["WH_P"]

        test_class = Slug_Forecasting(whp_pandas_extravar.copy())

        assert "random" not in test_class.slug_df.columns, "In this example, random column should have been dropped"

    def test_stationarity_check(self, whp_pandas, not_station_pd):
        """
        Unit test for stationarity_check method
        Parameters
        ----------
        whp_pandas : Pandas DataFrame
            whp and ts data frame
        not_station_pd : Pandas DataFrame
            whp and ts data frame with non stationary data
        """
        test_class = Slug_Forecasting(whp_pandas.copy())  # Instantiate class object
        test_class.stationarity_check()

        assert hasattr(test_class, "station_result"), "Station_result attribute is created"
        assert test_class.station_result[0] < 0.05, "In this example, p-value should be less than 5%"

        test_class.stationarity_check(diff=1)
        assert test_class.station_result[0] <= 0.0, "In this example, p-value should be 0%"

        test_class = Slug_Forecasting(not_station_pd.copy())  # Instantiate new object with non stationary data
        test_class.stationarity_check()
        assert test_class.station_result[0] > 0.05, "In this example, p-value should be more than 5%"

    def test_split_data(self, whp_pandas):
        """
        Unit test for split_data method
        Parameters
        ----------
        whp_pandas : Pandas DataFrame
            whp and ts data frame
        """
        test_class = Slug_Forecasting(whp_pandas.copy())
        test_class.stationarity_check()
        test_class.split_data()

        assert hasattr(test_class, "y_train"), "y_train attribute must have been create"
        assert hasattr(test_class, "y_pred"), "y_test attribute must have been create"

        assert len(test_class.y_train) == 180, "In this example, y_train should be 180 long"
        assert len(test_class.y_pred) == 60, "In this example, y_pred should be 60 long"

        test_class = Slug_Forecasting(whp_pandas.copy())

        # test train size data
        try:
            test_class.split_data(train_size=400)
            print("Not enough data to fulfill train_size requirement")
            raise ValueError
        except AssertionError:
            pass

    def test_ARIMA_model(self, whp_pandas):
        """
        Unit test for ARIMA_model method
        Parameters
        ----------
        whp_pandas : Pandas DataFrame
            whp and ts data frame
        """
        test_class = Slug_Forecasting(whp_pandas.copy())
        test_class.stationarity_check()
        test_class.split_data()

        test_class.ARIMA_model(1, 0, 1, show=False)  # Fit results
        assert hasattr(test_class, "fit_results"), "fit_results attribute must have been created"

    def test_error_metrics(self, whp_pandas):
        """
        Unit test for error_metrics method
        Parameters
        ----------
        whp_pandas : Pandas DataFrame
            whp and ts data frame
        """
        test_class = Slug_Forecasting(whp_pandas.copy())
        test_class.stationarity_check()
        test_class.split_data()
        test_class.ARIMA_model(1, 0, 1, show=False)

        # Test error metrics parameter entry
        assert test_class.error_metrics(error="other", verbose=False) == None, "Nothing is returned"

        # Test values returned for fitting regression
        assert len(test_class.error_metrics(error="fit", verbose=False)) == 4, \
            "Three variables must be returned in this example"
        mape, mse, rmse, r2 = test_class.error_metrics(error="fit", verbose=False)

        # test stats:
        assert r2 < 1.0, "Coefficient of Determination r2 should be less than 1"
        assert np.isclose(math.sqrt(mse), rmse, atol=0.01), \
            "Square rooted mean squared error should equal root mean squared error"

        test_class.ARIMA_pred(pred_time=60, show=False)  # Forecast values

        # Test values returned for forecasting regression
        assert len(test_class.error_metrics(error="pred", verbose=False)) == 4, \
            "Three variables must be returned in this example"
        mape, mse, rmse, r2 = test_class.error_metrics(error="pred", verbose=False)

        # test stats:
        assert r2 < 1.0, "Coefficient of Determination r2 should be less than 1"
        assert np.isclose(math.sqrt(mse), rmse, atol=0.01), \
            "Square rooted mean squared error should equal root mean squared error"

    def test_ARIMA_pred(self, whp_pandas):
        """
        Unit test for error_metrics method
        Parameters
        ----------
        whp_pandas : Pandas DataFrame
            whp and ts data frame
        """
        test_class = Slug_Forecasting(whp_pandas.copy())
        test_class.stationarity_check()
        test_class.split_data()
        test_class.ARIMA_model(1, 0, 1, show=False)
        test_class.ARIMA_pred(pred_time=60)
        assert hasattr(test_class, "forecast"), "Forecast attribute must have been created"
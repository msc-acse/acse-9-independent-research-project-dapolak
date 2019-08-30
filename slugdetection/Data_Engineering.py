# -*- coding: utf-8 -*-
"""
Part of slugdetection package

@author: Deirdree A Polak
github: dapolak
"""

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.window import Window

class Data_Engineering:
    """
    Tools to crop and select the raw well data. Converts data from a Spark dataframe to Pandas.

    Parameters
    ----------
    well : Spark data frame
        data frame containing the pressure, temperature and choke data from a well.

    Attributes
    ----------
    well_df : Spark data frame
        data frame containing all of the pressure, temperature and choke data from a well. None values have
        been dropped
    well_og : Spark data frame
        original data frame copy, with None values
    features : list of strings
        List of the features of the well, default "WH_P", "DH_P", "WH_T", "DH_T" and "WH_choke"
    thresholds : dictionary
        Dictionary with important features as keys, and their lower and upper thresholds as values. This is
        used for cropping out of range values. The set_thresholds method allows user to change or add values.
    """

    def __init__(self, well):
        self.well_df = well.na.drop()
        self.well_og = well
        self.features = ["WH_P", "DH_P", "WH_T", "DH_T", "WH_choke"]
        self.thresholds = {"WH_P": [0, 100],
                           "DH_P": [90, 150],
                           "WH_T": [0, 100],
                           "DH_T": [75, 95],
                           "WH_choke": [-1000, 1000]}

    def stats(self):
        """
        Describes the data in terms of the most common statistics, such as mean, std, max, min and count
        Returns
        -------
        stats : Spark DataFrame
            Stats of data frame attribute well_df
        """
        return self.well_df.describe()

    def shape(self):
        """
        Describes the shape of the Spark data frame well_df, with number of rows and number of columns

        Returns
        -------
        shape : int, int
            number of rows, number of columns
        """
        return self.well_df.count(), len(self.well_df.columns)

    def reset_well_df(self):
        """
        Resets Spark data frame attribute well_df to original state by overriding the well_df attribute
        """
        self.well_df = self.well_og.na.drop()

    def timeframe(self, start="01-JAN-01 00:01", end="01-JUL-19 00:01", date_format="dd-MMM-yy HH:mm",
                  datetime_format='%d-%b-%y %H:%M'):
        """
        For Spark DataFrame well_df attribute, crops the data to the inputted start and end date

        Parameters
        ----------
        start : str (optional)
            Wanted start date of cropped data frame (default is "01-JAN-01 00:01")
        end : str (optional)
            Wanted end date of cropped data frame (default is "01-JAN-19 00:01")
        date_format : str (optional)
            String format of inputted dates (default is "dd-MMM-yy HH:mm")
        datetime_format : str (optional)
            C standard data format for datetime (default is '%d-%b-%y %H:%M')
        """

        d1 = datetime.strptime(start, datetime_format)
        d2 = datetime.strptime(end, datetime_format)

        assert max((d1, d2)) == d2, "Assert end date is later than start date"

        # Crop to start date
        self.well_df = self.well_df.filter(
            F.col("ts") > F.to_timestamp(F.lit(start), format=date_format).cast('timestamp'))

        # Crop to end date
        self.well_df = self.well_df.filter(
            F.col("ts") < F.to_timestamp(F.lit(end), format=date_format).cast('timestamp'))

        return

    def set_thresholds(self, variable, max_, min_):
        """
        Sets the thresholds value of a variable

        Parameters
        ----------
        variable : str
            Name of variable, for example "WH_P"
        max_ : float
            Upper threshold of variable
        min_ : float
            Lower threshold of variable
        """
        assert isinstance(min_, float), "Minimum threshold must be a number"
        assert isinstance(max_, float), "Maximum threshold must be a number"
        assert max(min_, max_) == max_, "Maximum value must be larger than min"
        self.thresholds[variable] = [min_, max_]

    def data_range(self, verbose=True):
        """
        Ensures variables within the dataframe well_df are within range, as set by the attribute thresholds. The out of
        range values are replaced by the previous in range value

        Parameters
        ----------
        verbose : bool (optional)
            whether to allow for verbose (default is True)
        """
        window = Window.orderBy("ts")  # Spark Window ordering data frames by time

        lag_names = []  # Empty list to store column names
        for well_columns in self.well_df.schema.names:  # loop through all components (columns) of data

            if well_columns != "ts":  # no tresholding for timestamp

                if well_columns in self.thresholds.keys():
                    tresh = self.thresholds[well_columns]  # set thresholds values for parameter from dictionary
                else:
                    tresh = [-1000, 1000]  # if feature not in thresholds attribute, set large thresholds

                if verbose:
                    print(well_columns, "treshold is", tresh)

                for i in range(1, 10):  # Naive approach, creating large amount of lagged features columns
                    lag_col = well_columns + "_lag_" + str(i)
                    lag_names.append(lag_col)
                    self.well_df = self.well_df.withColumn(lag_col, F.lag(well_columns, i, 0).over(window))

                for i in range(8, 0, -1):
                    lag_col = well_columns + "_lag_" + str(i)
                    prev_lag = well_columns + "_lag_" + str(i + 1)

                    # apply minimum and maximum threshold to column, and replace out of range values with previous value
                    self.well_df = self.well_df.withColumn(lag_col,
                                                           F.when(F.col(lag_col) < tresh[0],
                                                                  F.col(prev_lag))
                                                           .otherwise(F.col(lag_col)))
                    self.well_df = self.well_df.withColumn(lag_col,
                                                           F.when(F.col(lag_col) > tresh[1],
                                                                  F.col(prev_lag)).otherwise(F.col(lag_col)))

                # apply minimum and maximum threshold to column, and replace out of range values with previous value
                lag_col = well_columns + "_lag_1"
                self.well_df = self.well_df.withColumn(well_columns,
                                                       F.when(F.col(well_columns) < tresh[0],
                                                              F.col(lag_col))
                                                       .otherwise(F.col(well_columns)))
                self.well_df = self.well_df.withColumn(well_columns,
                                                       F.when(F.col(well_columns) > tresh[1],
                                                              F.col(lag_col))
                                                       .otherwise(F.col(well_columns)))

        self.well_df = self.well_df.drop(*lag_names)
        return

    def clean_choke(self, method="99"):
        """
        Method to clean WH_choke variables values from the well_df Spark data frame attribute

        Parameters
        ----------
        method : str (optional)
            Method to clean out WH_choke values. "99" entails suppressing all the data rows where the choke is lower
            than 99%. "no_choke" entails setting to None all the rows where the WH_choke value is 0 or where it is non
            constant i.e. differential is larger than 1 or second differential is larger than 3 (default is '99').
        """

        assert ("WH_choke" in self.well_df.schema.names), 'In order to clean out WH choke data, WH choke column' \
                                                          'in well_df must exist'

        if method == "99":
            self.well_df = self.well_df.where("WH_choke > 99")  # Select well_df only where WH is larger than 99%

        elif method == "no_choke":

            # Select well_df only where WH choke is constant
            window = Window.orderBy("ts")  # Window ordering by time

            # Create differential and second differential columns for WH choke
            self.well_df = self.well_df.withColumn("WH_choke_lag", F.lag("WH_choke", 1, 0).over(window))
            self.well_df = self.well_df.withColumn("WH_choke_diff", F.abs(F.col("WH_choke") - F.col("WH_choke_lag")))
            self.well_df = self.well_df.withColumn("WH_choke_lag2", F.lag("WH_choke_lag", 1, 0).over(window))
            self.well_df = self.well_df.withColumn("WH_choke_diff2", F.abs(F.col("WH_choke") - F.col("WH_choke_lag2")))

            for col in self.well_df.schema.names:
                # Set all rows with WH choke less than 10 to 0
                self.well_df = self.well_df.withColumn(col, F.when(F.col("WH_choke") < 10, None).
                                                       otherwise(F.col(col)))
                # Select well_df where WH choke gradient is less than 1, set rows with high gradient to None
                self.well_df = self.well_df.withColumn(col,
                                                       F.when(F.col("WH_choke_diff") > 1, None).
                                                       otherwise(F.col(col)))
                # Select well_df where WH choke curvature is less than 3, set rows with higher values to None
                self.well_df = self.well_df.withColumn(col,
                                                       F.when(F.col("WH_choke_diff2") > 3, None).
                                                       otherwise(F.col(col)))
        else:
            print("Clean choke method inputted is not know. Try 99 or no_choke")
        return

    def df_toPandas(self, stats=True, **kwargs):
        """
        Creates a copy of Spark data frame attribute well_df in Pandas format. Also calculates and stores the
        mean and standard deviations of each column in the Pandas data frame in the class attributes means and stds.

        Parameters
        ----------
        stats : bool (optional)
            Bool asserting whether or not to calculate means and standard deviations of each columns/variable (default
            is True)
        kwargs :
            features : list of str
                feature names/ column headers to include in pandas data frame pd_df attribute

        Returns
        -------
        pd_df : Pandas data frame
            Pandas data frame of original well_df Spark data frame
        """

        if "features" in kwargs.keys():  # if features specified in kwargs, update feature attribute
            self.features = kwargs["features"]

        cols = self.features.copy()
        cols.append("ts")

        print("Converting Spark data frame to Pandas")
        self.pd_df = self.well_df.select(cols).toPandas()  # convert selected columns of data frame to Pandas
        print("Converted")

        if stats:  # If stats is true, calculate and store mean and std as attributes
            self.means = pd.DataFrame([[0 for i in range(len(self.features))]], columns=self.features)
            self.stds = pd.DataFrame([[0 for i in range(len(self.features))]], columns=self.features)
            for f in self.features:
                self.means[f] = self.pd_df[f].mean()  # Compute and store mean of column in means attribute
                self.stds[f] = self.pd_df[f].std()  # Compute and store std of column in stds attribute

        return self.pd_df

    def standardise(self, df):
        """
        Standardises the data based on the attributes means and stds as calculated when the original dataframe was
        converted to Pandas.

        Parameters
        ----------
        df : Pandas data frame
            Input data frame to be standardised

        Returns
        -------
        df : Pandas data frame
            Input data frame standardised
        """

        for feature_ in self.means.columns:  # For all features
            if (feature_ != 'ts') & (feature_ in df.columns):
                avg = self.means[feature_][0]  # get mean for feature from means attribute
                std = self.stds[feature_][0]  # ger std for feature from stds attribute

                df[feature_] -= avg  # Standardise column
                df[feature_] /= std

        return df

    def plot(self, start=0, end=None, datetime_format="%d-%b-%y %H:%M",
             title="Well Pressure and Temperature over time", ax2_label="Temperature in C // Choke %", **kwargs):
        """
        Simple plot function to plot the pd_df pandas data frame class attribute.

        Parameters
        ----------
        start : int or str (optional)
            Index or date at which to start plotting the data (default is 0)
        end : int or str (optional)
            Index or date at which to stop plotting the data (default is None)
        datetime_format : str (optional)
            C standard data format for datetime (default is '%d-%b-%y %H:%M')
        title : str (optional)
            Plot title (default is "Well Pressure and Temperature over time")
        ax2_label : str (optional)
            Label for second axis, for non pressure features (default is "Temperature in C // Choke %")
        kwargs :
            features: list of str
                List of features to include in the plot

        Returns
        -------
        : Figure
           data plot figure
        """

        assert hasattr(self, "pd_df"), "Pandas data frame pd_df attribute must exist"
        assert not self.pd_df.empty, "Pandas data frame cannot be empty"

        # If features has been specified in kwargs passed
        if "features" in kwargs.keys():  # if only selected features
            self.features = kwargs["features"]

        for f in self.features:  # Check features exist
            assert (f in self.pd_df.columns), f + "must be contained in pd_df"

        if isinstance(start, int):  # If start date inputted as an index
            assert start >= 0, "Start index must be positive"
            assert start <= len(self.pd_df), "Start index must be less than the last index of pd_df attribute"

        if isinstance(end, int):  # If start date inputted as an index
            assert end >= 0, "End index must be positive"

        if isinstance(start, str):  # If a string has been passed for the start date
            date = datetime.strptime(start, datetime_format)
            assert np.any(self.pd_df.isin([date])), "Start time must exist in pandas data frame"
            start = self.pd_df['ts'][self.pd_df['ts'].isin([date])].index.tolist()[0]  # Get start date as in index

        if isinstance(end, str):  # If a string has been passed for the end date
            date = datetime.strptime(end, datetime_format)
            assert np.any(self.pd_df.isin([date])), "End time must exist in pandas data frame"
            end = self.pd_df['ts'][self.pd_df['ts'].isin([date])].index.tolist()[0]  # Get end date as in index

        if end is not None:  # If end index/date has been specified
            assert max((start, end)) == end, "Assert end date is later than start date"

        fig, ax = plt.subplots(1, 1, figsize=(30, 12))  # Create subplot
        ax2 = ax.twinx()  # Instantiate secondary axis that shares the same x-axis

        lines = []  # Create empty list to store lines and corresponding labels
        colours = ['C' + str(i) for i in range(len(self.features))]  # Create list of colour for plots lines

        for col, c in zip(self.features, colours):
            if col[-1] == "P":  # If pressure, plot on main axis
                a, = ax.plot(self.pd_df["ts"][start:end], self.pd_df[col][start:end], str(c) + ".", label=col)
                ax.set_ylabel("Pressure in BarG")
                lines.append(a)
            else:  # For other features, like Temperature and Choke, plot on secondary axis
                a, = ax2.plot(self.pd_df["ts"][start:end], self.pd_df[col][start:end], c + '.', label=col)
                ax2.set_ylabel(ax2_label)
                lines.append(a)
        ax.legend(lines, [l.get_label() for l in lines])
        ax.set_xlabel("Time")
        ax.grid(True, which='both')
        ax.set_title(title)

        return fig


def confusion_mat(cm, labels, title='Confusion Matrix', cmap='RdYlGn', **kwargs):
    """
    Simple confusion matrix plotting method. Inspired by Scikit Learn Confusionp Matrix plot example.

    Parameters
    ----------
    cm : numpy array or list
      Confusion matrix as outputted by Scikit Learn Confusion Matrix method.
    labels : list of str
      Labels to use on the plot of the Confusion Matrix. Must match number of rows in the confusion matrix.
    title : str (optional)
      Title that will be printed above confusion matrix plot
    cmap : str (optional)
      Colour Map of confusion matrix
    kwargs :
        figsize : tuple of int or int
            Matplotlib key word to set size of plot

    Returns
    -------
    : Figure
       confusion matrix figure
    """

    assert (len(labels) == len(cm[0])), "There must be the same number of columns in the confusion matrix as there" \
                                        "is labels available"

    fig, ax = plt.subplots()
    if "figsize" in kwargs.keys():
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=kwargs["figsize"])

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=labels, yticklabels=labels,
           title=title, ylabel='True label', xlabel='Predicted label')

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

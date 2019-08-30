# -*- coding: utf-8 -*-
"""
Part of slugdetection package

@author: Deirdree A Polak
github: dapolak
"""


import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, r2_score

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class Slug_Forecasting:
    """
    Trains an ARIMA model to forecast slug flow in offshore oil wells.
    Parameters
    ----------
    slug_flow_whp : Pandas DataFrame
        well data frame, including a WHP and ts column. Must be presenting slug flow characteristics
    Attributes
    ----------
    slug_df : Pandas DataFrame
        well data frame, with DateTimeIndex and single WHP column.
    """

    def __init__(self, slug_flow_whp):
        assert (len(slug_flow_whp) >= 240), "Assert there is enough data to train the model on"
        assert "WH_P" in slug_flow_whp.columns
        assert "ts" in slug_flow_whp.columns
        self.slug_df = slug_flow_whp.set_index('ts').copy()

        for col in self.slug_df.columns:
            if col != "WH_P":
                self.slug_df = self.slug_df.drop(col, axis=1)

    def stationarity_check(self, diff=0):
        """
        Checks whether slug_df attribute is stationary by applying the Augmented Dickey-Fuller test.

        Parameters
        ----------
        diff : int (optional)
            If data is not stationary, can be used to stationarise it. Value should be used as the d parameter in
            the ARIMA model training (default is 0)
        """

        slugs = self.slug_df["WH_P"].copy()  # Get copy of slug df for testing

        for i in range(diff):  # Apply differencing
            slugs = slugs.diff()[1:]

        self.station_result = adfuller(slugs)  # Apply augmented dickey-fuller test

        # Print results for user information
        print('ADF Statistic: %f' % self.station_result[0])
        print('p-value: %f' % self.station_result[1])
        print('Critical Values:')
        for key, value in self.station_result[4].items():
            print('\t%s: %.3f' % (key, value))

    def split_data(self, train_size=180, predict_size=60):
        """
        Splits data into a training set and a testing set. Split is performed chronologically.
        Parameters
        ----------
        train_size : int
            Number of minutes to train the data on
        predict_size : int
            Number of minutes to calculate prediction error on
        """
        assert isinstance(train_size, int), "Train size must be an integer"
        assert isinstance(predict_size, int), "Test size is an integer"
        assert train_size + predict_size <= len(self.slug_df), \
            "There must be enough data to split into trin size and predit_size"

        self.y_train = self.slug_df[:train_size].copy()
        self.y_pred = self.slug_df[train_size:train_size + predict_size].copy()

    def autocorrelation_plot(self, lags=25):
        """
        Plots Auto Correlation figure
        Parameters
        ----------
        lags : int
            Number of lags to plot ACF for

        Returns
        -------
        : Figure
           ACF figure
        """
        return plot_acf(self.y_train, lags=lags)

    def partial_autocorrelation_plot(self, lags=25):
        """
        Plots Partial Auto Correlation figure
        Parameters
        ----------
        lags : int
            Number of lags to plot PACF for

        Returns
        -------
        : Figure
           PACF figure
        """
        return plot_pacf(self.y_train, lags=lags)

    def ARIMA_model(self, p, d, q, show=True):
        """
        Instantiate, fits and plots best fit of an ARIMA model with the parameters p, d and q.
        Parameters
        ----------
        p : int
            ARIMA Auto Regressive Property. As approximated from the PACF plot.
        d : int
            ARIMA Integrated Property. As approximated by the difference value required to reach stationarity
        q : int
            ARIMA Moving Average Property. As approximated from the ACF plot.

        Returns
        -------
        : Figure
           ARIMA fit figure
        """

        assert hasattr(self, "y_train"), "Data must have been split"

        ARIMA_model = ARIMA(self.y_train, order=(p, d, q))  # Instantiate ARIMA model with parameters and y_train data
        self.fit_results = ARIMA_model.fit(disp=-1)  # Fit ARIMA model
        if show:
            # Plot fit
            f, ax = plt.subplots(1, 1, figsize=(12, 5))
            ax.plot(self.y_train, "C0-", label="Actual")
            ax.plot(self.fit_results.fittedvalues, 'r-', label="Fitted")
            ax.set_title('ARIMA model (%i, %i, %i) for WHP' % (p, d, q))
            ax.set_xlabel("Time")
            ax.set_ylabel("Pressure in BarG")
            ax.legend()
            return f

    def error_metrics(self, error, verbose=True):
        """
        Computes error metrics for the error of the regression as compared to the true data

        Parameters
        ----------
        error : str
            Keyword to compute error metrics for the ARIMA model regression for the training data, or the forecast for
            the testing data. Takes "fit" ot "pred".

        Returns
        -------
        mape : float
            Mean Absolute Percentage Error for the regression. The smaller the value, the better the fit
        mse : float
            Mean Squared Error for the regression. The smaller error, the better the fit
        rmse : float
            Root Mean Squared Error for the regression. The smaller the error the better the fit
        r2 : float
            Coefficient of Determination for the regression. The closer to 1, the better the fit.
        """

        if error == "fit":
            assert hasattr(self, "fit_results"), "ARIMA model must have been created and fitted"
            forecast = self.fit_results.fittedvalues
            true = self.y_train
            resid = self.fit_results.resid
        elif error == "pred":
            assert hasattr(self, "forecast"), "Forecast attribute must have been created"
            assert len(self.forecast) <= len(self.y_pred)
            forecast = self.forecast
            true = self.y_pred[:len(forecast)]
            resid = forecast - true["WH_P"]
        else:
            print("Parameter not recognised. Try 'fit' or 'pred'")
            return

        mse = round(mean_squared_error(true, forecast), 3)
        rmse = round(math.sqrt(sum(resid ** 2) / len(true)), 3)
        r2 = round(r2_score(true, forecast), 3)
        mape = resid / true["WH_P"]
        mape = round(mape.mean() * 100, 3)

        if verbose:
            print("Mean Absolute Percentage Error: ", mape)
            print("Mean Squared Error: %f" % mse)
            print("Root Mean Squared Error: %f" % rmse)
            print("R2 Determination: %f" % r2)

        return mape, mse, rmse, r2

    def error_metrics_plot(self, error):
        """
        Plots infographics on the error of the ARIMA model regression

        Parameters
        ----------
        error : str
            Keyword to compute error metrics for the ARIMA model regression for the training data, or the forecast for
            the testing data. Takes "fit" ot "pred".

        Returns
        -------
        : Figure
           Error infographics figure
        """
        from scipy.stats import norm, gaussian_kde

        if error == "fit":
            assert hasattr(self, "fit_results"), "ARIMA model must have been created and fitted"
            forecast = self.fit_results.fittedvalues
            true = self.y_train
            resid = self.fit_results.resid
        elif error == "pred":
            assert hasattr(self, "forecast"), "Forecast attribute must have been created"
            assert len(self.forecast) <= len(self.y_pred)
            forecast = self.forecast
            true = self.y_pred[:len(forecast)]
            resid = forecast - true["WH_P"]
        else:
            print("Parameter not recognised. Try 'fit' or 'pred'")
            return

        mape, mse, rmse, r2 = self.error_metrics(error=error, verbose=False)

        fig, ax = plt.subplots(2, 2, constrained_layout=True)

        # Residual
        ax[0][0].plot(resid)
        ax[0][0].set_title("Residuals")
        ax[0][0].set_xlabel("Time")
        ax[0][0].set_ylabel("Residuals")
        ax[0][0].grid(True)
        ax[0][0].set_xticklabels([''])
        ax[0][0].legend()

        # Display R2 correlation
        ax[0][1].plot(forecast, true, 'k.')
        ax[0][1].plot([math.floor(min(forecast)), math.ceil(max(forecast))],
                      [math.floor(min(forecast)), math.ceil(max(forecast))], 'r-')
        ax[0][1].set_xlabel("Forecast values")
        ax[0][1].set_ylabel("True values")
        ax[0][1].set_title("R2 : %4f" % r2)
        ax[0][1].legend()

        # Display Auto Correlation Graph
        plot_acf(resid, lags=20, ax=ax[1][0])
        ax[1][0].set_title('ACF Residuals')

        # Display Density function
        x = np.linspace(min(resid), max(resid), 100)
        ax[1][1].hist(resid / max(resid), bins=10)  # history
        ax2 = ax[1][1].twinx()
        density = gaussian_kde(resid)
        ax2.plot(x, density(x), 'C1', label="KDE")
        ax2.plot(x, norm.pdf(x, 0), 'C2', label="N(0,1)")

        ax2.axvline(linestyle='--', color='gray')
        ax[1][1].set_title('Density plot')
        ax[1][1].set_xlabel("Resid")
        ax[1][1].set_ylabel("Density")
        ax2.legend()

        return fig


    def ARIMA_pred(self, pred_time, y_true=True, show=True):
        """
        Forecasts and plots the expected values
        Instantiate, fits and plots best fit of an ARIMA model with the parameters p, d and q.
        Parameters
        ----------
        pred_time : int
            Number of minutes to forecast until
        y_true : bool (optional)
            Whether true data for the forecasting interval is known (default is True)

        Returns
        -------
        : Figure
           ARIMA forecast figure
        """

        assert hasattr(self, "fit_results")
        self.forecast, se, conf = self.fit_results.forecast(pred_time, alpha=0.05)  # 95% conf

        if show:
            fig, ax = plt.subplots(figsize=(20, 5))

            ax.plot(self.y_train, label='Training')
            if y_true:
                ax.plot(self.y_pred[:pred_time], label='Actual')
            ax.plot(self.y_pred.index[:pred_time], self.forecast, label='Forecast')
            ax.fill_between(self.y_pred.index[:pred_time], conf[:, 0], conf[:, 1],
                            color='k', alpha=.15, label="95 % Confidence")
            ax.set_title('Forecast vs Actuals')
            ax.legend()
            return fig
class Slug_Labelling(Data_Engineering):
    """
    Clusters together flow types based on behaviour of variables using Unsupervised Learning
    """

    def __init__(self, well):
        """
        Parameters
        ----------
        well : Spark Data Frame
            well data frame, includes continuous pressure and temperature data
        """
        Data_Engineering.__init__(self, well)

    def feature_vector(self, window_size=20, step=5, standardise=True, keep_ts=False):
        """
        Transforms the raw data into features vectors containing the all the data points in the time window set by
        the window_size. The feature vectors can then be used for classification.

        Parameters
        ----------
        window_size : int
            Length of time in minutes that will be considered in the feature vectors (optional, default is 20)
        step : int
            Time step between each window of time. Can also be considered as the overlap in time. For example, if the
            step is set to 5, a feature vector will be created each five minutes of the continuous dataset
            (optional, default is 5)
        standardise : bool
            Whether to standardise the data (optional, default is True)
        keep_ts : bool
            Whether to keep to timestamp column (optional, default is False)

        """

        # save user selected window size and step as attributes for future uses
        self.window_size = window_size
        self.step = step

        assert hasattr(self, "pd_df"), "Attribute Pandas data frame must exist"
        assert not self.pd_df.empty, "Attribute Pandas data frame cannot be empty"

        # based on user entry, standardise data
        if standardise:
            self.feature_vec = self.standardise(self.pd_df.copy())
        else:
            self.feature_vec = self.pd_df.copy()

        # Loop through window size to create new features
        for i in range(1, self.window_size + 1):
            new_features = []  # emtpy list to save new features names
            for feature_ in self.pd_df.columns:
                if feature_ != 'ts':
                    new_features.append(feature_ + "_" + str(i))  # create new column names for each time step
            # fill new columns with lagged data
            self.feature_vec[new_features] = self.feature_vec[self.features].shift(periods=-i).fillna(0)

        self.feature_vec = self.feature_vec[self.window_size:]  # get rid of 0 values at the beginning
        self.feature_vec = self.feature_vec[::self.step]  # only keep every step-th row

        if not keep_ts:  # based on user entry, delete timestamp column
            self.feature_vec = self.feature_vec.drop("ts", axis=1)  # get rid of timestamp

        return

    def Kmean_classification(self, n_labels, verbose=True):
        """
        Classifies the data from the feature_vec attribute created in the feature_vector method in n_labels clusters.

        Parameters
        ----------
        n_labels : int
            Number of labels to sort the feature_vec data in (number of clusters)
        verbose : bool
            Whether to print results
        """

        assert hasattr(self, "feature_vec"), "Feature vector attribute must exist"

        self.n_clusters = n_labels  # set the user defined number of clustered as a class attribute

        kmeans = KMeans(n_clusters=self.n_clusters)  # Instantiate KMeans model
        kmeans.fit(self.feature_vec)  # Fit KMeans model to feature vector data
        self.labels = kmeans.labels_  # Set labels attribute as KMEans defined labels for the data

        if verbose:
            for i in range(self.n_clusters):
                # for each cluster, print the number of feature vectors clustered, and relative percentage to the
                # training data set
                mask = self.labels == i
                print("Cluster", i, "contains", round(len(self.feature_vec[mask]) / len(self.feature_vec) * 100, 3),
                      "% of the data\nor", len(self.feature_vec[mask]), "points")

        return

    def get_labels(self):
        """
        Get the labels of the data windows

        Returns
        -------
        labels: list of int
          List of labels. There will be as many different labels as the number of clusters selected
          for the Kmeans_classification method.
        """
        assert hasattr(self, "labels"), "Attribute labels must exist"
        return self.labels

    def label_plot(self, data_streams):
        """
        Plotting method to visualise how the feature_vec data was clustered along three features

        Parameters
        ----------
        data_streams : list of str
            List of the three variables to plot the data against

        """
        assert hasattr(self, "labels"), "Attribute labels must exist"
        assert hasattr(self, "feature_vec"), "Feature vector attribute must exist"
        assert len(data_streams) == 3, "Assert there are 3 variables to plot the cluster graph"
        assert len(self.labels) == len(self.feature_vec), "Assert there is the same number of labels as data points"

        for var in data_streams:
            assert var in self.pd_df.columns, var + "does not exist in pandas data frame attribute"

        # Create figure and axis
        fig = plt.figure()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        # Set each variables in the data stream parameter as an axis
        # Plot the values, and colour them depending on their label as clustered by KMeans
        ax.scatter(self.feature_vec[data_streams[0]], self.feature_vec[data_streams[1]],
                   self.feature_vec[data_streams[2]], c=self.labels.astype(float))
        ax.set_xlabel(data_streams[0])
        ax.set_ylabel(data_streams[1])
        ax.set_zlabel(data_streams[2])
        display(fig)

    @property
    def unpack_feature_vector(self):
        """
        After the creation of the labels for each feature_vec, unpack the vectors into short continuous data frames
        of size window_size attribute for visualisation purposes.

        Returns
        -------
        unpacked_list: list of Pandas data frames
            List of size n_labels. Eac entry in the list is a datframe containing the data for the windows of
            time clustered in one cluster.

        """
        assert hasattr(self, "feature_vec"), "Feature vector attribute must exist"
        assert hasattr(self, "labels"), "Attribute labels must exist"

        unpacked_list = []  # Create empty list to append pandas data frames
        headers = []  # Create empty list to save header names

        for feature_ in self.features:  # loop through each features
            new_features = []  # Empty list to append column header
            for i in range(1, self.window_size + 1):
                new_features.append(feature_ + "_" + str(i))  # create new column names for each time step
            headers.append(new_features)  # Append list of same features headers to headers list

        for i in range(self.n_clusters):
            temp = pd.DataFrame()  # Create temporary pandas data frame
            mask = self.labels == i  # Create mask for current label
            masked_input = self.feature_vec[mask]  # Create secondary data frame for feature vector for current label

            for j, feature_ in enumerate(self.features):
                # Add data for each feature as a list to the temp data frame. The temp data frame will have a total
                # of len(self.features) columns. Each column is named after a feature and contains the list of the
                # data for that specific feature from the feature vec data frame. For example, column 'WH_P' will
                # contain a list [12.0, 12.1, 12.2, 12.1, 12.3] if the window size was set to five by the user.
                temp[feature_] = masked_input[headers[j]].values.tolist()

            # Create a timestamp column, with time from 0 to minus the window size
            temp['ts'] = [[-i for i in range(1, self.window_size + 1)]] * len(masked_input)

            # Append temp data frame, for the current label, to unpacked list
            unpacked_list.append(temp)

        return unpacked_list

    def plot_feature_vectors(self, examples=1000):
        """
        Plotting function to overlay the feature vectors clustered together. This visualisation can then assist users
        in labelling the flow types based on knowledge of variable behaviour

        Parameters
        ----------
        examples : int
            Number of feature vectors to plot in each visualisation

        """
        assert hasattr(self, "labels"), "Attribute labels must exist"
        assert hasattr(self, "n_clusters"), "Attribute n_clusters must exist"

        # Unpack feature vector data into plottable data frame
        unpacked_list = self.unpack_feature_vector

        fig, ax = plt.subplots(self.n_clusters, 1, figsize=(15, int(3 * self.n_clusters)), constrained_layout=True)

        for i in range(self.n_clusters):
            for j in range(examples):  # overlay n_examples data from each label
                if j < len(unpacked_list[i]):  # if there's enough data in the cluster
                    for k, feature_ in enumerate(self.features):
                        ax[i].plot(unpacked_list[i]["ts"][j], unpacked_list[i][feature_][j],
                                   "C" + str(k) + "o-", label=feature_ if (j == 0) & (i == 0) else '_nolegend_')
                        ax[i].set_title("Cluster" + str(i))
                        ax[i].set_xlabel("Minutes to 0")
                        ax[i].grid(True, which='both')
                        ax[i].set_ylabel("Pressure//Temperature//Choke")
        fig.legend()
        display(fig)

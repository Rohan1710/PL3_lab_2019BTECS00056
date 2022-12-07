from cProfile import label
from collections import Counter
import math
import smtpd
from ssl import Options
from statistics import mean
from turtle import update
import matplotlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.express as px
from logging import critical
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import itertools

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from os import system
from graphviz import Source
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

from sklearn.preprocessing import MinMaxScaler
import math
import random
# import csv

genre = st.radio(
    "Main menu",
    ('Assignment-1', 'Assignment-2', 'Assignment-3', 'Assignment-4', 'Assignment-5', 'Assignment-6', 'Assignment-7', 'Assignment-8'))


if genre == 'Assignment-1':
    st.title('DM Assignment-1')

    uploaded_file = st.file_uploader(label="Choose a file",
                                     type=['csv', 'xlsx'])

    global df
    global numeric_columns
    if uploaded_file is not None:
        print(uploaded_file)
        print("Hello")
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            print(e)
            df = pd.read_excel(uploaded_file)

    try:
        st.write(df)
        numeric_columns = list(df.columns)
    except Exception as e:
        print(e)
        st.write("Please upload file")

    st.subheader(
        "The measures of central tendency and dispersion of data")

    mean1 = df['sepal.length'].mean()

    median1 = df['sepal.length'].median()
    mode1 = df['sepal.length'].mode()
    # midRange1 = df['Salary'].()

    std1 = df['sepal.length'].std()
    var1 = df['sepal.length'].var()
    quantile1 = df['sepal.length'].quantile()

    print('Mean salary: ' + str(mean1))

    st.write("Mean : " + str(mean1))
    st.write("Median : " + str(median1))
    st.write("Mode : " + str(mode1))
    # st.write("Midrange : " + str(midRange1))
    st.write("variance : " + str(var1))
    st.write("Standard Deviation : " + str(std1))
    st.write("Quantile : " + str(quantile1))

    print(df.columns.values)
    print(df['sepal.length'].values)

    # Calculate mean
    mean1 = 0
    count = 0
    for x in df['sepal.length'].values:
        mean1 += x
        count = count + 1
    mean1 = mean1/count
    print(mean1)

    # Calcualte Median
    data1 = df['sepal.length'].values

    def median(data1):
        sorted_data = sorted(data1)
        data_len = len(sorted_data)

        middle = (data_len - 1) // 2

        if middle % 2:
            return sorted_data[middle]
        else:
            return (sorted_data[middle] + sorted_data[middle + 1]) / 2.0

    print(median(data1))

    n = len(data1)

    # calulate mode
    d = Counter(data1)
    get_mode = dict(d)
    mode = [k for k, v in get_mode.items() if v == max(list(d.values()))]

    if len(mode) == n:
        get_mode = "No mode found"
    else:
        get_mode = "Mode is / are: " + ', '.join(map(str, mode))

    print(get_mode)

    # calculate midrange
    n = len(data1)
    arr = []
    for i in range(len(data1)):
        # arr.append(df.loc[i, attribute])
        arr.sort()
    # print("Midrange of given dataset is ("+") "+str((arr[n-1]+arr[0])/2))
    # st.write("Midrange of given dataset is ("
    #  + ") "+str((arr[n-1]+arr[0])/2))

    def variance(data):
        # Number of observations
        n = len(data)
        # Mean of the data
        mean = sum(data) / n
        # Square deviations
        deviations = [(x - mean) ** 2 for x in data]
        # Variance
        variance = sum(deviations) / n
        return variance

    print(variance(data1))

    def stdev(data):
        var = variance(data)
        std_dev = math.sqrt(var)
        return std_dev

    print(stdev(data1))

    def rank(data1):
        sorted_data = sorted(data1)
        n = len(sorted_data)
        mid = 0
        if n % 2:
            mid = (n + 1)/2
        else:
            mid = n/2
        mid = int(mid)
        print('mid' + str(mid))
        print('Range' + str(n - mid))
        st.write("Range : " + str(n - mid))
        # print(sorted_data[mid])
        # iq1 = int(mid/2)
        # iq2 = iq1 + int((n-mid)/2)
        # r1 = sorted_data[iq2] - sorted_data[iq1]
        # print(str(iq1) + ' ' + str(iq2))
        # print(r1)
        Q1 = np.median(sorted_data[:mid])

        # Third quartile (Q3)
        Q3 = np.median(sorted_data[mid:])

        # Interquartile range (IQR)
        IQR = Q3 - Q1
        print('Interquartile Range' + str(IQR))
        st.write("Interquartile Range : " + str(IQR))

    # smtpd.qqplot(data1, line='45')
    # st.show(smtpd.qqplot(data1, line='45'))

    rank(data1)

    st.subheader(
        "Graphical display of above calculated statistical description of data")
    # Add a select weight to the sidebar
    chart_select = st.sidebar.selectbox(
        label="Select the chart type",
        options=['Scatterplits', 'Lineplots', 'Histogram', 'Boxplot']
    )

    if chart_select == 'Scatterplits':
        st.sidebar.subheader("Scatterplot Settings")
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            plot = px.scatter(data_frame=df, x=x_values, y=y_values)
            # display chart
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    if chart_select == 'Lineplots':
        st.sidebar.subheader("Lineplot Settings")
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            plot = px.line(data_frame=df, x=x_values, y=y_values)
            # display chart
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    if chart_select == 'Histogram':
        st.sidebar.subheader("Histogram Settings")
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            plot = px.histogram(data_frame=df, x=x_values, y=y_values)
            # display chart
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    if chart_select == 'Boxplot':
        st.sidebar.subheader("Boxplot Settings")
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            plot = px.box(data_frame=df, x=x_values, y=y_values)
            # display chart
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

if genre == 'Assignment-2':
    st.title('DM Assignment-2')

    uploaded_file = st.file_uploader(
        label="Choose a file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        print(uploaded_file)
        df = pd.read_csv(uploaded_file)

        st.write(df)
        numeric_columns = list(df.columns)

        pd.set_option('display.max_columns', 100)
        edu = pd.read_csv("xAPI-Edu-Data.csv")

        # print(edu.columns)

        # contingency = pd.crosstab(edu['ParentAnsweringSurvey'], edu["GradeID"])

        # Add a select weight to the sidebar
        st.sidebar.subheader("Select Attributes")
        try:
            attribute1 = st.sidebar.selectbox(
                'Attribute-1', options=numeric_columns)
            attribute2 = st.sidebar.selectbox(
                'Attribute-2', options=numeric_columns)
            contingency = pd.crosstab(edu[attribute1], edu[attribute2])

            # display chart
            st.subheader("Contingency Table")
            st.table(contingency)
        except Exception as e:
            print(e)

        # Print contingency table
        print(contingency)

        # st.write(contingency)

        observed_values = contingency.values
        print(observed_values)

        #  we will get expected values
        val = stats.chi2_contingency(contingency)
        print(val)
        st.write(val)

        alpha = 0.05
        dof = 9  # degree of freedom

        # calculating critical value
        critical_value = stats.chi2.ppf(q=1 - alpha, df=dof)
        print(critical_value)

        # st.subheader("Chi-square value(Expected)" + str(val[''][]))
        st.subheader("Chi-square value(Critical) : " + str(critical_value))

        # Conslusion
        st.subheader("Conclusion : ")

        chi2_value_expected = val[0]
        print(chi2_value_expected)
        if chi2_value_expected < critical_value:
            print("So, the chi2 test statistic ( " + str(chi2_value_expected) + ") we calculated is smaller than the chi2 critical ( " +
                  str(critical_value) + ") value we have got from the distribution. So, we do not have enough evidence to reject the null hypothesis.")
            st.write("So, the chi2 test statistic ( " + str(chi2_value_expected) + ") we calculated is smaller than the chi2 critical ( " +
                     str(critical_value) + ") value we have got from the distribution. So, we do not have enough evidence to reject the null hypothesis.")
        else:
            print("So, the chi2 test statistic ( " + str(chi2_value_expected) + ") we calculated is greater than the chi2 critical ( " +
                  str(critical_value) + ") value we have got from the distribution. So, we do have enough evidence to reject the null hypothesis.")

        # Correlation analysis : Correlation coefficient (Pearson coefficient) & Covariance
        st.subheader(
            "Correlation analysis using Correlation coefficient (Pearson coefficient) & Covariance")

        st.sidebar.subheader("Select Attributes(for correlation coefficient)")
        try:
            attribute1 = st.sidebar.selectbox(
                'Attribute-11', options=numeric_columns)
            attribute2 = st.sidebar.selectbox(
                'Attribute-22', options=numeric_columns)
            var = np.corrcoef(df[attribute1], df[attribute2])
            corr, _ = pearsonr(df[attribute1], df[attribute2])
            print('Pearsons correlation: %.3f' % corr)
            st.write('Pearsons correlation: %.3f' % corr)
            # display chart
            st.subheader("Table")
            st.write(var)
        except Exception as e:
            print(e)

        # Calculating coefficient with formulas
        # list1 = list(newDf['salary'])
        # list2 = list(newDf['salbegin'])
        # n = len(list1)
        # mean1 = sum(list1) / n
        # mean2 = sum(list2) / n

        # n, mean1, mean2
        # num = 0
        # for i in range(n):
        #     num = num + (list1[i] - mean1) * (list2[i] - mean2)

        # num
        # SS1 = 0
        # for i in list1:
        #     SS1 = SS1 + (i - mean1)**2

        # SS2 = 0
        # for i in list2:
        #     SS2 = SS2 + (i - mean2)**2

        # SS1, SS2
        # pearsonCorr = num / (SS1 * SS2)**0.5
        # pearsonCorr

        # Normalization using following techniques :
        # 1. Min-max normalization
        attribute_to_be_normalized = st.sidebar.selectbox(
            'attribute_to_be_normalized', options=numeric_columns)
        min = df['attribute_to_be_normalized'].min()
        max = df[attribute_to_be_normalized].max()
        for i in range(len(df)):
            df.loc[i, attribute_to_be_normalized] = (
                (df.loc[i, attribute_to_be_normalized]-min)/(max-min))
            print(df['attribute_to_be_normalized'])
            st.write(df['attribute_to_be_normalized'])

        # st.plotly_chart(df.plot(kind='bar'))
        # x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
        # y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
        # plot = px.scatter(data_frame=df, x=x_values, y=y_values)
        # # display chart
        # st.plotly_chart(plot)
        # # copy the data
        # df_min_max_scaled = df.copy()

        # v = stats.zscore(df_min_max_scaled)

        # # view normalized data
        # st.plotly_chart(df_min_max_scaled)

        # rho = np.corrcoef(x)

        # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
        # for i in [0,1,2]:
        #     ax[i].scatter(x[0,],x[1+i,])
        #     ax[i].title.set_text('Correlation = ' + "{:.2f}".format(rho[0,i+1]))
        #     ax[i].set(xlabel='x',ylabel='y')
        # fig.subplots_adjust(wspace=.4)
        # plt.show()

if genre == 'Assignment-3':
    st.title('DM Assignment-3')

    uploaded_file = st.file_uploader(
        label="Choose a file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        print(uploaded_file)
        df = pd.read_csv(uploaded_file)

        st.write(df)
        numeric_columns = list(df.columns)

        print("Dataset Length: ", len(df))
        print("Dataset Shape: ", df.shape)

        # Printing the dataset obseravtions
        print("Dataset: ", df.head())

        # Separate the independent and dependent variables using the slicing method.
        # x = df.values[:, 1.5]
        # y = df.values[:, 0]

        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

        print(df.isnull().sum())  # checked if any attribute has null value

        columns = df.columns
        feature_cols = columns[1:]

        target_cols = columns[0:1]

        feature_data = df[:][1:]
        target_data = df[:][0:1]
        X = df[feature_cols]
        Y = df[target_cols]

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.3, random_state=1)

        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier(max_depth=2, random_state=1)

        # Train Decision Tree Classifer
        clf = clf.fit(X_train, y_train)

        # Predict the response for test dataset
        y_pred = clf.predict(X_test)

        # text.delete("1.0","end")
        st.write("Model Accuracy: " +
                 str(metrics.accuracy_score(y_test, y_pred)))

        c_matrix = confusion_matrix(y_test, y_pred)
        print(c_matrix)
        st.write(str(c_matrix))

        print("True Positive:" + str(c_matrix[0][0]))
        st.write("True Positive:" + str(c_matrix[0][0]))

        st.subheader("Dicision Tree")
        print(X.columns)
        print(Y.columns)
        plt.figure(figsize=(4, 3), dpi=150)
        # st.plotly_chart(plot_tree(clf, feature_names=X.columns, filled=True))
        # st.write(plt.show(block=True))

        # st.pyplot(tree.plot_tree(clf))
        tree = export_graphviz(clf)

        st.graphviz_chart(tree)

        # Tabulate the results in confusion matrix and evaluate the performance of above classifier using following metrics :
        st.write('Tabulate the results in confusion matrix and evaluate the performance of above classifier using following metrics :')

        tp = c_matrix[1][1]
        tn = c_matrix[2][2]
        fp = c_matrix[1][2]
        fn = c_matrix[2][1]

        # Recognition rate

        print('he;;p')
        # precision score
        val = metrics.precision_score(y_test, y_pred, average='macro')
        print('Precision score : ' + str(val))
        st.write('Precision score : ' + str(val))

        # Accuracy score
        val = metrics.accuracy_score(y_test, y_pred)
        st.write('Accuracy score : ' + str(val))

        #
if genre == 'Assignment-4':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    st.title('DM Assignment-4')

    uploaded_file = st.file_uploader(
        label="Choose a file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        print(uploaded_file)
        df = pd.read_csv(uploaded_file)

        st.write(df)
        numeric_columns = list(df.columns)

    print("Dataset Length: ", len(df))
    print("Dataset Shape: ", df.shape)

    # Printing the dataset obseravtions
    print("Dataset: ", df.head())

    # Separate the independent and dependent variables using the slicing method.
    # x = df.values[:, 1.5]
    # y = df.values[:, 0]

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

    print(df.isnull().sum())  # checked if any attribute has null value

    columns = df.columns
    feature_cols = columns[1:]

    target_cols = columns[0:1]

    feature_data = df[:][1:]
    target_data = df[:][0:1]
    X = df[feature_cols]
    Y = df[target_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=1)

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(max_depth=2, random_state=1)

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # text.delete("1.0","end")
    st.write("Model Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)))

    c_matrix = confusion_matrix(y_test, y_pred)
    print(c_matrix)
    st.write(str(c_matrix))

    print("True Positive:" + str(c_matrix[0][0]))
    st.write("True Positive:" + str(c_matrix[0][0]))

    print(X.columns)
    print(Y.columns)
    plt.figure(figsize=(4, 3), dpi=150)
    # st.plotly_chart(plot_tree(clf, feature_names=X.columns, filled=True))
    # st.write(plt.show(block=True))

    # st.pyplot(tree.plot_tree(clf))
    tree = export_graphviz(clf)

    st.graphviz_chart(tree)

    # I believe that this answer is more correct than the other answers here:

    st.subheader('Extracting rules for iris - dataset')

    def get_rules(tree, feature_names, class_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        paths = []
        path = []

        def recurse(node, path, paths):

            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                p1, p2 = list(path), list(path)
                p1 += [f"({name} <= {np.round(threshold, 3)})"]
                recurse(tree_.children_left[node], p1, paths)
                p2 += [f"({name} > {np.round(threshold, 3)})"]
                recurse(tree_.children_right[node], p2, paths)
            else:
                path += [(tree_.value[node], tree_.n_node_samples[node])]
                paths += [path]

        recurse(0, path, paths)

        # sort by samples count
        samples_count = [p[-1][1] for p in paths]
        ii = list(np.argsort(samples_count))
        paths = [paths[i] for i in reversed(ii)]

        rules = []
        for path in paths:
            rule = "if "

            for p in path[:-1]:
                if rule != "if ":
                    rule += " and "
                rule += str(p)
            rule += " then "
            if class_names is None:
                rule += "response: "+str(np.round(path[-1][0][0][0], 3))
            else:
                classes = path[-1][0][0]
                l = np.argmax(classes)
                rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
            rule += f" | based on {path[-1][1]:,} samples"
            rules += [rule]

        return rules

    rules = get_rules(clf, iris.feature_names, iris.target_names)
    for r in rules:
        print(r)
        st.write(r)

if genre == "Assignment-5":
    st.title('DM Assignment-5')

    uploaded_file = st.file_uploader(label="Choose a file",
                                     type=['csv', 'xlsx'])

    if uploaded_file is not None:
        print(uploaded_file)
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        numeric_columns = list(df.columns)

        st.dataframe(df)

        rad5 = st.radio("Select", ["Regression classifier",
                        "Naive Bayesian classifier", "k-NN classifier", "Artificial Neural Network Classifier"])

        if rad5 == "Regression classifier":

            int_class = []

            # Setosa = 1
            # Versicolor = 2
            # Virginica = 3

            for i in df['variety']:
                if i == 'Setosa':
                    int_class.append(1)
                if i == 'Versicolor':
                    int_class.append(2)
                if i == 'Virginica':
                    int_class.append(3)

            df['int_class'] = int_class

            st.dataframe(df)

            # defining feature matrix(X) and response vector(y)
            x = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
            y = df['int_class']

            # y = y[:-2]
            print(x)
            print(y)

            reg = linear_model.LinearRegression()
            reg.fit(x, y)

            coefficient = reg.coef_
            intercept = reg.intercept_

            print(coefficient)
            print(intercept)

            st.write('Setosa : 1')
            st.write('Versicolor : 2')
            st.write('Virginica : 3')

            st.write()

            sepal_length = number = st.number_input('Sepal.length')
            sepal_width = number = st.number_input('Sepal.width')
            petal_length = number = st.number_input('patel.length')
            petal_width = number = st.number_input('patel.width')

            predicted = reg.predict(
                [[sepal_length, sepal_width, petal_length, petal_width]])
            print('class value')
            print(predicted)

            st.write('Class value : ' + str(predicted))

            # plt.scatter(x, y)
            # plt.xlabel('x')
            # plt.ylabel('y')
            # plt.show()
            # plot = px.scatter(
            #     df['sepal.length'], df['variety'], x='x', y='y')
            # st.ploty_chart(plot)

            # plot = px.scatter(x, y, x='x', y='y')
            # st.ploty_chart(plot)
            # reg.predict()

        if rad5 == "Naive Bayesian classifier":

            def accuracy_score(y_true, y_pred):
                """	score = (y_true - y_pred) / len(y_true) """

                return round(float(sum(y_pred == y_true))/float(len(y_true)) * 100, 2)

            def pre_processing(df):
                """ partioning data into features and target """

                X = df.drop([df.columns[-1]], axis=1)
                y = df[df.columns[-1]]

                return X, y

            class NaiveBayes:

                """
                    Bayes Theorem:
                                                    Likelihood * Class prior probability
                            Posterior Probability = -------------------------------------
                                                        Predictor prior probability

                                                    P(x|c) * p(c)
                                        P(c|x) = ------------------
                                                        P(x)
                """

                def __init__(self):
                    """
                        Attributes:
                            likelihoods: Likelihood of each feature per class
                            class_priors: Prior probabilities of classes
                            pred_priors: Prior probabilities of features
                            features: All features of dataset
                    """
                    self.features = list
                    self.likelihoods = {}
                    self.class_priors = {}
                    self.pred_priors = {}

                    self.X_train = np.array
                    self.y_train = np.array
                    self.train_size = int
                    self.num_feats = int

                def fit(self, X, y):

                    self.features = list(X.columns)
                    self.X_train = X
                    self.y_train = y
                    self.train_size = X.shape[0]
                    self.num_feats = X.shape[1]

                    for feature in self.features:
                        self.likelihoods[feature] = {}
                        self.pred_priors[feature] = {}

                        for feat_val in np.unique(self.X_train[feature]):
                            self.pred_priors[feature].update({feat_val: 0})

                            for outcome in np.unique(self.y_train):
                                self.likelihoods[feature].update(
                                    {feat_val+'_'+outcome: 0})
                                self.class_priors.update({outcome: 0})

                    self._calc_class_prior()
                    self._calc_likelihoods()
                    self._calc_predictor_prior()

                def _calc_class_prior(self):
                    """ P(c) - Prior Class Probability """

                    for outcome in np.unique(self.y_train):
                        outcome_count = sum(self.y_train == outcome)
                        self.class_priors[outcome] = outcome_count / \
                            self.train_size

                def _calc_likelihoods(self):
                    """ P(x|c) - Likelihood """

                    for feature in self.features:

                        for outcome in np.unique(self.y_train):
                            outcome_count = sum(self.y_train == outcome)
                            feat_likelihood = self.X_train[feature][self.y_train[self.y_train == outcome].index.values.tolist(
                            )].value_counts().to_dict()

                            for feat_val, count in feat_likelihood.items():
                                self.likelihoods[feature][feat_val +
                                                          '_' + outcome] = count/outcome_count

                def _calc_predictor_prior(self):
                    """ P(x) - Evidence """

                    for feature in self.features:
                        feat_vals = self.X_train[feature].value_counts(
                        ).to_dict()

                        for feat_val, count in feat_vals.items():
                            self.pred_priors[feature][feat_val] = count / \
                                self.train_size

                def predict(self, X):
                    """ Calculates Posterior probability P(c|x) """

                    results = []
                    X = np.array(X)

                    for query in X:
                        probs_outcome = {}
                        for outcome in np.unique(self.y_train):
                            prior = self.class_priors[outcome]
                            likelihood = 1
                            evidence = 1

                            for feat, feat_val in zip(self.features, query):
                                likelihood *= self.likelihoods[feat][feat_val + '_' + outcome]
                                evidence *= self.pred_priors[feat][feat_val]

                            posterior = (likelihood * prior) / (evidence)

                            probs_outcome[outcome] = posterior

                        result = max(
                            probs_outcome, key=lambda x: probs_outcome[x])
                        results.append(result)

                    return np.array(results)

            if __name__ == "__main__":

                # Weather Dataset
                print("\nWeather Dataset:")

                df = pd.read_table("../Data/weather.txt")
                # print(df)

                # Split fearures and target
                X, y = pre_processing(df)

                nb_clf = NaiveBayes()
                nb_clf.fit(X, y)

                print("Train Accuracy: {}".format(
                    accuracy_score(y, nb_clf.predict(X))))

                # Query 1:
                query = np.array([['Rainy', 'Mild', 'Normal', 't']])
                print("Query 1:- {} ---> {}".format(query, nb_clf.predict(query)))

                # Query 2:
                query = np.array([['Overcast', 'Cool', 'Normal', 't']])
                print("Query 2:- {} ---> {}".format(query, nb_clf.predict(query)))

                # Query 3:
                query = np.array([['Sunny', 'Hot', 'High', 't']])
                print("Query 3:- {} ---> {}".format(query, nb_clf.predict(query)))
# --------------------------------------------------------------------------------------

        if rad5 == "k-NN classifier":

            arr = []
            arr = df["variety"].unique()

            x = df.iloc[:, [0, 1, 2, 3]].values
            y = df.iloc[:, 4].values

            # Splitting the dataset into training and test set.

            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.25, random_state=0)

            # feature Scaling

            st_x = StandardScaler()
            x_train = st_x.fit_transform(x_train)
            x_test = st_x.transform(x_test)

            print(y_train)
            print("train")
            print(y_test)

            point = x_train[0]
            distance_points = []
            print(np.linalg.norm(point - x_train[2]))
            j = 0
            for i in range(len(x_train)):
                # distance_points[i] = np.linalg.norm(point - x_train[j])
                temp = point - x_train[i]
                sum = np.dot(temp.T, temp)
                distance_points.append(np.sqrt(sum))

            for i in range(len(x_train), len(df)):
                distance_points.append(1000)

            classifier = KNeighborsClassifier(
                n_neighbors=5, metric='minkowski', p=2)
            classifier.fit(x_train, y_train)

            y_pred = classifier.predict(x_test)

            m = confusion_matrix(y_test, y_pred)

            # print(point)
            # print(x_train[2])

            df["distance"] = distance_points

            x = df.iloc[:, [0, 1, 2, 3, 5]].values
            y = df.iloc[:, 4].values

            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.25)

            st_x = StandardScaler()
            x_train = st_x.fit_transform(x_train)
            x_test = st_x.transform(x_test)

            df = df.sort_values(by=['distance'])
            # print(df1)
            # st.subheader("After sorting")
            # st.dataframe(df)

            k_value = st.selectbox("k-value", [1, 3, 5, 7])

            df_first_k = df.head(k_value+1)

            st.dataframe(df_first_k)

            nearest_neighbour = (df_first_k['variety']).mode()
            st.subheader("Nearest " + str(k_value) +
                         " neighbours ", nearest_neighbour)

        if rad5 == "Artificial Neural Network Classifier":
            def Sigmoid(Z):
                return 1/(1+np.exp(-Z))

            def Relu(Z):
                return np.maximum(0, Z)

            def dRelu2(dZ, Z):
                dZ[Z <= 0] = 0
                return dZ

            def dRelu(x):
                x[x <= 0] = 0
                x[x > 0] = 1
                return x

            def dSigmoid(Z):
                s = 1/(1+np.exp(-Z))
                dZ = s * (1-s)
                return dZ

            class dlnet:
                def __init__(self, x, y):
                    self.debug = 0
                    self.X = x
                    self.Y = y
                    self.Yh = np.zeros((1, self.Y.shape[1]))
                    self.L = 2
                    self.dims = [9, 15, 1]
                    self.param = {}
                    self.ch = {}
                    self.grad = {}
                    self.loss = []
                    self.lr = 0.003
                    self.sam = self.Y.shape[1]
                    self.threshold = 0.5

                def nInit(self):
                    np.random.seed(1)
                    self.param['W1'] = np.random.randn(
                        self.dims[1], self.dims[0]) / np.sqrt(self.dims[0])
                    self.param['b1'] = np.zeros((self.dims[1], 1))
                    self.param['W2'] = np.random.randn(
                        self.dims[2], self.dims[1]) / np.sqrt(self.dims[1])
                    self.param['b2'] = np.zeros((self.dims[2], 1))
                    return

                def forward(self):
                    Z1 = self.param['W1'].dot(self.X) + self.param['b1']
                    A1 = Relu(Z1)
                    self.ch['Z1'], self.ch['A1'] = Z1, A1

                    Z2 = self.param['W2'].dot(A1) + self.param['b2']
                    A2 = Sigmoid(Z2)
                    self.ch['Z2'], self.ch['A2'] = Z2, A2

                    self.Yh = A2
                    loss = self.nloss(A2)
                    return self.Yh, loss

                def nloss(self, Yh):
                    loss = (1./self.sam) * (-np.dot(self.Y,
                                                    np.log(Yh).T) - np.dot(1-self.Y, np.log(1-Yh).T))
                    return loss

                def backward(self):
                    dLoss_Yh = - (np.divide(self.Y, self.Yh) -
                                  np.divide(1 - self.Y, 1 - self.Yh))

                    dLoss_Z2 = dLoss_Yh * dSigmoid(self.ch['Z2'])
                    dLoss_A1 = np.dot(self.param["W2"].T, dLoss_Z2)
                    dLoss_W2 = 1./self.ch['A1'].shape[1] * \
                        np.dot(dLoss_Z2, self.ch['A1'].T)
                    dLoss_b2 = 1./self.ch['A1'].shape[1] * \
                        np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1], 1]))

                    dLoss_Z1 = dLoss_A1 * dRelu(self.ch['Z1'])
                    dLoss_A0 = np.dot(self.param["W1"].T, dLoss_Z1)
                    dLoss_W1 = 1./self.X.shape[1] * np.dot(dLoss_Z1, self.X.T)
                    dLoss_b1 = 1. / \
                        self.X.shape[1] * np.dot(dLoss_Z1,
                                                 np.ones([dLoss_Z1.shape[1], 1]))

                    self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1
                    self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1
                    self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2
                    self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2

                    return

                def pred(self, x, y):
                    self.X = x
                    self.Y = y
                    comp = np.zeros((1, x.shape[1]))
                    pred, loss = self.forward()

                    for i in range(0, pred.shape[1]):
                        if pred[0, i] > self.threshold:
                            comp[0, i] = 1
                        else:
                            comp[0, i] = 0

                    print("Acc: " + str(np.sum((comp == y)/x.shape[1])))

                    return comp

                def gd(self, X, Y, iter=3000):
                    np.random.seed(1)

                    self.nInit()

                    for i in range(0, iter):
                        Yh, loss = self.forward()
                        self.backward()

                        if i % 500 == 0:
                            # print("Cost after iteration %i: %f" % (i, loss))
                            self.loss.append(loss)

                    plt.plot(np.squeeze(self.loss))
                    plt.ylabel('Loss')
                    plt.xlabel('Iter')
                    plt.title("Lr =" + str(self.lr))
                    st.pyplot(plt)

                    return

            def plotCf(a, b, t):
                cf = confusion_matrix(a, b)
                st.dataframe(cf)
                plt.imshow(cf, cmap=plt.cm.Blues, interpolation='nearest')
                plt.colorbar()
                plt.title(t)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                tick_marks = np.arange(len(set(a)))  # length of classes
                class_labels = ['0', '1']
                plt.xticks(tick_marks, class_labels)
                plt.yticks(tick_marks, class_labels)
                thresh = cf.max() / 2.
                for i, j in itertools.product(range(cf.shape[0]), range(cf.shape[1])):
                    plt.text(j, i, format(
                        cf[i, j], 'd'), horizontalalignment='center', color='white' if cf[i, j] > thresh else 'black')

                st.pyplot(plt)

                # print(type(data))
            df = pd.read_csv(
                "D:\\ALL_sem\\Sem7\\Data_Mining_lab\\breast-cancer-wisconsin1.csv", header=None)
            df = df[~df[6].isin(['?'])]
            df = df.astype(float)
            df.iloc[:, 10].replace(2, 0, inplace=True)
            df.iloc[:, 10].replace(4, 1, inplace=True)

            df.head(3)
            scaled_df = df
            names = df.columns[0:10]
            scaler = MinMaxScaler()
            scaled_df = scaler.fit_transform(df.iloc[:, 0:10])
            scaled_df = pd.DataFrame(scaled_df, columns=names)
            x = scaled_df.iloc[0:500, 1:10].values.transpose()
            y = df.iloc[0:500, 10:].values.transpose()

            xval = scaled_df.iloc[501:683, 1:10].values.transpose()
            yval = df.iloc[501:683, 10:].values.transpose()

            print(df.shape, x.shape, y.shape, xval.shape, yval.shape)

            nn = dlnet(x, y)
            nn.lr = 0.07
            nn.dims = [9, 15, 1]
            nn.gd(x, y, iter=40000)
            pred_train = nn.pred(x, y)
            pred_test = nn.pred(xval, yval)
            print("Pred test is:", pred_test)
            st.write("Accuracy:", str(
                np.sum((pred_test == yval)/xval.shape[1])))
            nn.threshold = 0.5

            nn.X, nn.Y = x, y
            target = np.around(np.squeeze(y), decimals=0).astype(np.int)
            predicted = np.around(np.squeeze(nn.pred(x, y)),
                                  decimals=0).astype(np.int)
            plotCf(target, predicted, 'Cf Training Set')

            nn.X, nn.Y = xval, yval
            target = np.around(np.squeeze(yval), decimals=0).astype(np.int)
            predicted = np.around(np.squeeze(
                nn.pred(xval, yval)), decimals=0).astype(np.int)
            plotCf(target, predicted, 'Cf Validation Set')
            nn.X, nn.Y = xval, yval
            yvalh, loss = nn.forward()
            print("\ny", np.around(yval[:, 0:50, ], decimals=0).astype(np.int))
            print("\nyh", np.around(
                yvalh[:, 0:50, ], decimals=0).astype(np.int), "\n")

if genre == "Assignment-6":
    st.title('DM Assignment-6')

    Clustering_type = st.radio("CLUSTERING TYPES", (
        'Hierarchical clustering - AGNES & DIANA', 'k-Means', 'k-Medoids (PAM)', 'DBSCAN'))

    if Clustering_type == "Hierarchical clustering - AGNES & DIANA":
        # Hierarchical Clustering

        # Importing the libraries
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd

        uploaded_file = st.file_uploader(label="Choose a file",
                                         type=['csv', 'xlsx'])

        if uploaded_file is not None:
            print(uploaded_file)
            dataset = pd.read_csv(uploaded_file)
            st.dataframe(dataset)
            # numeric_columns = list(df.columns)

            st.dataframe(dataset)

            st.subheader("Hierarchical clustering - AGNES & DIANA")

            # Importing the dataset
            # dataset = df
            X = dataset.iloc[:, [3, 4]].values
            # y = dataset.iloc[:, 3].values

            # Splitting the dataset into the Training set and Test set
            """from sklearn.cross_validation import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

            # Feature Scaling
            """from sklearn.preprocessing import StandardScaler
                sc_X = StandardScaler()
                X_train = sc_X.fit_transform(X_train)
                X_test = sc_X.transform(X_test)
                sc_y = StandardScaler()
                y_train = sc_y.fit_transform(y_train)"""

            # Using the dendrogram to find the optimal number of clusters
            import scipy.cluster.hierarchy as sch
            fig = plt.figure(figsize=(6, 6))
            dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
            plt.title('Dendrogram')
            plt.xlabel('Customers')
            plt.ylabel('Euclidean distances')
            plt.show()
            st.pyplot(fig)

            # Fitting Hierarchical Clustering to the dataset
            from sklearn.cluster import AgglomerativeClustering
            hc = AgglomerativeClustering(
                n_clusters=5, affinity='euclidean', linkage='ward')
            y_hc = hc.fit_predict(X)

            # Visualising the clusters
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1],
                        s=100, c='red', label='Cluster 1')
            plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1],
                        s=100, c='blue', label='Cluster 2')
            plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1],
                        s=100, c='green', label='Cluster 3')
            plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1],
                        s=100, c='cyan', label='Cluster 4')
            plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1],
                        s=100, c='magenta', label='Cluster 5')
            plt.title('Clusters of customers')
            plt.xlabel('Annual Income (k$)')
            plt.ylabel('Spending Score (1-100)')
            plt.legend()
            plt.show()
            st.pyplot(fig)

    # if Clustering_type == "k-Means":
    #     # K-Means Clustering

    #     # Importing the libraries
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     import pandas as pd

    #     uploaded_file = st.file_uploader(label="Choose a file",
    #                                      type=['csv', 'xlsx'])

    #     if uploaded_file is not None:
    #         print(uploaded_file)
    #         dataset = pd.read_csv(uploaded_file)
    #         st.dataframe(dataset)
    #         # numeric_columns = list(df.columns)

    #         st.dataframe(dataset)

            # Importing the dataset

            st.subheader("K-Mean Clustering")
            dataset = pd.read_csv('Mall_Customers.csv')
            X = dataset.iloc[:, [3, 4]].values
            # y = dataset.iloc[:, 3].values

            # Splitting the dataset into the Training set and Test set
            """from sklearn.cross_validation import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

            # Feature Scaling
            """from sklearn.preprocessing import StandardScaler
            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)
            sc_y = StandardScaler()
            y_train = sc_y.fit_transform(y_train)"""

            # Using the elbow method to find the optimal number of clusters
            from sklearn.cluster import KMeans
            wcss = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init='k-means++',
                                random_state=42)
                kmeans.fit(X)
                wcss.append(kmeans.inertia_)
            fig = plt.figure(figsize=(6, 6))
            plt.plot(range(1, 11), wcss)
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            plt.show()
            st.pyplot(fig)

            # Fitting K-Means to the dataset
            kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
            y_kmeans = kmeans.fit_predict(X)

            # Visualising the clusters
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1],
                        s=100, c='red', label='Cluster 1')
            plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1],
                        s=100, c='blue', label='Cluster 2')
            plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],
                        s=100, c='green', label='Cluster 3')
            plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1],
                        s=100, c='cyan', label='Cluster 4')
            plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1],
                        s=100, c='magenta', label='Cluster 5')
            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
                        :, 1], s=300, c='yellow', label='Centroids')
            plt.title('Clusters of customers')
            plt.xlabel('Annual Income (k$)')
            plt.ylabel('Spending Score (1-100)')
            plt.legend()
            plt.show()
            st.pyplot(fig)

    # if Clustering_type == "k-Medoids (PAM)":

    # if Clustering_type == "DBSCAN":
        # Let's take another example
        # Step 1-

        # Here, I am creating a dataset with only two features so that we can visualize it easily.
        # For creating the dataset I have created a function PointsInCircum
        # which takes the radius and number of data points as arguments
        # and returns an array of data points which when plotted forms a circle.
        # We do this with the help of sin and cosine curves.

        st.subheader("DBSCAN Clustering")
        np.random.seed(42)
        # Function for creating datapoints in the form of a circle

        def PointsInCircum(r, n=100):
            return [(math.cos(2*math.pi/n*x)*r+np.random.normal(-30, 30), math.sin(2*math.pi/n*x)*r+np.random.normal(-30, 30)) for x in range(1, n+1)]

        # Step 2-
        # One circle won’t be sufficient to see the clustering ability of DBSCAN.
        # Therefore, I have created three concentric circles of different radii.
        # Also, I will add noise to this data so that we can see how different types of clustering algorithms deals
        # with noise.

        # Creating data points in the form of a circle
        df = pd.DataFrame(PointsInCircum(500, 1000))
        df = df.append(PointsInCircum(300, 700))
        df = df.append(PointsInCircum(100, 300))

        # Adding noise to the dataset
        df = df.append([(np.random.randint(-600, 600),
                       np.random.randint(-600, 600)) for i in range(300)])

        print(df.head())

        # Step 3
        # Let’s plot these data points and see how they look in the feature space.
        # Here, I use the scatter plot for plotting these data points. Use the following syntax:
        fig = plt.figure(figsize=(6, 6))

        plt.scatter(df[0], df[1], s=15, color='grey')
        plt.title('Dataset', fontsize=10)
        plt.xlabel('Feature 1', fontsize=10)
        plt.ylabel('Feature 2', fontsize=10)
        plt.show()
        st.pyplot(fig)
        from sklearn.cluster import DBSCAN

        dbscan = DBSCAN()
        dbscan.fit(df[[0, 1]])

        # DB Scan Plot
        df['DBSCAN_labels'] = dbscan.labels_

        # Plotting resulting clusters
        fig = plt.figure(figsize=(5, 5))
        plt.scatter(df[0], df[1], c=df['DBSCAN_labels'], s=15)
        plt.title('DBSCAN Clustering', fontsize=10)
        plt.xlabel('Feature 1', fontsize=10)
        plt.ylabel('Feature 2', fontsize=10)
        plt.show()
        st.pyplot(fig)


if genre == "Assignment-7":
    st.title("Assignment-7")
    # uploaded_file = st.file_uploader(label="Choose a file",
    #                                  type=['csv', 'xlsx'])

    # if uploaded_file is not None:
    #     print(uploaded_file)
    #     df = pd.read_csv(uploaded_file)
    #     st.dataframe(df)
    #     numeric_columns = list(df.columns)

    #     st.dataframe(df)
    # st.subheader(
    #     "Implementatio of Apriori algorithm for generating association rules")
    # import numpy as np
    # import matplotlib.pyplot as plt
    # import pandas as pd
    # from apyori import apriori

    # # Importing Dataset from system
    # store_data = pd.read_csv(
    #     'D:\ALL_sem\Sem7\Data_Mining_lab\Assignmnet07\store_data.csv')

    # # store_data.head()  # to check the header

    # # keeping header as None
    # # store_data = pd.read_csv(
    # # 'D:\ALL_sem\Sem7\Data_Mining_lab\Assignmnet07\store_data.csv', header='none')
    # num_record = len(store_data)
    # print(num_record)
    # st.write("Number of records :" + str(num_record))

    # record = []
    # for i in range(0, num_record):
    #     record.append([str(store_data.values[i, j]) for j in range(0, 5)])

    # # print(record)
    # print('hello')

    # association_rules = apriori(
    #     record, min_support=0.0053, min_confidence=0.2, min_lift=3, min_length=2)
    # print('hello')
    # association_results = list(association_rules)
    # print(len(association_results))  # to check the Total Number of Rules mined
    # st.write("Total Number of Association rules mined : " +
    #          str(len(association_results)))
    # # to print the first item the association_rules list to see the first rule
    # print(association_results[0])
    # st.write("Association results[0] : ")
    # st.write(association_results[0])

    # for item in association_results:
    #     #  to display the rule, the support, the confidence, and lift for each rule in a more clear way:

    #     # first index of the inner list
    #     # Contains base item and add item
    #     pair = item[0]
    #     items = [x for x in pair]
    #     print("Rule: " + items[0] + " -> " + items[1])
    #     st.write("Rule: " + items[0] + " -> " + items[1])

    #     # second index of the inner list
    #     print("Support: " + str(item[1]))
    #     st.write("Support: " + str(item[1]))

    #     # third index of the list located at 0th
    #     # of the third index of the inner list

    #     print("Confidence: " + str(item[2][0][2]))
    #     st.write("Confidence: " + str(item[2][0][2]))
    #     print("Lift: " + str(item[2][0][3]))
    #     st.write("Lift: " + str(item[2][0][3]))
    #     print("=====================================")
    #     st.write("=====================================")

    import streamlit as st
    import numpy as np
    import pandas as pd
    import math
    import matplotlib.pyplot as plt
    import collections
    from scipy.cluster import hierarchy
    from sklearn import datasets
    from random import randint
    import random
    import plotly.express as px
    import altair as alt
    import seaborn as sns

    uploaded_file = st.file_uploader(label="Choose a file",
                                     type=['csv', 'xlsx'])

    if uploaded_file is not None:
        print(uploaded_file)
        dataset = pd.read_csv(uploaded_file)
        st.dataframe(dataset)
        # numeric_columns = list(df.columns)

        st.dataframe(dataset)

        def app(dataset):

            st.header("Assignment 7")

            url = 'https://raw.githubusercontent.com/Datta00007/DataSets/main/house-votes-84.data.csv'
            df = pd.read_csv(url)

            # st.write(df[:5])
            st.subheader("US Congress House votes dataset")
            st.dataframe(df)
            d = pd.DataFrame(df)
            data = d
            d.head()

            df_rows = d.to_numpy().tolist()

            cols = []
            for i in data.columns:
                cols.append(i)
            st.write("Attributes:", len(cols))
            # st.write(cols)
            newDataSet = []
            # st.write(len(df_rows))
            i, cnt = 0, 0
            for row in df_rows:
                i += 1
                if '?' in row:
                    continue
                else:
                    lst = []
                    cnt += 1
                    for k in range(1, len(row)):
                        if row[k] == 'y':
                            lst.append(cols[k])
                    newDataSet.append(lst)

                    # st.write(row)
                    # st.write("--------------")
            # st.write(cnt)
            # st.write(newDataSet)
            # newDataSet.drop()

            data = []

            for i in range(len(newDataSet)):
                # data[i] = newDataSet[i]
                data.append([i, newDataSet[i]])

            # st.write(data)

            # extract distinct items

            init = []
            for i in data:
                for q in i[1]:
                    if(q not in init):
                        init.append(q)
            init = sorted(init)

            st.write("Init:", len(init))

            # st.write(init)

            sp = st.number_input("Enter confidence threshold",
                                 min_value=0.0, max_value=1.0, value=1.0, format="%.1f")
            s = int(sp*len(init))
            s

            from collections import Counter
            c = Counter()
            for i in init:
                for d in data:
                    if(i in d[1]):
                        c[i] += 1
            # st.write("C1:")
            for i in c:
                pass
                # st.write(str([i])+": "+str(c[i]))
            # st.write()
            l = Counter()
            for i in c:
                if(c[i] >= s):
                    l[frozenset([i])] += c[i]
            # st.write("L1:")
            for i in l:
                pass
                # st.write(str(list(i))+": "+str(l[i]))
            # st.write()
            pl = l
            pos = 1
            for count in range(2, 1000):
                nc = set()
                temp = list(l)
                for i in range(0, len(temp)):
                    for j in range(i+1, len(temp)):
                        t = temp[i].union(temp[j])
                        if(len(t) == count):
                            nc.add(temp[i].union(temp[j]))
                nc = list(nc)
                c = Counter()
                for i in nc:
                    c[i] = 0
                    for q in data:
                        temp = set(q[1])
                        if(i.issubset(temp)):
                            c[i] += 1
                # st.write("C"+str(count)+":")
                for i in c:
                    pass
                    # st.write(str(list(i))+": "+str(c[i]))
                # st.write()
                l = Counter()
                for i in c:
                    if(c[i] >= s):
                        l[i] += c[i]
                # st.write("L"+str(count)+":")
                for i in l:
                    pass
                    # st.write(str(list(i))+": "+str(l[i]))
                # st.write()
                if(len(l) == 0):
                    break
                pl = l
                pos = count
            st.write("Result: ")
            st.write("L"+str(pos)+":")
            for i in pl:
                st.write(str(list(i))+" : "+str(pl[i]))
            st.write()

            st.subheader("Rules Generation")
            from itertools import combinations
            for l in pl:
                cnt = 0
                c = [frozenset(q) for q in combinations(l, len(l)-1)]
                mmax = 0
                for a in c:
                    b = l-a
                    ab = l
                    sab = 0
                    sa = 0
                    sb = 0
                    for q in data:
                        temp = set(q[1])
                        if(a.issubset(temp)):
                            sa += 1
                        if(b.issubset(temp)):
                            sb += 1
                        if(ab.issubset(temp)):
                            sab += 1
                    temp = sab/sa*100
                    if(temp > mmax):
                        mmax = temp
                    temp = sab/sb*100
                    if(temp > mmax):
                        mmax = temp
                    cnt += 1
                    sqrtab = math.sqrt(sa*sb)
                    st.write(str(cnt) + "  :  " + str(list(a))+" -> "+str(list(b))+" = Confidence : "+str(sab/sa*100)+"%"+" |  Lift: "+str((sab/sa*sb)/100)+" | All confidence measure = " + str(
                        (sab/max(sa, sb))*100) + " | max_confidence_measure = " + str(max(sa, sb)/100) + " | Kulczynski measure= " + str((sa+sb)/2/100) + " | Cosine measure = " + str(sab/sqrtab))
                    cnt += 1
                    st.write(str(cnt) + "  :  " + str(list(b))+" -> "+str(list(a))+" = Confidence : "+str(sab/sb*100)+"%"+" |  Lift: "+str((sab/sa*sb)/100)+" | All confidence measure = " + str(
                        (sab/max(sa, sb))*100) + " | max_confidence_measure = " + str(max(sa, sb)/100) + " | Kulczynski measure= " + str((sa+sb)/2/100) + " | Cosine measure = " + str(sab/sqrtab))
                curr = 1
                st.write("choosing:")
                mmax = sp*100
                for a in c:
                    b = l-a
                    ab = l
                    sab = 0
                    sa = 0
                    sb = 0
                    for q in data:
                        temp = set(q[1])
                        if(a.issubset(temp)):
                            sa += 1
                        if(b.issubset(temp)):
                            sb += 1
                        if(ab.issubset(temp)):
                            sab += 1
                    temp = sab/sa*100
                    if(temp >= mmax):
                        st.write(curr)
                    curr += 1
                    temp = sab/sb*100
                    if(temp >= mmax):
                        st.write(curr)
                    curr += 1
                st.write()
                st.write()
        app(dataset)
if genre == "Assignment-8":

    st.title('DM Assignment-8')
    # Import libraries
    from urllib.request import urljoin
    from bs4 import BeautifulSoup
    import requests
    from urllib.request import urlparse
    import operator

    import streamlit as st
    import numpy as np
    import pandas as pd
    import math
    import matplotlib.pyplot as plt
    import collections
    from scipy.cluster import hierarchy
    from sklearn import datasets
    from random import randint
    import random
    import plotly.express as px
    import altair as alt
    import seaborn as sns

    uploaded_file = st.file_uploader(label="Choose a file",
                                     type=['csv', 'xlsx', 'txt'])

    if uploaded_file is not None:
        print(uploaded_file)
        dataset = pd.read_csv(uploaded_file)
        st.dataframe(dataset)

        def app(dataset):

            st.header("Assignment 8")

            def printf(url):
                st.markdown(
                    f'<p style="color:#000;font:lucida;font-size:20px;">{url}</p>', unsafe_allow_html=True)

            operation = st.selectbox(
                "Operation", ["WebCrawler", 'PageRank', 'HITS'])

            if operation == "WebCrawler":
                input_url = st.text_input("Paste URL here")
                # Set for storing urls with same domain
                links_intern = set()
                # input_url = "http://www.walchandsangli.ac.in/"
                depth = st.number_input(
                    "Enter depth (less than 5)", value=1, max_value=5, min_value=0)
                # depth = 5

                # Set for storing urls with different domain
                links_extern = set()

                # Method for crawling a url at next level

                def level_crawler(input_url):
                    temp_urls = set()
                    current_url_domain = urlparse(input_url).netloc

                    # Creates beautiful soup object to extract html tags
                    beautiful_soup_object = BeautifulSoup(
                        requests.get(input_url).content, "lxml")

                    # Access all anchor tags from input
                    # url page and divide them into internal
                    # and external categories
                    idx = 0
                    for anchor in beautiful_soup_object.findAll("a"):
                        href = anchor.attrs.get("href")
                        if(href != "" or href != None):
                            href = urljoin(input_url, href)
                            href_parsed = urlparse(href)
                            href = href_parsed.scheme
                            href += "://"
                            href += href_parsed.netloc
                            href += href_parsed.path
                            final_parsed_href = urlparse(href)
                            is_valid = bool(final_parsed_href.scheme) and bool(
                                final_parsed_href.netloc)
                            if is_valid:
                                if current_url_domain not in href and href not in links_extern:
                                    idx += 1
                                    st.write(f"link {idx} - {href}")
                                    links_extern.add(href)
                                if current_url_domain in href and href not in links_intern:
                                    idx += 1
                                    st.write(f"link {idx} - {href}")
                                    links_intern.add(href)
                                    temp_urls.add(href)
                    return temp_urls

                def crawl(input_url, depth):

                    if(depth == 0):
                        st.write("Page - {}".format(input_url))

                    elif(depth == 1):
                        level_crawler(input_url)

                    else:
                        # We have used a BFS approach
                        # considering the structure as
                        # a tree. It uses a queue based
                        # approach to traverse
                        # links upto a particular depth.
                        queue = []
                        queue.append(input_url)
                        for j in range(depth):
                            st.subheader(f"Level {j} -")
                            idx = 0
                            for count in range(len(queue)):
                                idx += 1
                                url = queue.pop(0)
                                printf(f"Page {idx} : {url} ")
                                urls = level_crawler(url)
                                for i in urls:
                                    queue.append(i)

                if st.button("Crawl"):
                    crawl(input_url, depth)

            if operation == "PageRank":
                st.dataframe(dataset.head(1000), width=1000, height=500)

                # Adjacency Matrix representation in Python

                class Graph(object):

                    # Initialize the matrix
                    def __init__(self, size):
                        self.adjMatrix = []
                        self.inbound = dict()
                        self.outbound = dict()
                        self.pagerank = dict()
                        self.vertex = set()
                        self.cnt = 0
                        # for i in range(size+1):
                        #     self.adjMatrix.append([0 for i in range(size+1)])
                        self.size = size

                    # Add edges
                    def add_edge(self, v1, v2):
                        if v1 == v2:
                            printf("Same vertex %d and %d" % (v1, v2))
                        # self.adjMatrix[v1][v2] = 1
                        self.vertex.add(v1)
                        self.vertex.add(v2)
                        if self.inbound.get(v2, -1) == -1:
                            self.inbound[v2] = [v1]
                        else:
                            self.inbound[v2].append(v1)
                        if self.outbound.get(v1, -1) == -1:
                            self.outbound[v1] = [v2]
                        else:
                            self.outbound[v1].append(v2)

                        # self.adjMatrix[v2][v1] = 1

                    # Remove edges
                    # def remove_edge(self, v1, v2):
                    #     if self.adjMatrix[v1][v2] == 0:
                    #         print("No edge between %d and %d" % (v1, v2))
                    #         return
                    #     self.adjMatrix[v1][v2] = 0
                    #     self.adjMatrix[v2][v1] = 0

                    def __len__(self):
                        return self.size

                    # Print the matrix
                    def print_matrix(self):
                        # if self.size < 1000:
                        #     for row in self.adjMatrix:
                        #         for val in row:
                        #             printf('{:4}'.format(val), end="")
                        #         printf("\n")
                        #     printf("Inbound:")
                        #     st.write(self.inbound)

                        #     printf("Outbound:")
                        #     st.write(self.outbound)
                        # else:
                        pass

                    def pageRank(self):
                        self.cnt = 0
                        if len(self.pagerank) == 0:
                            for i in self.vertex:
                                self.pagerank[i] = 1/self.size
                        prevrank = self.pagerank
                        # print(self.pagerank)
                        for i in self.vertex:
                            pagesum = 0.0
                            inb = self.inbound.get(i, -1)
                            if inb == -1:
                                continue
                            for j in inb:
                                pagesum += (self.pagerank[j] /
                                            len(self.outbound[j]))
                            self.pagerank[i] = pagesum
                            if (prevrank[i]-self.pagerank[i]) <= 0.1:
                                self.cnt += 1

                    def printRank(self):
                        printf(self.pagerank)

                    def arrangeRank(self):
                        sorted_rank = dict(
                            sorted(self.pagerank.items(), key=operator.itemgetter(1), reverse=True))
                        # printf(sorted_rank)
                        printf("PageRank Sorted : "+str(len(sorted_rank)))
                        i = 1
                        printf(f"Rank ___ Node ________ PageRank Score")
                        for key, rank in sorted_rank.items():
                            if i == 11:
                                break
                            printf(f"{i} _____ {key} ________ {rank}")
                            i += 1

                        # st.dataframe(sorted_rank)

                def main():
                    g = Graph(7)
                    input_list = []

                    d = 0.5
                    for i in range(len(dataset)):
                        input_list.append(
                            [dataset.loc[i, 'fromNode'], dataset.loc[i, 'toNode']])
                        g.add_edge(
                            dataset.loc[i, 'fromNode'], dataset.loc[i, 'toNode'])
                    size = len(g.vertex)
                    if size <= 10000:
                        adj_matrix = np.zeros([size+1, size+1])

                        for i in input_list:
                            adj_matrix[i[0]][i[1]] = 1

                        st.subheader("Adjecency Matrix")
                        st.dataframe(adj_matrix, width=1000, height=500)

                    printf("Total Node:"+str(len(g.vertex)))
                    printf("Total Edges: "+str(len(input_list)))
                    # for i in input_list:

                    # g.print_matrix()

                    i = 0
                    while i < 5:
                        if g.cnt == g.size:
                            break
                        g.pageRank()
                        i += 1
                    # g.printRank()
                    g.arrangeRank()

                main()

            if operation == "HITS":
                input_list = []

                st.subheader("Dataset")
                st.dataframe(dataset.head(1000), width=1000, height=500)
                vertex = set()
                for i in range(len(dataset)):
                    input_list.append(
                        [dataset.loc[i, 'fromNode'], dataset.loc[i, 'toNode']])
                    vertex.add(dataset.loc[i, 'fromNode'])
                    vertex.add(dataset.loc[i, 'toNode'])
                size = len(vertex)
                adj_matrix = np.zeros([size+1, size+1])

                for i in input_list:
                    adj_matrix[i[0]][i[1]] = 1

                printf("No of Nodes: "+str(size))
                printf("No of Edges: "+str(len(dataset)))
                st.subheader("Adjecency Matrix")
                st.dataframe(adj_matrix, width=1000, height=500)
                A = adj_matrix
                # st.dataframe(A)
                At = adj_matrix.transpose()
                st.subheader("Transpose of Adj matrix")
                st.dataframe(At)

                u = [1 for i in range(size+1)]
                v = np.matrix([])
                for i in range(5):
                    v = np.dot(At, u)
                    u = np.dot(A, v)

                # u.sort(reverse=True)
                hubdict = dict()
                for i in range(len(u)):
                    hubdict[i] = u[i]

                authdict = dict()
                for i in range(len(v)):
                    authdict[i] = v[i]

                printf("Hub weight matrix (U)")
                st.dataframe(u)
                printf("Hub weight vector (V)")
                st.dataframe(v)
                hubdict = dict(
                    sorted(hubdict.items(), key=operator.itemgetter(1), reverse=True))
                authdict = dict(
                    sorted(authdict.items(), key=operator.itemgetter(1), reverse=True))
                # printf(sorted_rank)
                printf("HubPages : ")
                i = 1
                printf(f"Rank ___ Node ________ Hubs score")
                for key, rank in hubdict.items():
                    if i == 11:
                        break
                    printf(f"{i} _____ {key} ________ {rank}")
                    i += 1

                printf("Authoritative Pages : ")
                i = 1
                printf(f"Rank ___ Node ________ Auth score")
                for key, rank in authdict.items():
                    if i == 11:
                        break
                    printf(f"{i} _____ {key} ________ {rank}")
                    i += 1

                # u = sorted(u, reverse=True)
                # printf("Hub weight matrix (U)")
                # st.dataframe(u)
                # v = sorted(v, reverse=True)
                # printf("Hub weight vector Authority (V)")
                # st.dataframe(v[:11])
        app(dataset)

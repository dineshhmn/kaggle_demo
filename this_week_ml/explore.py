import os
import math
import random
import pandas as pd
import jaydebeapi as jp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import intercluster_distance


def get_data():
    """
    Simple get data
    :return: pandas df of data
    """
    return pd.read_csv('../data.csv')

def get_boxplots(df):
    """
    We know all data is numeric in nature, lets plot box plots for each
    Groups of columns per chart can be modified here
    :return: None, plots displayed
    :return: None

    NOTE: Plot skips ID column by using max(5*x, 1)
    """
    number_iter = math.ceil(len(df.columns)/5) # number of columns per chart
    x = 0
    while x < number_iter:
        df_orig[df_orig.columns[max(5*x,1):5*(x+1)]].plot(kind='box')
        df_orig[df_orig.columns[max(5*x,1):5*(x+1)]].plot(kind='kde')
        x+=1
    plt.show()

def get_means(df):
    """
    Lets also check the spreads of the data
    :param df: pandas df to show plots
    :return: None, plots displayed

    NOTE: Plot skips ID column by using list(df.columns)[1:]
    """
    mean_lst = []
    std_lst = []
    for col in list(df.columns)[1:]:
        m = df[col].mean()
        s = df[col].std()
        mean_lst.append(m)
        std_lst.append(s)
    plt.scatter(range(len(mean_lst)), mean_lst, marker="+")
    for x in range(len(std_lst)): # Just to get column numbers
        plt.text(x, -1, x, color="red", fontsize="8")
    plt.scatter(range(len(std_lst)), std_lst,marker="*")
    plt.scatter(range(len(std_lst)), [x*2 for x in std_lst], marker="o")
    plt.show()

def get_base_clusters(df, ncl=8):
    """
    :param df: data with columns
    :return: submission for basic Kmeans

    Also plot kmeans output
    """
    ndf = df.iloc[:, 1:].to_numpy()
    km = KMeans(n_clusters = ncl, random_state=0)
    km_out = km.fit_predict(ndf)
    count_dct = {}
    for z in km_out:
        if z not in count_dct.keys():
            count_dct[z] = 0
        else: count_dct[z] += 1
    plt.bar(count_dct.keys(), count_dct.values())
    plt.show()
    intercluster_distance(km, ndf, random_state=0)

    # Join output from results to original df by ID/index
    submission_df = pd.DataFrame(df.id).join(pd.DataFrame(km_out, columns=['Predicted']), how='inner')
    return submission_df

def pca_check(df):
    """
    :param lst: df filtered
    :return: None
    Some features can be dropped
    """
    ndf = df.iloc[:,1:].to_numpy()
    pca = PCA()
    pca.fit(ndf)
    plt.plot(pca.explained_variance_ratio_)
    plt.show()

def kernel_pca(df):
    ndf = df.iloc[:,1:].to_numpy()
    pca = KernelPCA(kernel='cosine')
    pca.fit(ndf)
    plt.plot(pca.explained_variance_ratio_)
    plt.show()

if __name__ == "__main__":
    random.seed(10)
    df_orig = get_data()
    get_means(df_orig)

    # Step Zero: base submission
    df_sub0 = get_base_clusters(df_orig)

    # SUBMISSION
    # df_sub0.to_csv("sub.csv", index=False)
    # # Base Submission has a score of 0.23237, how to improve?
    #
    # # Step One: explore data
    get_boxplots(df_orig)
    #
    # # we can see that columns f_00 through f_06,
    # # and f_14 through f_21 are redundant with very little variance in them
    # # Removing those columns
    df_fil = df_orig.iloc[:, [0,1,8,9,10,11,12,13,14,23,24,25,26,27,28,29]]
    # # Step Two: use simple clustering methods
    # Get the same score from this tooo
    df_sub = get_base_clusters(df_fil)

    # SUBMISSION
    # df_sub.to_csv("sub_0.csv", index=False)
    #
    # # Kmeans has inertia of 8.25M, lets tune this to get as close as possible
    # # Inertia is not a normalized metric: we just know that lower values are better and zero is optimal.
    # # But in very high-dimensional spaces, Euclidean distances tend to become inflated
    # # (this is an instance of the so-called “curse of dimensionality”).
    # # Running a dimensionality reduction algorithm such as Principal component analysis (PCA) prior to k-means
    # # clustering can alleviate this problem and speed up the computations.
    #
    # # Lets try to apply some dimension reduction techniques
    pca_check(df_fil)
    # # From PCA elbow we can see that many features can be dropped
    df_fil2 = df_fil.iloc[:, [0,1,8,9,10,11,12,13,14]]
    df_sub2 = get_base_clusters(df_fil2)
    df_sub2.to_csv("sub_1.csv", index=False)
    # Score dropped more :-(






import header
from utils import *
import pickle
import os
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

FILE_PATH = "./feature_vector.pkl"
CLUSTER_NUM = 3
viewport_dataset_path = header.viewport_dataset
viewport_train_path = os.path.join(viewport_dataset_path, "train")
viewport_val_path = os.path.join(viewport_dataset_path, "val")
viewport_test_path = os.path.join(viewport_dataset_path, "test")

def cluster_cal(viewport_train_path, ac):
    filenames = os.listdir(viewport_train_path)
    filenames.sort()
    res = [0, 0, 0, 0]
    sep = [47102, 14987, 17128, 23551]

    for filename in tqdm(filenames):
        count = 0
        user_id = int(filename.split("_")[-1][4:]) - 1
        file_path=os.path.join(viewport_train_path, filename)
        with open(file_path, 'rb' ) as f:
            data = pickle.load(f)
            data = simplify_data(data)
            for i in range(1, len(data)):
                if data[i][1] != data[i - 1][1]: count += 1 
        res[ac[user_id]] += count
    res = [res[i] / sep[i] for i in range(4)]
    print(res)
    return res
        
           
def cluster(viewport_train_path, save_path):
    filenames = os.listdir(viewport_train_path)
    filenames.sort()
    user_num = header.user_num
    user_set = {}
    score_list = {}
    user_name_list = []
    for i in range(1, user_num + 1):
        user_name = "user" + str(i)
        user_name_list.append(user_name)
        user_set[user_name] = []
        score_list[user_name] = {}
    for filename in tqdm(filenames):
        user_name = filename.split("_")[-1]
        file_path=os.path.join(viewport_train_path, filename)
        with open(file_path, 'rb' ) as f:
            data = pickle.load(f)
            data = simplify_data(data)
            for item in data:
                user_set[user_name].append(item)
    for name in user_name_list:
        d1 = user_set[name]
        for k,d2 in user_set.items():
            if k not in score_list[name]:
                score_list[name][k] = 0
            for i in range(len(d2)):
                if d1[i] == d2[i]:
                    score_list[name][k] += 1

    feature_vec = []
    for i in range(1, user_num + 1):
        user_name1 = "user" + str(i)
        feature = []
        factor = float(score_list[user_name1][user_name1])
        for j in range(1, user_num + 1):
            user_name2 = "user" + str(j)
            feature.append(score_list[user_name1][user_name2] / factor)
        feature_vec.append(feature)

    pickle.dump(feature_vec, open(save_path, 'wb'))
    #print(score_list)
    #print(feature_vec)

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def get_cluster(cluster_num, file_path):
    feature_vec = None
    cluster_user = []
    for i in range(cluster_num):
        cluster_user.append([])

    with open(file_path, 'rb' ) as f:
        feature_vec = pickle.load(f)
    # print(feature_vec)
    model = AgglomerativeClustering(n_clusters=cluster_num)
    ac = model.fit_predict(feature_vec)
    for i in range(len(ac)):
        user_name = "user" + str(i + 1)
        cluster_user[ac[i]].append(user_name)

    return feature_vec, cluster_user, ac

def plot_tree(cluster_num, file_path, fname):
    feature_vec = None
    cluster_user = []
    for i in range(cluster_num):
        cluster_user.append([])

    with open(file_path, 'rb' ) as f:
        feature_vec = pickle.load(f)
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(feature_vec)
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode='level', p=30)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.savefig(fname)
    plt.close()
    #plt.show()

def pca_analysis(vec, res, cluster_num, fname):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(vec)
    res = np.array(res)
    res = np.reshape(res, (-1, 1))
    principalComponents = np.append(principalComponents, res, 1)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'target'])
    fig, ax = plt.subplots(figsize =(12, 8))
    plt.xlabel('Principal Component 1', fontsize = 24, fontweight ='bold')
    plt.ylabel('Principal Component 2', fontsize = 24, fontweight ='bold')
    # plt.title('Two Components PCA', fontsize = 20, fontweight ='bold')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    # plt.grid(b=True)
    # ax.set_axisbelow(True)
    targets = None
    colors = None
    if cluster_num == 3:
        targets = [0, 1, 2]
        colors = ['m', 'g', 'b']
    elif cluster_num == 4:
        targets = [0, 1, 2, 3]
        colors = ['r', 'g', 'b', 'y']
    elif cluster_num == 5:
        targets = [0, 1, 2, 3, 4]
        colors = ['m', 'g', 'b', 'y', 'k']        

    s = [[] for i in range(cluster_num)]
    cnt = 0
    for target, color in zip(targets,colors):
        indicesToKeep = principalDf['target'] == target
        s[cnt] = plt.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
                , principalDf.loc[indicesToKeep, 'principal component 2']
                , c = color
                , s = 50)
        cnt += 1
    plt.legend(s, ("Group 1", "Group 2", "Group 3", "Group 4"), fontsize=24, frameon=True)
    # fname += str(cluster_num) + "_new.pdf"
    fname = "../Results/pca.pdf"
    plt.savefig(fname)
    plt.show()
    plt.close()
# cluster(viewport_train_path, FILE_PATH)
# ret = get_cluster(CLUSTER_NUM, FILE_PATH)
# print(ret)

def find_best_k(vec):
    fig, ax = plt.subplots(figsize =(12, 8))
    # fig.suptitle("Title for whole figure", fontsize=16)
    plt.title('Elbow Method Using Distortion Score', fontweight ='bold', fontsize = 20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    vec = np.array(vec)
    model = KMeans()
    visualizer = KElbowVisualizer(model, ax=ax, k=(1,11))
    visualizer.fit(vec)        # Fit the data to the visualizer
    visualizer.show("./elbow.pdf")        # Finalize and render the figure 

if __name__ == "__main__":
    # cluster(viewport_val_path, "./feature_vector_val.pkl")
    # cluster(viewport_test_path, "./feature_vector_test.pkl")
    # plot_tree(header.cluster_num, header.cluster_file_path, "Dendrogram_train.png")
    # plot_tree(header.cluster_num, "./feature_vector_val.pkl", "Dendrogram_val.png")
    # plot_tree(header.cluster_num, "./feature_vector_test.pkl", "Dendrogram_test.png")
    
    cluster_num = 4
    vec, cluster_users, ac1 = get_cluster(cluster_num, header.cluster_file_path)
    print(cluster_users)
    print(ac1)
    cluster_cal(viewport_train_path, ac1)
    # find_best_k(vec)

    
    # pca_analysis(vec, ac1, cluster_num, "scatter_train")
    # vec, _, ac2 = get_cluster(cluster_num, "./feature_vector_val.pkl")
    # pca_analysis(vec, ac1, cluster_num, "scatter_val")
    # vec, _, ac3 = get_cluster(cluster_num, "./feature_vector_test.pkl")
    # pca_analysis(vec, ac1, cluster_num, "scatter_test")
    


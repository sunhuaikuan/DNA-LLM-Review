import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import imblearn
from imblearn.under_sampling import RandomUnderSampler,NearMiss, TomekLinks, EditedNearestNeighbours, ClusterCentroids
from imblearn.over_sampling import RandomOverSampler,ADASYN,SMOTE
from imblearn.combine import SMOTEENN

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix,roc_curve, auc, roc_auc_score, accuracy_score,precision_score,recall_score,f1_score, roc_auc_score,classification_report 


def load_data(pkl_dir, chrom=None):

    data_dict={}
    chrom_dict={}
    
    my_set = {i for i in range(1, 24)} | {'X', 'Y'}
    
    if chrom in my_set:
        pkl_file = open(pkl_dir+str(chrom)+'-embeddings.pkl', 'rb')
        chrom_dict = pickle.load(pkl_file)
        data_dict.update(chrom_dict)
        pkl_file.close()
    else:    
        for i in range(1,23):
            pkl_file = open(pkl_dir+str(i)+'-embeddings.pkl', 'rb')
            chrom_dict = pickle.load(pkl_file)
            data_dict.update(chrom_dict)
            pkl_file.close()


    pkl_file = open(pkl_dir+'X-embeddings.pkl', 'rb')
    chrom_dict = pickle.load(pkl_file)
    data_dict.update(chrom_dict)
    pkl_file.close()

    pkl_file = open(pkl_dir+'Y-embeddings.pkl', 'rb')
    chrom_dict = pickle.load(pkl_file)
    data_dict.update(chrom_dict)
    pkl_file.close()


    values_list = list(data_dict.values())    
    data_array = np.vstack(values_list)

    # data_array.shape
    return data_array



def split_data(data_array, ratio):

    # Each Class Data-Size
    df_class_0 = pd.DataFrame(data_array[data_array[:,-1]==0])
    df_class_1 = pd.DataFrame(data_array[data_array[:,-1]==1])

    count_class_0 = df_class_0.shape[0]
    count_class_1 = df_class_1.shape[0]

    # For each class data, split into Traning/Test dataframes
    train_class_0_size = round(count_class_0 * ratio)
    df_class_0_train = df_class_0[:train_class_0_size]
    df_class_0_test  = df_class_0[train_class_0_size:]    

    train_class_1_size = round(count_class_1 * ratio)
    df_class_1_train = df_class_1[:train_class_1_size]
    df_class_1_test  = df_class_1[train_class_1_size:]

    
    # Merget training data of two classes into x_train, y_train
    df_train_over = pd.concat([df_class_0_train, df_class_1_train],axis=0)
    train_array=np.array(df_train_over)

    y_train = train_array[:,-1]
    x_train = train_array[:, :-1]
    
    
    # Merge validate data of two classes into x_validate, y_validate
    df_test_over = pd.concat([df_class_0_test, df_class_1_test],axis=0)
    test_array=np.array(df_test_over)

    y_validate = test_array[:,-1]
    x_validate = test_array[:, :-1]



    return x_train, y_train, x_validate, y_validate



def split_upsampled_data(data_array, ratio):

    # Each Class Data-Size
    df_class_0 = pd.DataFrame(data_array[data_array[:,-1]==0])
    df_class_1 = pd.DataFrame(data_array[data_array[:,-1]==1])

    count_class_0 = df_class_0.shape[0]
    count_class_1 = df_class_1.shape[0]
    
    # For each class data, split into Traning/Test dataframes
    train_class_0_size = round(count_class_0 * ratio)
    df_class_0_train = df_class_0[:train_class_0_size]
    df_class_0_test  = df_class_0[train_class_0_size:]

    train_class_1_size = round(count_class_1 * ratio)
    df_class_1_train = df_class_1[:train_class_1_size]
    df_class_1_test  = df_class_1[train_class_1_size:]

    # Oversampling on class_1 
    df_class_1_over=pd.DataFrame(df_class_1_train).sample(df_class_0_train.shape[0], replace=True)
    df_class_1_over.shape

    # Merget training data of two classes into x_train, y_train
    df_train_over = pd.concat([df_class_0_train, df_class_1_over],axis=0)
    train_array=np.array(df_train_over)

    y_train = train_array[:,-1]
    x_train = train_array[:, :-1]    
    
    # Merge validate data of two classes into x_validate, y_validate
    df_test_over = pd.concat([df_class_0_test, df_class_1_test],axis=0)
    test_array=np.array(df_test_over)

    y_validate = test_array[:,-1]
    x_validate = test_array[:, :-1]


    return x_train, y_train, x_validate, y_validate


def undersample_test_data(X_train,y_train):

    rus=RandomUnderSampler(sampling_strategy="not minority")

    X_res, y_res=rus.fit_resample(X_train,y_train)

    y_res_series = pd.Series(y_res)

    # Plot pie chart
    # y_res_series.value_counts().plot.pie(autopct='%0.2f')
    # print(y_res_series.value_counts())
    
    return  X_res, y_res



def oversample_test_data(X_train,y_train):

    ros=RandomOverSampler(sampling_strategy="not majority")

    X_res, y_res=ros.fit_resample(X_train,y_train)

    y_res_series = pd.Series(y_res)
    # Plot pie chart
    y_res_series.value_counts().plot.pie(autopct='%0.2f')
    print(y_res_series.value_counts())


    return  X_res, y_res



def load_imb_data(method):

    ratio=0.8

    # x_train, y_train, x_test, y_test = split_upsampled_data(data_array, ratio)
    x_train, y_train, x_test, y_test = split_data(data_array, ratio)

    if method=="UNDERSAMPLE":
        x_train,y_train = undersample_test_data(x_train,y_train)
    elif method=="OVERSAMPLE":
        x_train,y_train = oversample_test_data(x_train,y_train)
    elif method=="SMOTEENN":
        sm=SMOTEENN(random_state = 2)
        x_train,y_train=sm.fit_resample(x_train,y_train)
    elif method=="NEARMISS":
        nearmiss=NearMiss(version=3)
        x_train,y_train=nearmiss.fit_resample(x_train,y_train)
    elif method=="TOMEKLINKS":
        tl=TomekLinks()
        x_train,y_train=tl.fit_resample(x_train,y_train)
    elif method=="CLUSTERCENTROIDS":        
        cc = ClusterCentroids(random_state=9)
        x_train,y_train=cc.fit_resample(x_train,y_train)
    elif method=="SMOTE":        
        x_train,y_train = SMOTE().fit_resample(x_train,y_train) # SMOTE: Synthetic Minority Oversampling Technique                 
    elif method=="ADASYN":        
        x_train,y_train = ADASYN().fit_resample(x_train,y_train)  # ADASYN: Adaptive Synthetic
        
    print(x_train.shape,y_train.shape, x_test.shape, y_test.shape)
    
    scaler = MinMaxScaler()
    scaler.fit(x_train) 
    x_train = scaler.transform(x_train)
    x_test  = scaler.transform(x_test)
    
    return x_train,y_train, x_test, y_test



def Print_Result(y_test, y_pred, plotfile=""):
    print(classification_report(y_test, y_pred))
    print("f1_score", f1_score(y_test, y_pred))
    print("precision_score", precision_score(y_test, y_pred))
    print("recall_score",recall_score(y_test, y_pred))
    print("roc_auc_score",roc_auc_score(y_test, y_pred))
    print("confusion_matrix\n",confusion_matrix(y_test, y_pred))
    
    
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Compute ROC AUC score
    roc_auc_score_value = roc_auc_score(y_test, y_pred)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    if (plotfile!=""):
        plt.savefig(plotfile)
    plt.show()

    print("ROC AUC score:", roc_auc_score_value)
    
    
    
    
def load_imb_data(data_array, method):

    ratio=0.8

    # x_train, y_train, x_test, y_test = split_upsampled_data(data_array, ratio)
    x_train, y_train, x_test, y_test = split_data(data_array, ratio)

    if method=="UNDERSAMPLE":
        x_train,y_train = undersample_test_data(x_train,y_train)
    elif method=="OVERSAMPLE":
        x_train,y_train = oversample_test_data(x_train,y_train)
    elif method=="SMOTEENN":
        sm=SMOTEENN(random_state = 2)
        x_train,y_train=sm.fit_resample(x_train,y_train)
    elif method=="NEARMISS":
        nearmiss=NearMiss(version=3)
        x_train,y_train=nearmiss.fit_resample(x_train,y_train)
    elif method=="TOMEKLINKS":
        tl=TomekLinks()
        x_train,y_train=tl.fit_resample(x_train,y_train)
    elif method=="CLUSTERCENTROIDS":        
        cc = ClusterCentroids(random_state=9)
        x_train,y_train=cc.fit_resample(x_train,y_train)
    elif method=="SMOTE":        
        x_train,y_train = SMOTE().fit_resample(x_train,y_train) # SMOTE: Synthetic Minority Oversampling Technique                 
    elif method=="ADASYN":        
        x_train,y_train = ADASYN().fit_resample(x_train,y_train)  # ADASYN: Adaptive Synthetic
        
    print(x_train.shape,y_train.shape, x_test.shape, y_test.shape)
    
    scaler = MinMaxScaler()
    scaler.fit(x_train) 
    x_train = scaler.transform(x_train)
    x_test  = scaler.transform(x_test)
    
    return x_train,y_train, x_test, y_test
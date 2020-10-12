# Import libraries
import streamlit as st
import numpy as np
import pandas as pd
import time
import sklearn
import io

import matplotlib.pyplot as plt
import seaborn as sns

# Set all options
plt.style.use('seaborn-notebook')
plt.rcParams["figure.figsize"] = (20, 3)
pd.options.display.float_format = '{:20,.4f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
sns.set(context="paper", font="monospace")
st.set_option('deprecation.showfileUploaderEncoding', False)

# Metrics 
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

# Import packages for preprocessing of data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

# Import classifiers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC, SVC





# Utility Functions
def quality_report(df):
    dtypes = df.dtypes
    missing_points = df.isnull().sum().sort_values(ascending=False)
    mean = df.mean()
    no_unique = df.T.apply(lambda x: x.nunique(), axis=1)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)
    quality_df = pd.concat([dtypes, percent, missing_points, mean, no_unique], axis=1, keys=['Dtypes', 'Percent', 'Missing_Points','Mean', 'Unique' ])
    return quality_df


def data_descr(df):
    d = df.describe()
    return d
    

def object_count_plot(df):
            for var in df.columns:
                if df[var].dtype == 'object':
                    print(df[var].value_counts())
                    plt.figure(figsize=(12,5))
                    g = sns.countplot(x=var, data=df)
                    g.set_xticklabels(g.get_xticklabels(), rotation=90, ha='right' )
                    plt.tight_layout()
                    plt.show()


def feature_corre(df):
    num_features = df.select_dtypes(exclude = "object")
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(num_features.corr(), annot=True, ax=ax)
    st.pyplot(fig)
    
    
def heat_map(df):
    st.text('Effect of the different classes')
    hi = sns.pairplot(df, vars=list(df.columns), hue='target')
    st.pyplot(hi)


def numeric_distribution_plot(df):
    for col in df.columns:
        if df[col].dtype != 'object':
            st.write(df[col].describe())
            fig, ax = plt.subplots(figsize=(12,5))
            plt.title("Distribution of "+ col)
            ax = sns.distplot(df[col].dropna())
            plt.tight_layout()
            st.pyplot(fig)
            plt.show()


def main():
    
    page = st.sidebar.selectbox("Choose a page", ['Homepage', 'Data Exploration', 'Data Visualization', 'Data Processing and Modeling'])
    st.text('Select a page in the sidebar')
    st.title("Welcome to DataHacker Hub")

    project_name = st.text_input("What is the name of your project?")

    # Uploading the data
    try:
        file_upload_train = st.file_uploader("Upload your training dataset", type="csv")
        if file_upload_train is not None:
            train = pd.read_csv(file_upload_train)
            st.success("File uploaded successfully!")
    except UnboundLocalError:
        st.write("Please upload the training set.")
    try:   
        file_upload_test = st.file_uploader("Upload your test dataset", type="csv")
        if file_upload_test is not None:
            test = pd.read_csv(file_upload_test)
            st.success("File uploaded successfully!")
    except UnboundLocalError:
            st.write("Please upload the training set.")
    try:   
        file_upload_sample = st.file_uploader("Upload your sample submission file", type="csv")
        if file_upload_sample is not None:
            sample = pd.read_csv(file_upload_sample)
            st.success("File uploaded successfully!")
    except UnboundLocalError:
            st.write("Please upload the training set.")

    # Extract the target variable
    try:
        train_copy = train.copy()
        target_var = st.text_input("What is the feature name for your target column?")
        if target_var in train.columns:
            target_var_col = train[target_var]
        elif target_var not in train.columns:
            st.write("Sorry %s is not among the features" % str(target_var))
        train.drop(target_var, inplace=True, axis=1)
    except UnboundLocalError:
        st.write("Pls upload your data!")

    # A placeholder to store the scaled data
    #train_scaled_place = np.zeros((train.shape))
    #target_place = np.zeros((target_var_col.shape[0], ))
    #test_scaled_place = np.zeros((test.shape))

    
    
    if page == 'Homepage':
        st.header("Hi :wave: Welcome to the DataHacker Home page. We help you automate your Machine Learning projects from start to finish without writing a single code!")
    
    elif page == "Data Exploration":
        try:
            st.subheader("Properties of the training data:\n")
            st.dataframe(train.head(5))

            st.write("Shape of training data:", train.shape)
            st.write("**Training data report:**\n")

            quality = quality_report(train)
            st.dataframe(quality)

            st.write("**Statistics of the training data**")
            st.dataframe(data_descr(train))

        except UnboundLocalError:
            st.write("No data to explore! Upload your training data.")

        try:
            st.subheader("**Properties of the test data:**\n")
            st.dataframe(test.head(5))

            st.write("Shape of test data:", test.shape)
            st.write("**Test data report:**\n")

            quality = quality_report(test)
            st.dataframe(quality)

            st.write("**Statistics of the test data**")
            st.dataframe(data_descr(test))

        except UnboundLocalError:
            st.write("No data to explore! Upload your test data.")
        
    elif page == "Data Visualization":
        chart = st.selectbox(
            "What type of visualization do you want?",
            ("Count Plot", "Distribution Plot", "Correlations", "HeatMap", "Histogram Plot")
        )
        if chart == "Count Plot":

            try:
                # Countplot is used for categorical features so the test set annot havea countplot chart
                obj_cplot = object_count_plot(train_copy)
                st.write("**Count Plot of the Training Data:**\n")
                st.dataframe(obj_cplot)
            except UnboundLocalError:
                st.write("No data to visualize! Upload your training data.")
        elif chart == "Distribution Plot":

            try:
                st.write("**Here is a distribution plot of the training data:**\n")
                dist_plot_train = numeric_distribution_plot(train)
            
            except UnboundLocalError:
                st.write("No data to visualize! Upload your training data.")
            try:
                st.write("**Here is a distribution plot of the test data:**\n")
                dist_plot_test = numeric_distribution_plot(test)
            
            except UnboundLocalError:
                st.write("No data to visualize! Upload your test data.")

        elif chart == "Correlations":
            try:
                st.write("**Here are the correlation between the features:**")
                feature_corre(train)
            except UnboundLocalError:
                st.write("No data to visualize! Upload your training data.")
            try:
                st.write("**Here are the correlation between the features:**")
                feature_corre(test)
            except UnboundLocalError:
                st.write("No data to visualize! Upload your test data.")
        
        elif chart == "HeatMap":
            try:
                heat_map(train_copy)
            except UnboundLocalError:
                st.write("No data to visualize! Upload your training data.")


        elif chart == "Histogram Plot":
            st.write("We will soon add this feature...")
    
    elif page == "Data Processing and Modeling":

        st.header("Here we go! Let's preprocess your data.")
        # Drop the target column and convert to NumPy arrays
        
        train_num = train.values
        target_num = target_var_col.values
        test_num = test.values

        #Append the target to the placeholder
        target_place = target_num

        # Scaling options
        scaler_opt = st.radio(
            "Do you want to scale your data?",
            ("Yes", "No")
        )
        if scaler_opt == "Yes":
            scaling_opt = st.selectbox(
                "How do you want to scale your data?",
                ("StandardScaler", "MinMaxScaler", "Normalizer", "RobustScaler")
            )
        
            if scaling_opt == "StandardScaler":
                scaler = StandardScaler().fit(train_num)
                train_scaled = scaler.transform(train_num)
                test_scaled = scaler.transform(test_num)

                # Append it to the placeholder 
                train_scaled_place = train_scaled
                test_scaled_place = test_scaled

                # Return the scaled data
                st.write('Here is the scaled data:')
                st.dataframe(train_scaled_place)

            elif scaling_opt == "MinMaxScaler":
                scaler = MinMaxScaler().fit(train_num)
                train_scaled= scaler.transform(train_num)
                test_scaled = scaler.transform(test_num)
                
                # Append to the placeholder
                train_scaled_place = train_scaled
                test_scaled_place = test_scaled

                # Return the scaled data
                st.write('Here is the scaled data:')
                st.write(train_scaled_place)

            elif scaling_opt == "Normalizer":
                scaler = Normalizer().fit(train_num)
                train_scaled= scaler.transform(train_num)
                test_scaled = scaler.transform(test_num)
                
                # Append to the placeholder
                train_scaled_place = train_scaled
                test_scaled_place = test_scaled

                # Return the scaled data
                st.write('Here is the scaled data:')
                st.write(train_scaled_place)
            
            elif scaling_opt == "RobustScaler":
                scaler = RobustScaler().fit(train_num)
                train_scaled= scaler.transform(train_num)
                test_scaled = scaler.transform(test_num)
                
                # Append to the placeholder
                train_scaled_place = train_scaled
                test_scaled_place = test_scaled

                # Return the scaled data
                st.write('Here is the scaled data:')
                st.write(train_scaled_place)
                
            
        elif scaler_opt == "No":
            train_scaled_place = train_num
            test_scaled_place = test_num
            
   

        st.subheader("Good job :+1:! Your data is ready to be trained:clap:.")
        
        split = st.selectbox(
            "How do you like to split your data?",
            ("CrossValidation", "Random_Splitting")
        )
        ests = st.selectbox(
            "Choose a model to train your data.",
            ("LogisticRegression", "CatBoostClassifier", "XGBClassifier",
             "MLPClassifier", "GradientBoostingClassifier", "DecisionTreeClassifier",
             "RandomForestClassifier", "KNNClassifier")
             )
        

        if split == "Random_Splitting" and ests == "LogisticRegression":
            
            X_train, X_test, y_train, y_test = train_test_split(train_scaled_place, target_place, random_state=0, stratify=target_place, test_size=0.30)

            with st.spinner("Training model..."):
                model = LogisticRegression().fit(X_train, y_train)
                pred = model.predict(X_test)

                # Score with metrics (LogLoss)
                logloss_score = log_loss(np.argmax(model.predict_proba(X_test), axis=1), y_test)
                

                # Score with accuracy
                r2_score = model.score(X_test, y_test)

                # Score with roc_auc
                

                #scores_list = ["LogLoss_score", "R2_score", "ROC_AUC_score"]
                #score_list = [log_score, r2_score, roc_auc]

                # Prediction on test set
                test_pred = model.predict(test_scaled_place)

                # Return the values
                st.write("Accuracy Perfomance on the validation set is {}".format(r2_score))
                st.write("Logloss Perfomance on the validation set is {}".format(logloss_score))
                #st.write("**Performance of your model on the validation set**:\n")
                #for i, j in zip(scores_list, score_list):
                   # st.write("The {} is {}".format(i, j))
                    
                pred_df = pd.DataFrame(columns=["Iris_ID", "predicted_class"])
                pred_df["predicted_class"] = test_pred
                    
                st.table(pred_df)
                st.success("Finished training!")
        if split == "CrossValidation" and ests == "LogisticRegression":
            FOLDS = st.number_input("How many folds do you want to split your data?", step=1, min_value=0, max_value=30, value=0)
            with st.spinner("Training model..."):
                kfold = StratifiedKFold(n_splits = int(FOLDS), shuffle=True)
                oos_y = []
                oos_pred = []
                num = 0

                for train, test in kfold.split(train_scaled_place, target_place):
                    num += 1
                    st.write("__Training fold {} ...__".format(num))
                    X_train, X_test = train_scaled_place[train], train_scaled_place[test]
                    y_train, y_test = target_place[train], target_place[test]

                    model = LogisticRegression().fit(X_train, y_train)
                    r2_score = model.score(X_test, y_test)
                    oos_y.append(y_test)
                    oos_pred.append(model.predict(X_test))
                oos_y = np.concatenate(oos_y)
                oos_pred = np.concatenate(oos_pred)
                score = accuracy_score(oos_pred, oos_y)
                st.write("Perfomance on the validation set  with {} fold is {}".format(int(FOLDS), score))
                
                test_pred = model.predict(test_scaled_place)
                pred_df = pd.DataFrame(columns=["Iris_ID", "predicted_class"])
                pred_df["predicted_class"] = test_pred
                        
                st.table(pred_df)
                st.success("Finished Training!")        
        
        
        


        

        

if __name__ == '__main__':
    main()
        
            
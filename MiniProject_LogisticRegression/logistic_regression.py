
import os 
import sys 

import numpy as np 
import pandas as pd 
import matplotlib .pyplot as plt 
import seaborn as sns 

from sklearn .model_selection import train_test_split 
from sklearn .preprocessing import LabelEncoder ,StandardScaler 
from sklearn .linear_model import LogisticRegression 
from sklearn .metrics import (
accuracy_score ,
precision_score ,
recall_score ,
f1_score ,
roc_auc_score ,
roc_curve ,
confusion_matrix ,
classification_report ,
)

from imblearn .over_sampling import SMOTE 





DATA_FILE ="heart.csv"
GRAPHS_DIR ="graphs"
OUTPUT_DIR ="output"
TARGET_COLUMN ="Heart Disease Status"
RANDOM_STATE =42 
TEST_SIZE =0.20 


sns .set_style ("whitegrid")
plt .rcParams ["figure.dpi"]=100 
plt .rcParams ["font.size"]=10 
plt .rcParams ["axes.titleweight"]="bold"





def create_project_folders ():

    os .makedirs (GRAPHS_DIR ,exist_ok =True )
    os .makedirs (OUTPUT_DIR ,exist_ok =True )


def load_dataset (file_path ):

    try :
        dataframe =pd .read_csv (file_path )
        return dataframe 
    except FileNotFoundError :
        print (
        f"[ERROR] Could not find '{file_path }'. "
        "Please make sure 'heart.csv' is in the same folder as this script."
        )
        sys .exit (1 )
    except pd .errors .ParserError as error :
        print (f"[ERROR] Failed to parse '{file_path }': {error }")
        sys .exit (1 )
    except Exception as error :
        print (f"[ERROR] An unexpected error occurred while loading the dataset: {error }")
        sys .exit (1 )


def save_figure (figure ,filename ):

    filepath =os .path .join (GRAPHS_DIR ,filename )
    figure .tight_layout ()
    figure .savefig (filepath ,dpi =300 ,bbox_inches ="tight")
    plt .close (figure )





def handle_missing_values (dataframe ):

    dataframe =dataframe .copy ()


    numerical_columns =dataframe .select_dtypes (include =[np .number ]).columns .tolist ()
    categorical_columns =dataframe .select_dtypes (exclude =[np .number ]).columns .tolist ()


    if TARGET_COLUMN in categorical_columns :
        categorical_columns .remove (TARGET_COLUMN )


    for column in numerical_columns :
        if dataframe [column ].isnull ().any ():
            mean_value =dataframe [column ].mean ()
            dataframe [column ]=dataframe [column ].fillna (mean_value )


    for column in categorical_columns :
        if dataframe [column ].isnull ().any ():
            mode_value =dataframe [column ].mode ()[0 ]
            dataframe [column ]=dataframe [column ].fillna (mode_value )

    return dataframe 





def encode_categorical_features (dataframe ):

    dataframe =dataframe .copy ()
    categorical_columns =dataframe .select_dtypes (exclude =[np .number ]).columns .tolist ()

    if TARGET_COLUMN in categorical_columns :
        categorical_columns .remove (TARGET_COLUMN )

    label_encoders ={}
    for column in categorical_columns :
        encoder =LabelEncoder ()
        dataframe [column ]=encoder .fit_transform (dataframe [column ].astype (str ))
        label_encoders [column ]=encoder 

    return dataframe 





def plot_eda_collage (raw_dataframe ):

    figure ,axes =plt .subplots (2 ,3 ,figsize =(18 ,10 ))
    figure .suptitle (
    "Exploratory Data Analysis of Heart Disease Dataset",
    fontsize =18 ,
    fontweight ="bold",
    )


    sns .histplot (raw_dataframe ["Age"],kde =True ,color ="#4C72B0",ax =axes [0 ,0 ])
    axes [0 ,0 ].set_title ("Age Distribution")
    axes [0 ,0 ].set_xlabel ("Age (years)")
    axes [0 ,0 ].set_ylabel ("Frequency")


    sns .histplot (raw_dataframe ["Blood Pressure"],kde =True ,color ="#DD8452",ax =axes [0 ,1 ])
    axes [0 ,1 ].set_title ("Blood Pressure Distribution")
    axes [0 ,1 ].set_xlabel ("Blood Pressure (mmHg)")
    axes [0 ,1 ].set_ylabel ("Frequency")


    sns .histplot (raw_dataframe ["Cholesterol Level"],kde =True ,color ="#55A868",ax =axes [0 ,2 ])
    axes [0 ,2 ].set_title ("Cholesterol Level Distribution")
    axes [0 ,2 ].set_xlabel ("Cholesterol Level (mg/dL)")
    axes [0 ,2 ].set_ylabel ("Frequency")


    sns .histplot (raw_dataframe ["BMI"],kde =True ,color ="#C44E52",ax =axes [1 ,0 ])
    axes [1 ,0 ].set_title ("BMI Distribution")
    axes [1 ,0 ].set_xlabel ("BMI")
    axes [1 ,0 ].set_ylabel ("Frequency")


    exercise_counts =raw_dataframe ["Exercise Habits"].value_counts ()
    sns .barplot (
    x =exercise_counts .index ,
    y =exercise_counts .values ,
    hue =exercise_counts .index ,
    palette ="viridis",
    legend =False ,
    ax =axes [1 ,1 ],
    )
    axes [1 ,1 ].set_title ("Exercise Habits")
    axes [1 ,1 ].set_xlabel ("Exercise Habit Level")
    axes [1 ,1 ].set_ylabel ("Count")


    smoking_counts =raw_dataframe ["Smoking"].value_counts ()
    axes [1 ,2 ].pie (
    smoking_counts .values ,
    labels =smoking_counts .index ,
    autopct ="%1.1f%%",
    colors =["#8172B2","#CCB974"],
    startangle =90 ,
    wedgeprops ={"edgecolor":"white","linewidth":1 },
    )
    axes [1 ,2 ].set_title ("Smoking Status")

    save_figure (figure ,"figure1_eda_collage.png")


def plot_class_distribution_before_smote (dataframe ):

    figure ,axis =plt .subplots (figsize =(7 ,6 ))
    class_counts =dataframe [TARGET_COLUMN ].value_counts ()

    bars =axis .bar (
    class_counts .index .astype (str ),
    class_counts .values ,
    color =["#4C72B0","#C44E52"],
    edgecolor ="black",
    )


    for bar in bars :
        height =bar .get_height ()
        axis .annotate (
        f"{int (height )}",
        xy =(bar .get_x ()+bar .get_width ()/2 ,height ),
        xytext =(0 ,5 ),
        textcoords ="offset points",
        ha ="center",
        fontweight ="bold",
        )

    axis .set_title ("Dataset Class Distribution Before SMOTE")
    axis .set_xlabel ("Heart Disease Status")
    axis .set_ylabel ("Number of Samples")

    save_figure (figure ,"figure2_class_distribution_before_smote.png")


def plot_missing_values_chart (dataframe ):

    missing_counts =dataframe .isnull ().sum ()
    missing_counts =missing_counts [missing_counts >0 ].sort_values (ascending =False )

    figure ,axis =plt .subplots (figsize =(12 ,6 ))

    if missing_counts .empty :

        axis .text (
        0.5 ,0.5 ,"No missing values found in the dataset.",
        ha ="center",va ="center",fontsize =14 ,
        )
        axis .axis ("off")
    else :
        axis .bar (
        missing_counts .index ,
        missing_counts .values ,
        color ="#DD8452",
        edgecolor ="black",
        )
        axis .set_xticks (range (len (missing_counts .index )))
        axis .set_xticklabels (missing_counts .index ,rotation =45 ,ha ="right")
        axis .set_ylabel ("Number of Missing Values")
        axis .set_xlabel ("Column Name")

    axis .set_title ("Missing Values per Column")
    save_figure (figure ,"figure3_missing_values.png")





def encode_target_variable (dataframe ):

    dataframe =dataframe .copy ()
    target_encoder =LabelEncoder ()
    dataframe [TARGET_COLUMN ]=target_encoder .fit_transform (dataframe [TARGET_COLUMN ])

    return dataframe ,target_encoder 


def separate_features_and_target (dataframe ):

    feature_matrix =dataframe .drop (columns =[TARGET_COLUMN ])
    target_vector =dataframe [TARGET_COLUMN ]

    return feature_matrix ,target_vector 





def split_scale_and_balance (feature_matrix ,target_vector ):


    X_train ,X_test ,y_train ,y_test =train_test_split (
    feature_matrix ,
    target_vector ,
    test_size =TEST_SIZE ,
    random_state =RANDOM_STATE ,
    stratify =target_vector ,
    )

    feature_names =feature_matrix .columns .tolist ()


    scaler =StandardScaler ()
    X_train_scaled =scaler .fit_transform (X_train )
    X_test_scaled =scaler .transform (X_test )


    smote =SMOTE (random_state =RANDOM_STATE )
    X_train_balanced ,y_train_balanced =smote .fit_resample (X_train_scaled ,y_train )

    return X_train_balanced ,X_test_scaled ,y_train_balanced ,y_test ,feature_names ,scaler 





def train_logistic_regression (X_train ,y_train ):

    model =LogisticRegression (
    max_iter =1000 ,
    random_state =RANDOM_STATE ,
    solver ="lbfgs",
    )
    model .fit (X_train ,y_train )

    return model 





def evaluate_model (model ,X_test ,y_test ):

    y_pred =model .predict (X_test )
    y_pred_proba =model .predict_proba (X_test )[:,1 ]

    accuracy =accuracy_score (y_test ,y_pred )
    precision =precision_score (y_test ,y_pred )
    recall =recall_score (y_test ,y_pred )
    f1 =f1_score (y_test ,y_pred )
    roc_auc =roc_auc_score (y_test ,y_pred_proba )
    conf_matrix =confusion_matrix (y_test ,y_pred )
    class_report =classification_report (y_test ,y_pred ,target_names =["No","Yes"])

    return {
    "accuracy":accuracy ,
    "precision":precision ,
    "recall":recall ,
    "f1_score":f1 ,
    "roc_auc":roc_auc ,
    "confusion_matrix":conf_matrix ,
    "classification_report":class_report ,
    "y_pred":y_pred ,
    "y_pred_proba":y_pred_proba ,
    }





def plot_confusion_matrix (conf_matrix ):

    figure ,axis =plt .subplots (figsize =(7 ,6 ))

    sns .heatmap (
    conf_matrix ,
    annot =True ,
    fmt ="d",
    cmap ="Blues",
    cbar =True ,
    xticklabels =["No","Yes"],
    yticklabels =["No","Yes"],
    linewidths =0.5 ,
    linecolor ="white",
    ax =axis ,
    )

    axis .set_title ("Confusion Matrix")
    axis .set_xlabel ("Predicted Label")
    axis .set_ylabel ("True Label")

    save_figure (figure ,"figure4_confusion_matrix.png")


def plot_roc_curve (y_test ,y_pred_proba ,roc_auc ):

    false_positive_rate ,true_positive_rate ,_ =roc_curve (y_test ,y_pred_proba )

    figure ,axis =plt .subplots (figsize =(7 ,6 ))

    axis .plot (
    false_positive_rate ,
    true_positive_rate ,
    color ="#C44E52",
    linewidth =2.5 ,
    label =f"ROC Curve (AUC = {roc_auc :.4f})",
    )
    axis .plot ([0 ,1 ],[0 ,1 ],color ="gray",linestyle ="--",linewidth =1.5 ,label ="Random Guess")

    axis .set_xlim ([0.0 ,1.0 ])
    axis .set_ylim ([0.0 ,1.05 ])
    axis .set_xlabel ("False Positive Rate")
    axis .set_ylabel ("True Positive Rate")
    axis .set_title ("ROC Curve - Logistic Regression")
    axis .legend (loc ="lower right")

    save_figure (figure ,"figure5_roc_curve.png")


def plot_model_performance_comparison (metrics_dict ):

    metric_names =["Accuracy","Precision","Recall","F1 Score"]
    metric_values =[
    metrics_dict ["accuracy"],
    metrics_dict ["precision"],
    metrics_dict ["recall"],
    metrics_dict ["f1_score"],
    ]

    figure ,axis =plt .subplots (figsize =(8 ,6 ))
    bars =axis .bar (
    metric_names ,
    metric_values ,
    color =["#4C72B0","#DD8452","#55A868","#C44E52"],
    edgecolor ="black",
    )


    for bar ,value in zip (bars ,metric_values ):
        axis .annotate (
        f"{value :.3f}",
        xy =(bar .get_x ()+bar .get_width ()/2 ,value ),
        xytext =(0 ,5 ),
        textcoords ="offset points",
        ha ="center",
        fontweight ="bold",
        )

    axis .set_ylim ([0 ,1.1 ])
    axis .set_ylabel ("Score")
    axis .set_title ("Model Performance Comparison")

    save_figure (figure ,"figure6_model_performance_comparison.png")





def plot_correlation_heatmap (dataframe ):

    numerical_dataframe =dataframe .select_dtypes (include =[np .number ])
    correlation_matrix =numerical_dataframe .corr ()

    figure ,axis =plt .subplots (figsize =(14 ,12 ))

    sns .heatmap (
    correlation_matrix ,
    annot =True ,
    fmt =".2f",
    cmap ="coolwarm",
    center =0 ,
    square =True ,
    linewidths =0.5 ,
    annot_kws ={"size":7 },
    cbar_kws ={"shrink":0.8 },
    ax =axis ,
    )

    axis .set_title ("Correlation Heatmap (Numerical Features)",fontsize =16 )
    plt .setp (axis .get_xticklabels (),rotation =45 ,ha ="right")
    plt .setp (axis .get_yticklabels (),rotation =0 )

    save_figure (figure ,"figure7_correlation_heatmap.png")


def plot_feature_importance (model ,feature_names ):

    coefficients =model .coef_ [0 ]
    importance_dataframe =pd .DataFrame ({
    "Feature":feature_names ,
    "Coefficient":coefficients ,
    "Absolute_Coefficient":np .abs (coefficients ),
    }).sort_values (by ="Absolute_Coefficient",ascending =False )

    figure ,axis =plt .subplots (figsize =(10 ,max (6 ,len (feature_names )*0.35 )))

    colors =["#C44E52"if coef <0 else "#4C72B0"for coef in importance_dataframe ["Coefficient"]]

    axis .barh (
    importance_dataframe ["Feature"],
    importance_dataframe ["Absolute_Coefficient"],
    color =colors ,
    edgecolor ="black",
    )

    axis .invert_yaxis ()
    axis .set_xlabel ("Absolute Coefficient Value")
    axis .set_title ("Logistic Regression Feature Importance")

    save_figure (figure ,"figure8_feature_importance.png")

    return importance_dataframe 





def save_results_to_output (metrics_dict ):

    results_path =os .path .join (OUTPUT_DIR ,"results.txt")
    with open (results_path ,"w",encoding ="utf-8")as results_file :
        results_file .write ("Model: Logistic Regression\n\n")
        results_file .write (f"Accuracy: {metrics_dict ['accuracy']*100 :.2f}%\n")
        results_file .write (f"Precision: {metrics_dict ['precision']*100 :.2f}%\n")
        results_file .write (f"Recall: {metrics_dict ['recall']*100 :.2f}%\n")
        results_file .write (f"F1 Score: {metrics_dict ['f1_score']*100 :.2f}%\n")
        results_file .write (f"ROC-AUC: {metrics_dict ['roc_auc']:.4f}\n")





def main ():


    create_project_folders ()


    raw_dataframe =load_dataset (DATA_FILE )


    imputed_dataframe =handle_missing_values (raw_dataframe )



    plot_eda_collage (imputed_dataframe )
    plot_class_distribution_before_smote (imputed_dataframe )
    plot_missing_values_chart (raw_dataframe )


    encoded_dataframe =encode_categorical_features (imputed_dataframe )


    encoded_dataframe ,target_encoder =encode_target_variable (encoded_dataframe )


    feature_matrix ,target_vector =separate_features_and_target (encoded_dataframe )


    (
    X_train_balanced ,
    X_test_scaled ,
    y_train_balanced ,
    y_test ,
    feature_names ,
    scaler ,
    )=split_scale_and_balance (feature_matrix ,target_vector )


    model =train_logistic_regression (X_train_balanced ,y_train_balanced )


    metrics_dict =evaluate_model (model ,X_test_scaled ,y_test )


    plot_confusion_matrix (metrics_dict ["confusion_matrix"])
    plot_roc_curve (y_test ,metrics_dict ["y_pred_proba"],metrics_dict ["roc_auc"])
    plot_model_performance_comparison (metrics_dict )


    plot_correlation_heatmap (encoded_dataframe )
    plot_feature_importance (model ,feature_names )


    save_results_to_output (metrics_dict )





if __name__ =="__main__":
    main ()

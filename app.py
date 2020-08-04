# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
import streamlit as st
import pandas as pd
import pickle

PATH = '.'

logit_model = pickle.load(open(PATH + "/models/logistic_model.pkl", 'rb'))
# rf_model = pickle.load(open(PATH + "/models/RF_model.pkl", 'rb'))
# nn_model = pickle.load(open(PATH+"/models/NN_model.pkl", 'rb'))
# nn_model = tf.keras.models.load_model(PATH + "/models/NN_model/1/")


_anticipated_cols = ['fnstatus2_Independent',
                     'fnstatus2_Partially Dependent',
                     'fnstatus2_Totally Dependent',
                     'fnstatus2_Unknown',
                     'hxcopd_No',
                     'hxcopd_Yes',
                     'hypermed_No',
                     'hypermed_Yes',
                     'elderly_elderly',
                     'elderly_not elderly',
                     'asa_three_asa 3',
                     'asa_three_not asa 3',
                     'anemia_anemia',
                     'anemia_no anemia',
                     'hypoalb_Low',
                     'hypoalb_Normal',
                     'dyspnea_new_No',
                     'dyspnea_new_Yes',
                     'diabetes_new_No',
                     'diabetes_new_Yes']

# _NN_threshold = .2


def probs_to_pred(probs, threshold=0.5):
    if float(probs) > threshold:
        return '**yes**, this patient should receive outpatient treatment.'
    else:
        return '**no**, this patient should not receive outpatient treatment.'


def input_dict_to_df(user_input_dict, anticipated_cols=_anticipated_cols, expected_columns=len(_anticipated_cols)):
    user_input_df = pd.get_dummies(pd.DataFrame(user_input_dict, index=[0]))
    missing_cols = list(set(anticipated_cols) - set(user_input_df.columns))
    for col in missing_cols:
        user_input_df[col] = 0
    assert(user_input_df.shape[1] == expected_columns)
    user_input_df = user_input_df[anticipated_cols]
    return user_input_df


st.title("Predicting if Outpatient treatment is a good fit")
st.markdown(
    """
This app allows the user to select different patient attributes and see if
outpatient treatment is recommended.
""")

user_input = dict()

user_input['fnstatus2'] = st.selectbox(
    "Financial Status?",
    ['Independent', 'Partially Dependent', 'Totally Dependent', 'Unknown']
)
user_input['hxcopd'] = st.selectbox(
    "Has COPD?",
    ['Yes', 'No']
)
user_input['hypermed'] = st.selectbox(
    "hypermed?",
    ['Yes', 'No']
)
user_input['elderly'] = st.selectbox(
    "Is elderly?",
    ['elderly', 'not elderly']
)
user_input['asa_three'] = st.selectbox(
    "Is ASA Score more than 3?",
    ['not asa 3', 'asa 3']
)
user_input['anemia'] = st.selectbox(
    "Is anemic?",
    ['anemia', 'no anemia']
)
user_input['hypoalb'] = st.selectbox(
    "hypoalb?",
    ['Normal', 'Low']
)
user_input['dyspnea_new'] = st.selectbox(
    "dyspnea_new?",
    ['Yes', 'No']
)
user_input['diabetes_new'] = st.selectbox(
    "Has diabetes?",
    ['Yes', 'No']
)

user_input_df = input_dict_to_df(user_input)

LR_probs = logit_model.predict_proba(user_input_df)[0][1]
# RF_probs = rf_model.predict_proba(user_input_df)[0][1]
# NN_probs = nn_model.predict(user_input_df)[0][0]

st.markdown(
    """
This patient's attributes:
- Financial Status: *{fnstatus2}*
- Has COPD: *{hxcopd}*
- Hypermed: *{hypermed}*
- Is elderly: *{elderly}*
- Is ASA Score more than 3? *{asa_three}*
- Is anemic: *{anemia}*
- hypoalb: *{hypoalb}*
- dyspnea_new: *{dyspnea_new}*
- Has diabetes: *{diabetes_new}*

We found that a logistic regression model had a better AUC than a Random Forests or three-layer neural network. All three performed better than a baseline prediction using only patient ASA score to predict outcomes ([AUC chart](https://imgur.com/a/1xcQGgJ)).

The Logistic regression model predicts that {LR_probs_pred} According to the model, the predicted probability of successful outpatient treatment = {LR_probability:.1f}%.
""".format(
        fnstatus2=user_input['fnstatus2'],
        hxcopd=user_input['hxcopd'],
        hypermed=user_input['hypermed'],
        elderly=user_input['elderly'],
        asa_three=user_input['asa_three'],
        anemia=user_input['anemia'],
        hypoalb=user_input['hypoalb'],
        dyspnea_new=user_input['dyspnea_new'],
        diabetes_new=user_input['diabetes_new'],
        LR_probability=LR_probs * 100,
        LR_probs_pred=probs_to_pred(LR_probs)
        # ,RF_probability=RF_probs * 100,
        # RF_probs_pred=probs_to_pred(RF_probs),
        # NN_pred=NN_probs,
        # NN_threshold=_NN_threshold,
        # NN_probs_pred=probs_to_pred(NN_probs, threshold=_NN_threshold)
    ))

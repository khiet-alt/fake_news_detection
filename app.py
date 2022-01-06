import numpy as np
import pandas as pd
import re
from vncorenlp import VnCoreNLP
import pickle 
import streamlit as st

vn_stopwords = []
with open('data/vietnamese_stopwords.txt', encoding="utf8") as file:
    for line in file.read().splitlines():
        vn_stopwords.append(line.strip())

LR = pickle.load(open('data/LogisticRegression.sav','rb'))
DTC = pickle.load(open('data/DecisionTreeClassifier.sav','rb'))
GBC = pickle.load(open('data/GradientBoostingClassifier.sav','rb'))
REG = pickle.load(open('data/LinearRegression.sav','rb'))
CLF = pickle.load(open('data/NearestCentroid.sav','rb'))
vectorization = pickle.load(open('data/vectorization.sav','rb'))

def preprocessing(X_df):
    for index, test_str in enumerate(X_df):
        # Remove http links
        filtered_str = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", test_str[0])
        # Remove all number in string
        filtered_str = ''.join([i for i in filtered_str if not i.isdigit()])
        # Remove all special characters and punctuation
        filtered_str = re.sub('\W+',' ', filtered_str)
        filtered_str = filtered_str.strip()
        X_df[index][0] = filtered_str.lower()

    X_df = X_df.flatten()

    # Remove stop words
    for index, test_str in enumerate(X_df):
        sw_removed = [w for w in test_str.split(' ') if not w in vn_stopwords]
        X_df[index] = ' '.join(sw_removed)

    return X_df

def tokennize_text(X_df):
    '''
    Tokenize the sentences into words
    return a numpy array of preprocessed text, each item in array is a paragraph after remove stopwords, special characters, ...
    '''
    annotator = VnCoreNLP("VnCoreNLP-1.1.1.jar")
    with annotator:
        for index, filtered_text in enumerate(X_df):
            token_text = annotator.tokenize(filtered_text)[0]
            X_df[index] = ' '.join(token_text)
    
    return X_df

def prediction(model_name, news):
    test_dict = {"text": [news]}
    test_df = pd.DataFrame(test_dict)

    text = test_df.iloc[:,:].values

    processed_text = tokennize_text(preprocessing(text))
    vector_text = vectorization.transform(processed_text)

    if model_name == 'Logistic Regression':
        return LR.predict(vector_text)

    elif model_name == 'Decision Tree':
        return DTC.predict(vector_text)

    elif model_name == 'Gradient Booting Classifier':
        return GBC.predict(vector_text)

    elif model_name == 'Linear Regession':
        y_pred = REG.predict(vector_text)
        y_pred[y_pred < 0.5] = 0
        y_pred[y_pred >= 0.5] = 1
        return y_pred

    elif model_name == 'Nearest Centroid Classifier':
        return CLF.predict(vector_text)

authors='''
19120525 - Lê Minh Hữu
19120533 - Ninh Duy Huy
19120544 - Cao Thanh Khiết
19120547 - Nguyễn Tuấn Khoa
19120557 - Trần Tuấn Kiệt'''

def main():
    st.title('VIETNAMESE FAKE NEWS DETECTION')
    st.text(authors)

    news = st.text_area('Write your news here')

    model = st.selectbox('Select your model',options=['Logistic Regression','Decision Tree',
                                                     'Gradient Booting Classifier','Linear Regession',
                                                     'Nearest Centroid Classifier'])

    if st.button('Detect'):
        predict = prediction(model, news)
        if predict[0] == 1:
            st.success("This is fake news.")
        else:
            st.success("This is not fake news.")

if __name__ == '__main__':
    main()
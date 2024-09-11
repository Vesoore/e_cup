import numpy as np 
import pandas as pd  
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import gc
import json
import re
from sklearn import preprocessing
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
import joblib

def load_data():

    attributes_path = './data/test/attributes_test.parquet'
    resnet_path = './data/test/resnet_test.parquet'
    text_and_bert_path = './data/test/text_and_bert_test.parquet'
    val_path = './data/test/test.parquet'



    attr = pd.read_parquet(attributes_path, engine='pyarrow')
    res_emb = pd.read_parquet(resnet_path, engine='pyarrow')
    text_and_bert = pd.read_parquet(text_and_bert_path, engine='pyarrow')
    test = pd.read_parquet(val_path, engine='pyarrow')
    
    res_emb = res_emb[['variantid', 'main_pic_embeddings_resnet_v1']]
    gc.collect()

    return attr, res_emb, text_and_bert, test

def extract_text_from_row(categories, attributes):
    category_text = ' '.join(
        [' '.join(map(str, v)) if isinstance(v, list) else str(v) for v in categories.values()]
    )
    
    attributes_text = ' '.join(
        [' '.join(map(str, v)) if isinstance(v, list) else str(v) for v in attributes.values()]
    )
    
    return f"{category_text} {attributes_text}"

def process_attr(df):

    df[['categories', 'characteristic_attributes_mapping']] = df[['categories', 'characteristic_attributes_mapping']].applymap(json.loads)
    df['combined_text'] = [
        extract_text_from_row(row['categories'], row['characteristic_attributes_mapping'])
        for _, row in df.iterrows()
    ]
    
    return df

def merge_data(test, resnet, attr,text_and_bert):
    df = test.merge(resnet[['variantid', 'main_pic_embeddings_resnet_v1']], left_on='variantid1', right_on='variantid', how='left')
    df = df.rename(columns={'main_pic_embeddings_resnet_v1': 'pic_embeddings_1'})
    df = df.drop(columns=['variantid'])

    df = df.merge(resnet[['variantid', 'main_pic_embeddings_resnet_v1']], left_on='variantid2', right_on='variantid', how='left')
    df = df.rename(columns={'main_pic_embeddings_resnet_v1': 'pic_embeddings_2'})
    df = df.drop(columns=['variantid'])
    
    df = df.merge(attr[['variantid','combined_text']], left_on='variantid1', right_on='variantid', how='left')
    df = df.rename(columns={'combined_text': 'text_1'})
    df = df.drop(columns=['variantid'])
    
    df = df.merge(attr[['variantid','combined_text']], left_on='variantid2', right_on='variantid', how='left')
    df = df.rename(columns={'combined_text': 'text_2'})
    df = df.drop(columns=['variantid'])
    
    df = df.merge(text_and_bert, left_on='variantid1', right_on='variantid', how='left')
    df = df.rename(columns={'name': 'name_1', 'description': 'description_1', 'name_bert_64': 'name_bert_1'})
    df = df.drop(columns=['variantid'])
    
    df = df.merge(text_and_bert, left_on='variantid2', right_on='variantid', how='left')
    df = df.rename(columns={'name': 'name_2', 'description': 'description_2', 'name_bert_64': 'name_bert_2'})
    df = df.drop(columns=['variantid'])

    return df

def add_tfidf_embs(df):
    vectorizer = joblib.load('vectorizer_64.pkl')

    texts_name_desc_attr = df.apply(lambda x: [x['name_1'] + " " + x['description_1'] + " " + x['text_1'], x['name_2'] + " " + x['description_2'] + " " + x['text_2']], axis=1).tolist()

    tf_feat_name_desc_attr = vectorizer.transform([' '.join(text) for text in texts_name_desc_attr]).toarray()

    df['tf_feat_name_desc'] = list(tf_feat_name_desc_attr)
    return df


def flatten_with_new_embs(df):
    embs_pic_cols_1 = [f'pic_emb_1_{i}' for i in range(128)]
    embs_pic_cols_2 = [f'pic_emb_2_{i}' for i in range(128)]

    new_embs_cols = [f'tfidf_emb_{i}' for i in range(64)]

    name_bert_cols_1 = [f'name_bert_1_{i}' for i in range(64)]
    name_bert_cols_2 = [f'name_bert_2_{i}' for i in range(64)]
    list_emb_pic_1= []
    for i in range(df.shape[0]):
        list_emb_pic_1.append(df['pic_embeddings_1'].iloc[i][0])
    
    list_emb_pic_2= []
    for i in range(df.shape[0]):
        list_emb_pic_2.append(df['pic_embeddings_2'].iloc[i][0])

    pic_data_1 = pd.DataFrame(list_emb_pic_1, columns=embs_pic_cols_1)
    pic_data_2 = pd.DataFrame(list_emb_pic_2, columns=embs_pic_cols_2)
    new_emb_data_cols = pd.DataFrame(df['tf_feat_name_desc'].tolist(), columns=new_embs_cols)
    name_bert_data_1 = pd.DataFrame(df['name_bert_1'].tolist(), columns=name_bert_cols_1)
    name_bert_data_2 = pd.DataFrame(df['name_bert_2'].tolist(), columns=name_bert_cols_2)

    df = df[['variantid1', 'variantid2']]
    dataset = pd.concat([df, name_bert_data_1, name_bert_data_2,pic_data_1,pic_data_2, new_emb_data_cols], axis=1)

    return dataset

def remove_tags(text):
    if text == "None":
        return ""  
    text = re.sub(r'<br\s*/?>', '', text)
    text = re.sub(r'<li\s*/?>', '', text)
    text = re.sub(r'&laquo;|&raquo;|&minus;', '', text)
    return text

def len_of_line(text):
    return len(text.split())

def make_json(text):
    return json.loads(text)


def create_new_features(text_df, attrib):
    text_df = pd.merge(text_df, attrib, on='variantid', how='outer')
    text_df['description'] = text_df['description'].astype(str).apply(remove_tags)
    text_df['len_of_sents'] = text_df['description'].apply(len_of_line)
    text_df['categories'] = text_df['categories'].apply(make_json)
    text_df['characteristic_attributes_mapping'] = text_df['characteristic_attributes_mapping'].apply(make_json)

    cats = {}
    for i in range(len(text_df['categories'])):
        category = text_df['categories'][i]['2']
        if category not in cats:
            cats[category] = 1
        else:
            cats[category] += 1
            
    cats = dict(sorted(cats.items(), key=lambda item: item[1], reverse=True))

    token2id_type = {}
    tokens = list(cats.keys())
    for i in range(len(list(cats.keys()))):
        token2id_type[tokens[i]] = i + 1

    cats = {}
    for i in range(len(text_df['categories'])):
        category = text_df['categories'][i]['1']
        if category not in cats:
            cats[category] = 1
        else:
            cats[category] += 1
        
    cats = dict(sorted(cats.items(), key=lambda item: item[1], reverse=True))


    token2id_type_2 = {}
    tokens = list(cats.keys())
    for i in range(len(list(cats.keys()))):
        token2id_type_2[tokens[i]] = i + 1
    cats = {}

    for i in range(len(text_df['categories'])):
        category = text_df['categories'][i]['3']
        if category not in cats:
            cats[category] = 1
        else:
            cats[category] += 1
            
    cats = dict(sorted(cats.items(), key=lambda item: item[1], reverse=True))


    token2id_type_3 = {}
    tokens = list(cats.keys())
    for i in range(len(list(cats.keys()))):
        token2id_type_3[tokens[i]] = i + 1

    def extract_category_1(category_dict):
        return token2id_type_2[category_dict.get("1", None)]

    def extract_category_2(category_dict):
        return token2id_type[category_dict.get("2", None)]

    def extract_category_3(category_dict):
        return token2id_type_3[category_dict.get("3", None)]
    
    text_df['categories_main'] = text_df['categories'].apply(extract_category_2)

    text_df['categories_1'] = text_df['categories'].apply(extract_category_1)

    text_df['categories_2'] = text_df['categories'].apply(extract_category_3)

    cat_stand_2 = text_df['categories_main'].tolist()
    standartezer = preprocessing.StandardScaler()

    cat_stand_2 = standartezer.fit_transform([[x] for x in cat_stand_2])
    cat_stand_2 = [item[0] for item in cat_stand_2]

    cat_stand_3 = text_df['categories_2'].tolist()
    standartezer = preprocessing.StandardScaler()

    cat_stand_3 = standartezer.fit_transform([[x] for x in cat_stand_2])
    cat_stand_3 = [item[0] for item in cat_stand_3]

    len_of_sents = text_df['len_of_sents'].tolist()
    standartezer = preprocessing.StandardScaler()
    len_of_sents = standartezer.fit_transform([[x] for x in len_of_sents])
    len_of_sents = [item[0] for item in len_of_sents]

    text_df['categories_main'] = cat_stand_2
    text_df['categories_2'] = cat_stand_3
    text_df['len_of_sents'] = len_of_sents
    return text_df[['variantid', 'len_of_sents', 'categories_main', 'categories_1', 'categories_2']]

def final_df(data, new_features):
    data = data.merge(new_features, left_on='variantid1', right_on='variantid', how='left')
    data = data.drop(['variantid'], axis=1)
    data = data.merge(new_features, left_on='variantid2', right_on='variantid', how='left')
    data = data.drop(['variantid'], axis=1)
    return data


def predict(data):
    model = CatBoostClassifier()
    model.load_model("catboost_model_default.cbm")
    data = data.drop(['variantid1', 'variantid2'], axis=1)
    
    probabilities = model.predict_proba(data)
    
    preds = (probabilities[:, 1] >= 0.55).astype(int)
    
    return preds

# def predict(data):
#     model = CatBoostClassifier()
#     model.load_model("catboost_model_default.cbm")
#     data = data.drop(['variantid1', 'variantid2'], axis=1)
#     preds = model.predict(data)
#     return preds


def main():

    attr, res_emb, text_and_bert, test = load_data()

    new_features = create_new_features(text_and_bert, attr)
    attr = process_attr(attr)
    attr = attr[['variantid', 'combined_text']]

    data = merge_data(test, res_emb, attr, text_and_bert)
    data = data.fillna("unk")
    data = add_tfidf_embs(data)
    data = flatten_with_new_embs(data)
    data = final_df(data, new_features)
    del new_features
    gc.collect()
    preds = predict(data)
    submission = pd.DataFrame({
        'variantid1': test['variantid1'],
        'variantid2': test['variantid2'],
        'target': preds
    })
    
    submission.to_csv('./data/submission.csv', index=False)

    
if __name__ == '__main__':
    main()

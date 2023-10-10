import pandas as pd
import numpy as np
import nltk
import re
import pandas as pd
from sklearn import feature_extraction, model_selection, pipeline,feature_selection
from textblob import TextBlob
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle
import os

nltk.download('wordnet')
nltk.download("stopwords")

df_ar = pd.read_json('./recipes_raw_nosource_ar.json',orient='index')
df_epi = pd.read_json('./recipes_raw_nosource_epi.json', orient='index')
df_fn = pd.read_json('./recipes_raw_nosource_fn.json', orient='index')

data =  pd.concat([df_ar,df_epi, df_fn])
data = data.reset_index()
data = data.drop(columns=['picture_link', 'index'])

data.dropna(inplace=True)

df_incredients_train = pd.read_json('./train.json')

df_incredients_test = pd.read_json('./test.json')

df_ingredients = pd.concat([df_incredients_train,df_incredients_test],axis=0)

additional_stop_words = ["advertisement", "advertisements",'ADVERTISEMENT'
                         "cup", "cups",
                         "tablespoon", "tablespoons", 
                         "teaspoon", "teaspoons", 
                         "ounce", "ounces",
                         "salt", 
                         "pepper", 
                         "pound", "pounds",
                         ]

def preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()

    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## back to string from list
    text = " ".join(lst_text)

    ## Remove digits
    text = ''.join([i for i in text if not i.isdigit()])

    ## remove mutliple space
    text = re.sub(' +', ' ', text)

    return text

def processing(row):
  ls = row['ingredients']
  return ' '.join(ls)

dataset_cuisine_ingredients = df_ingredients

dataset_cuisine_ingredients['ingredients'] = dataset_cuisine_ingredients.apply(lambda x: processing(x), axis=1)

dataset_cuisine_ingredients.dropna(inplace=True)
dataset_cuisine_ingredients = dataset_cuisine_ingredients.drop(columns=['id']).reset_index(drop=True)

stop_word_list = nltk.corpus.stopwords.words("english")

# Extend list of stop words
stop_word_list.extend(additional_stop_words)

dataset_cuisine_ingredients["ingredients_query"] = dataset_cuisine_ingredients["ingredients"].apply(lambda x: preprocess_text(x, flg_stemm=False, flg_lemm=True, lst_stopwords=stop_word_list))

 ## Tf-Idf (advanced variant of BoW)
vectorizer = feature_extraction.text.TfidfVectorizer(max_features=10000, ngram_range=(1,2))
corpus = dataset_cuisine_ingredients["ingredients_query"]
vectorizer.fit(corpus)

embedded_ingredients = vectorizer.transform(corpus)

dic_vocabulary = vectorizer.vocabulary_

dataset_cuisine_ingredients['cuisine'].unique()
    

# These should not be removed 
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')

# configure_plotly_browser_state()
blob = TextBlob(str(dataset_cuisine_ingredients['ingredients']))

labels = dataset_cuisine_ingredients["cuisine"]

# names = vectorizer.get_feature_names(labels)
names = vectorizer.get_feature_names_out()

p_value_limit = 0.95
        
dtf_features = pd.DataFrame(columns=["feature", "score", "labels"])

for cat in np.unique(labels):
    chi2, p = feature_selection.chi2(embedded_ingredients, labels==cat)
    new_data = pd.DataFrame({
        "feature": names,
        "score": 1 - p,
        "labels": cat
    })
    
    # Filter based on p_value_limit
    new_data = new_data[new_data["score"] > p_value_limit]
    
    # Concatenate new_data with dtf_features
    dtf_features = pd.concat([dtf_features, new_data], ignore_index=True)

# Sort the final DataFrame
dtf_features = dtf_features.sort_values(["labels", "score"], ascending=[True, False])


dtf_features["labels"].unique()
x = dtf_features["labels"].unique()

dict_label = {}
for cat in x:
  dict_label[cat] = dtf_features[dtf_features["labels"]==cat]["feature"].values[:10]

names = dtf_features["feature"].unique().tolist()


stop_word_list = nltk.corpus.stopwords.words("english")

# Extend list of stop words
stop_word_list.extend(additional_stop_words)

data["ingredients_query"] = data["ingredients"].apply(lambda x: preprocess_text(x, flg_stemm=False, flg_lemm=True, lst_stopwords=stop_word_list))

MODEL_PATH = "models/nlp"
MODEL_EMBEDDINGS_PATH = os.path.join(MODEL_PATH, 'similarity_embeddings')
CUISINE_CLASSES = ['brazilian','british','cajun_creole','chinese','filipino','french','greek','indian','irish','italian','jamaican','japanese','korean','mexican','moroccan','russian','southern_us','spanish','thai','vietnamese']
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(MODEL_EMBEDDINGS_PATH, exist_ok=True)

df_cuisine_ingredients = dataset_cuisine_ingredients

vectorizer = feature_extraction.text.TfidfVectorizer()

classifier = LogisticRegressionCV(cv=3, solver='saga', random_state=42, n_jobs=-1, verbose=0)

# pipeline
model = pipeline.Pipeline([("vectorizer", vectorizer),  
                                ("classifier", classifier)])

# Split the dataset
df_cuisine_ingredients_train, df_cuisine_ingredients_test = model_selection.train_test_split(df_cuisine_ingredients, test_size=0.3, random_state=42)
X_train = df_cuisine_ingredients_train['ingredients_query']; 
X_test = df_cuisine_ingredients_test['ingredients_query'];
y_train = df_cuisine_ingredients_train['cuisine']; 
y_test = df_cuisine_ingredients_test['cuisine']; 
# train classifier
model.fit(X_train, y_train)

predicted = model.predict(X_test)

predicted_prob = model.predict_proba(X_test)

classes = np.unique(y_test)

accuracy = metrics.accuracy_score(y_test, predicted)

def save_pkl(file, pkl_filename):
    with open(pkl_filename, 'wb') as pkl_file:
        pickle.dump(file, pkl_file)

save_pkl(model, os.path.join(MODEL_PATH, "pickle_model.pkl"))

def load_pkl(pkl_filename):
    with open(pkl_filename, 'rb') as pkl_file:
        return pickle.load(pkl_file)
    
df_main_recipe = data

model = load_pkl(os.path.join(MODEL_PATH, 'pickle_model.pkl'))
df_main_recipe["cuisine"] = model.predict(data["ingredients_query"].tolist())

data = df_main_recipe

def get_tokenize_text(input_text):
    # list of stopwords
    stop_word_list = nltk.corpus.stopwords.words("english")

    # Extend list of stop words
    stop_word_list.extend(additional_stop_words)

    return preprocess_text(input_text, flg_stemm=False, flg_lemm=True, lst_stopwords=stop_word_list)

def d2v_embeddings(data):
  data = data['ingredients_query'].tolist()
  tagged_data = [TaggedDocument(words=row.split(), tags=[str(index)]) for index, row in enumerate(data)]
  model = Doc2Vec(tagged_data, vector_size=50, window=2, min_count=1, workers=4, epochs = 100,min_alpha=0.00025,dm =1)
  return model
    

for cuisine in CUISINE_CLASSES:
    data_cuisine = data[data['cuisine'] == cuisine]
    print(data_cuisine.iloc[:,0:4])
    model_embedding = d2v_embeddings(data_cuisine.iloc[:,0:4])
    save_pkl(model_embedding, os.path.join(MODEL_EMBEDDINGS_PATH, f'd2v_{cuisine}.pkl'))
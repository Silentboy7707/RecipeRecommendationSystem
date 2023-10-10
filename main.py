import re
import nltk
import os
import pickle
from app import data

MODEL_PATH = "models/nlp"
MODEL_EMBEDDINGS_PATH = os.path.join(MODEL_PATH, 'similarity_embeddings')



additional_stop_words = ["advertisement", "advertisements",'ADVERTISEMENT'
                         "cup", "cups",
                         "tablespoon", "tablespoons", 
                         "teaspoon", "teaspoons", 
                         "ounce", "ounces",
                         "salt", 
                         "pepper", 
                         "pound", "pounds",
                         ]

def load_pkl(pkl_filename):
    with open(pkl_filename, 'rb') as pkl_file:
        return pickle.load(pkl_file)

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

def get_tokenize_text(input_text):
    # list of stopwords
    stop_word_list = nltk.corpus.stopwords.words("english")

    # Extend list of stop words
    stop_word_list.extend(additional_stop_words)

    return preprocess_text(input_text, flg_stemm=False, flg_lemm=True, lst_stopwords=stop_word_list)

#The function to predict the cuisine
def predict_cuisine(input_text):
    top = 3
    
    # Tokenize text
    tokenize_text = get_tokenize_text(input_text)
    
    # Get model
    model_path = os.path.join(MODEL_PATH, 'pickle_model.pkl')
    model = load_pkl(model_path)
    
    # Tokenize text
    tokenize_text = get_tokenize_text(input_text)

    # Get classes ordered by probability
    proba = model.predict_proba([tokenize_text])[0]

    # Sorted index list 
    indexes = sorted(range(len(proba)), key=lambda k: proba[k], reverse=True)

    # Get cuisine
    cuisine_labels = model.classes_.tolist()
    cusine_ordered = [cuisine_labels[ind] for ind in indexes]

    return cusine_ordered[:top]


ingredients_list = ['rice','bread','egg','onion']
ingredients_list2 = ['coffee','sugar']
text_u = ' '.join(ingredients_list)
text2 =  ' '.join(ingredients_list2)
cusines = predict_cuisine(text_u)
tokenize_text = get_tokenize_text(text_u).split()
    
cuisine = 'indian'

  # Load model from the selected cuisine
d2v = load_pkl(os.path.join(MODEL_EMBEDDINGS_PATH, f'd2v_{cuisine}.pkl'))
print(os.path.exists(os.path.join(MODEL_EMBEDDINGS_PATH, f'd2v_{cuisine}.pkl')))
print(d2v)
# Get embeddings
# print(tokenize_text)
embeddings = d2v.infer_vector(tokenize_text)
best_recipes = d2v.docvecs.most_similar([embeddings]) #gives you top 10 document tags and their cosine similarity

# Get recipes
best_recipes_index = [int(output[0]) for output in best_recipes]
best_recipes_index = [int(output[0]) for output in best_recipes]

data_recipes = data.iloc[:,0:4]
recipes_data = data_recipes[data_recipes.index.isin(best_recipes_index)].head(5)

sep = '\n\n'
for index, row in recipes_data.iterrows():
  title = 'Title: ' + row['title'] 
  ingredients=''
  list_ing = str(row['ingredients']).replace('ADVERTISEMENT', '').strip('][').split(', ')
  for ingredient in list_ing:
      ingredients += ingredient.replace("'", "") + '\n'
  ingredients = 'Ingredients: ' + '\n' + ingredients
  instructions = 'Instruction: '+ '\n' + row['instructions']

  txt = title + sep + ingredients + sep + instructions
  print(txt)

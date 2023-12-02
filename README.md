# README

## Analyse de sentiment pour les critiques de restaurants

### Aperçu

Ce code effectue une analyse de sentiment sur les critiques de restaurants en utilisant le langage de programmation Python. L'analyse est basée sur le sentiment des mots individuels dans les critiques. Le code utilise la bibliothèque SpaCy pour le traitement du langage naturel, la bibliothèque NLTK pour diverses tâches de traitement du langage naturel, et scikit-learn pour l'apprentissage automatique.

### Structure du Code

1. **Analyse XML :**
   - Le code commence par analyser des fichiers XML contenant des critiques de restaurants en utilisant la bibliothèque ElementTree.

```python
import spacy
import pandas as pd
import xml.etree.ElementTree as ET
from nltk.stem import WordNetLemmatizer

# Analyse des données d'entraînement
tree1 = ET.parse("./Restaurants_Train.xml")
root1 = tree1.getroot()

# Extraction des informations des données d'entraînement
data = []
for sentence in root1.findall('sentence'):
    # Extraction des attributs de chaque aspectTerm dans la phrase
    idi = sentence.get('id')
    text = sentence.find('text').text
    for neighbor in sentence.iter('aspectTerm'):
        neighbor.attrib['idi'] = idi
        neighbor.attrib['text'] = text
        data.append(neighbor.attrib)

# Création d'un DataFrame à partir des données extraites
df1 = pd.DataFrame(data)
df1.head(1)
```

2. **Tokenisation et étiquetage Partie-du-Discours (PoS) :**
   - Le code tokenize le texte à l'aide de SpaCy et extrait les étiquettes Partie-du-Discours (PoS) pour chaque jeton.

```python
# Tokenisation à l'aide de SpaCy
nlp = spacy.load('fr_core_news_sm')  # Charger le modèle SpaCy français
df1['token_text'] = df1.apply(lambda row: nlp(row["text"]), axis=1)

# Extraction des étiquettes PoS
pos_tags = []
for doc in df1['token_text']:
    pos_tags_doc = [(token.text, token.pos_) for token in doc]
    pos_tags.append(pos_tags_doc)
print(pos_tags)
```

3. **Analyse de sentiment :**
   - Le code calcule les scores de sentiment pour chaque mot en utilisant WordNet et SentiWordNet.

```python
# Analyse de sentiment en utilisant WordNet et SentiWordNet
# ...

# Insérer les colonnes sentiment_word et score_word dans le DataFrame
df1.insert(4, "sentiment_word", " ")
df1.insert(5, "score_word", " ")
# ...
```

4. **Extraction de fonctionnalités :**
   - Le code extrait des fonctionnalités du texte en utilisant la vectorisation TF-IDF.

```python
# Extraction de fonctionnalités en utilisant la vectorisation TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    min_df=5,
    max_df=0.8,
    sublinear_tf=True,
    use_idf=True
)

train = vectorizer.fit_transform(df1["text"])
test = vectorizer.transform(df2["text"])
```

5. **Classification de sentiment :**
   - Le code utilise un classificateur de machine learning SVM (Support Vector Machine) pour classer les sentiments et évalue les performances.

```python
# Classification de sentiment en utilisant SVM
from sklearn import svm
import time
from sklearn.metrics import classification_report

classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train, df1['polarity'])
t1 = time.time()
predecition_linear = classifier_linear.predict(test)
t2 = time.time()
```

### Dépendances

- Assurez-vous d'avoir les bibliothèques Python requises installées en utilisant les commandes suivantes :

```bash
pip install spacy pandas nltk matplotlib scikit-learn
```

- Téléchargez les ressources nécessaires pour SpaCy et NLTK :

```bash
python -m spacy download fr_core_news_sm
python -m nltk.downloader omw-1.4 averaged_perceptron_tagger wordnet sentiwordnet punkt stopwords
```

### Utilisation

1. Placez vos fichiers XML (par exemple, `Restaurants_Train.xml` et `Restaurants_Test_Gold.xml`) dans le même répertoire que le script.
2. Exécutez le script, il effectuera une analyse de sentiment sur les critiques de restaurants fournies.

### Licence

Ce code est fourni sous la [Licence MIT](LICENSE). N'hésitez pas à le modifier et à le distribuer selon vos besoins. Si vous trouvez ce code utile, veuillez envisager de donner du crédit en liant vers ce référentiel.

### Remerciements

- Le code utilise diverses bibliothèques et ressources, notamment SpaCy, NLTK et scikit-learn. Un merci particulier à leurs contributeurs respectifs.

Pour toute question ou problème, veuillez créer une issue GitHub ou contacter [khadijabendib41@gmail.com].

Joyeux codage!
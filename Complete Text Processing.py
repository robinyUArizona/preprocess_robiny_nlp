#!/usr/bin/env python
# coding: utf-8

# ## Complete Text Processing 

# ### General Feature Extraction
# - File loading
# - Word counts
# - Characters count
# - Average characters per word
# - Stop words count
# - Count #HashTags and @Mentions
# - If numeric digits are present in twitts
# - Upper case word counts

# ### Preprocessing and Cleaning
# - Lower case
# - Contraction to Expansion
# - Emails removal and counts
# - URLs removal and counts
# - Removal of RT
# - Removal of Special Characters
# - Removal of multiple spaces
# - Removal of HTML tags
# - Removal of accented characters
# - Removal of Stop Words
# - Conversion into base form of words
# - Common Occuring words Removal
# - Rare Occuring words Removal
# - Word Cloud
# - Spelling Correction
# - Tokenization
# - Lemmatization
# - Detecting Entities using NER
# - Noun Detection
# - Language Detection
# - Sentence Translation
# - Using Inbuilt Sentiment Classifier

# In[1]:


import pandas as pd
import numpy as np
import spacy


# In[2]:


from spacy.lang.en.stop_words import STOP_WORDS as stopwords


# In[3]:


df = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/twitter-data/master/twitter4000.csv', encoding = 'latin1')


# In[4]:





# In[5]:


df['sentiment'].value_counts()


# ## Word Counts

# In[6]:


len('this is text'.split())


# In[7]:


df['word_counts'] = df['twitts'].apply(lambda x: len(str(x).split()))


# In[8]:


df.sample(5)


# In[9]:


df['word_counts'].max()


# In[10]:


df['word_counts'].min()


# In[11]:


df[df['word_counts']==1]


# # Characters Count

# In[12]:


len('this is')


# In[13]:


def char_counts(x):
    s = x.split()
    x = ''.join(s)
    return len(x)


# In[14]:


char_counts('this is')


# In[15]:


df['char_counts'] = df['twitts'].apply(lambda x: char_counts(str(x)))


# In[16]:


df.sample(5)


# ## Average Word Length

# In[17]:


x = 'this is' # 6/2 = 3
y = 'thankyou guys' # 12/2 = 6


# In[18]:


df['avg_word_len'] = df['char_counts']/df['word_counts']


# In[19]:


df.sample(4)


# ## Stop Words Count 

# In[20]:


print(stopwords)


# In[21]:


len(stopwords)


# In[22]:


x = 'this is the text data'


# In[23]:


x.split()


# In[24]:


[t for t in x.split() if t in stopwords]


# In[25]:


len([t for t in x.split() if t in stopwords])


# In[26]:


df['stop_words_len'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t in stopwords]))


# In[27]:


df.sample(5)


# In[ ]:





# In[ ]:





# In[ ]:





# ## Count #HashTags and @Mentions 

# In[28]:


x = 'this is #hashtag and this is @mention'


# In[29]:


x.split()


# In[30]:


[t for t in x.split() if t.startswith('@')]


# In[31]:


len([t for t in x.split() if t.startswith('@')])


# In[32]:


df['hashtags_count'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.startswith('#')]))


# In[33]:


df['mentions_count'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.startswith('@')]))


# In[34]:


df.sample(5)


# In[ ]:





# ## If numeric digits are present in twitts

# In[35]:


x = 'this is 1 and 2'


# In[36]:


x.split()


# In[37]:


x.split()[3].isdigit()


# In[38]:


[t for t in x.split() if t.isdigit()]


# In[39]:


df['numerics_count'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.isdigit()]))


# In[40]:


df.sample(5)


# ## UPPER case words count 

# In[41]:


x = 'I AM HAPPY'
y = 'i am happy'


# In[42]:


[t for t in x.split() if t.isupper()]


# In[43]:


df['upper_counts'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.isupper()]))


# In[44]:


df.sample(5)


# In[45]:


df.iloc[3962]['twitts']


# In[ ]:





# In[ ]:





# # Preprocessing and Cleaning

# ## Lower Case Conversion 

# In[46]:


x = 'this is Text'


# In[47]:


x.lower()


# In[48]:


x = 45.0
str(x).lower()


# In[49]:


df['twitts'] = df['twitts'].apply(lambda x: str(x).lower())


# In[50]:


df.sample(5)


# ## Contraction to Expansion 

# In[51]:


contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how does",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
" u ": " you ",
" ur ": " your ",
" n ": " and ",
"won't": "would not",
'dis': 'this',
'bak': 'back',
'brng': 'bring'}


# In[52]:


x = "i'm don't he'll" # "i am do not he will"


# In[53]:


def cont_to_exp(x):
    if type(x) is str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x
    else:
        return x
    


# In[54]:


cont_to_exp(x)


# In[55]:


get_ipython().run_cell_magic('timeit', '', "df['twitts'] = df['twitts'].apply(lambda x: cont_to_exp(x))")


# In[56]:


df.sample(5)


# In[ ]:





# ## Count and Remove Emails 

# In[57]:


import re


# In[58]:


df[df['twitts'].str.contains('hotmail.com')]


# In[59]:


df.iloc[3713]['twitts']


# In[60]:


x = '@securerecs arghh me please  markbradbury_16@hotmail.com'


# In[61]:


re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)', x)


# In[62]:


df['emails'] = df['twitts'].apply(lambda x: re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)', x))


# In[63]:


df['emails_count'] = df['emails'].apply(lambda x: len(x))


# In[64]:


df[df['emails_count']>0]


# In[65]:


re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x)


# In[66]:


df['twitts'] = df['twitts'].apply(lambda x: re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x))


# In[67]:


df[df['emails_count']>0]


# In[ ]:





# ## Count URLs and Remove it 

# In[68]:


x = 'hi, thanks to watching it. for more visit https://youtube.com/kgptalkie'


# In[69]:


#shh://git@git.com:username/repo.git=riif?%


# In[70]:


re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)


# In[71]:


df['url_flags'] = df['twitts'].apply(lambda x: len(re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)))


# In[72]:


df[df['url_flags']>0].sample(5)


# In[73]:


x


# In[74]:


re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , x)


# In[75]:


df['twitts'] = df['twitts'].apply(lambda x: re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , x))


# In[76]:


df.sample(5)


# In[ ]:





# ## Remove RT 

# In[77]:


df[df['twitts'].str.contains('rt')]


# In[78]:


x = 'rt @username: hello hirt'


# In[79]:


re.sub(r'\brt\b', '', x).strip()


# In[80]:


df['twitts'] = df['twitts'].apply(lambda x: re.sub(r'\brt\b', '', x).strip())


# In[ ]:





# In[ ]:





# In[ ]:





# ## Special Chars removal or punctuation removal 

# In[81]:


df.sample(3)


# In[82]:


x = '@duyku apparently i was not ready enough... i...'


# In[83]:


re.sub(r'[^\w ]+', "", x)


# In[84]:


df['twitts'] = df['twitts'].apply(lambda x: re.sub(r'[^\w ]+', "", x))


# In[85]:


df.sample(5)


# In[ ]:





# ## Remove multiple spaces `"hi   hello    "`

# In[86]:


x =  'hi    hello     how are you'


# In[87]:


' '.join(x.split())


# In[88]:


df['twitts'] = df['twitts'].apply(lambda x: ' '.join(x.split()))


# In[ ]:





# ## Remove HTML tags

# In[89]:


get_ipython().system('pip install beautifulsoup4')


# In[90]:


from bs4 import BeautifulSoup


# In[91]:


x = '<html><h1> thanks for watching it </h1></html>'


# In[92]:


x.replace('<html><h1>', '').replace('</h1></html>', '') #not rec


# In[93]:


BeautifulSoup(x, 'lxml').get_text().strip()


# In[94]:


get_ipython().run_cell_magic('time', '', "df['twitts'] = df['twitts'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text().strip())")


# In[ ]:





# ## Remove Accented Chars 

# In[95]:


x = 'Áccěntěd těxt'


# In[96]:


import unicodedata


# In[97]:


def remove_accented_chars(x):
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return x


# In[98]:


remove_accented_chars(x)


# In[99]:


df['twitts'] = df['twitts'].apply(lambda x: remove_accented_chars(x))


# In[ ]:





# In[ ]:





# In[ ]:





# ## Remove Stop Words 

# In[100]:


x = 'this is a stop words'


# In[101]:


' '.join([t for t in x.split() if t not in stopwords])


# In[102]:


df['twitts_no_stop'] = df['twitts'].apply(lambda x: ' '.join([t for t in x.split() if t not in stopwords]))


# In[103]:


df.sample(5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Convert into base or root form of word 

# In[104]:


nlp = spacy.load('en_core_web_sm')


# In[105]:


x = 'this is chocolates. what is times? this balls'


# In[106]:


def make_to_base(x):
    x = str(x)
    x_list = []
    doc = nlp(x)
    
    for token in doc:
        lemma = token.lemma_
        if lemma == '-PRON-' or lemma == 'be':
            lemma = token.text

        x_list.append(lemma)
    return ' '.join(x_list)


# In[107]:


make_to_base(x)


# In[108]:


df['twitts'] = df['twitts'].apply(lambda x: make_to_base(x))


# In[109]:


df.sample(5)


# In[ ]:





# ## Common words removal 

# In[110]:


x = 'this is this okay bye'


# In[111]:


text = ' '.join(df['twitts'])


# In[112]:


len(text)


# In[113]:


text = text.split()


# In[114]:


len(text)


# In[115]:


freq_comm = pd.Series(text).value_counts()


# In[116]:


f20 = freq_comm[:20]


# In[117]:


f20


# In[118]:


df['twitts'] = df['twitts'].apply(lambda x: ' '.join([t for t in x.split() if t not in f20]))


# In[119]:


df.sample(5)


# In[ ]:





# ## Rare words removal 

# In[120]:


rare20 = freq_comm.tail(20)


# In[121]:


df['twitts'] = df['twitts'].apply(lambda x: ' '.join([t for t in x.split() if t not in rare20]))


# In[122]:


df.sample(5)


# ## Word Cloud Visualization 

# In[123]:


# !pip install wordcloud


# In[124]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[125]:


text = ' '.join(df['twitts'])


# In[126]:


len(text)


# In[127]:


wc = WordCloud(width=800, height=400).generate(text)
plt.imshow(wc)
plt.axis('off')
plt.show()


# In[ ]:





# ## Spelling Correction 

# In[128]:


get_ipython().system('pip install -U textblob')


# In[129]:


get_ipython().system('python -m textblob.download_corpora')


# In[130]:


from textblob import TextBlob


# In[131]:


x = 'thankks forr waching it'


# In[132]:


x = TextBlob(x).correct()


# In[133]:


x


# ## Tokenization using TextBlob
# 

# In[134]:


x = 'thanks#watching this video. please like it'


# In[135]:


TextBlob(x).words


# In[136]:


doc = nlp(x)
for token in doc:
    print(token)


# In[ ]:





# ## Detecting Nouns 

# In[137]:


x = 'Breaking News: Donal Trump, the president of the USA is looking to sign a deal to mine the moon'


# In[138]:


doc = nlp(x)


# In[139]:


for noun in doc.noun_chunks:
    print(noun)


# In[ ]:





# ## Language Translation and Detection

# Language Code: https://www.loc.gov/standards/iso639-2/php/code_list.php

# In[140]:


x


# In[141]:


tb = TextBlob(x)


# In[142]:


tb.detect_language()


# In[143]:


tb.translate(to = 'zh')


# In[ ]:





# ## Use TextBlob's Inbuilt Sentiment Classifier 

# In[144]:


from textblob.sentiments import NaiveBayesAnalyzer


# In[145]:


x = 'we all stands together. we are gonna win this fight'


# In[146]:


tb = TextBlob(x, analyzer=NaiveBayesAnalyzer())


# In[147]:


tb.sentiment


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





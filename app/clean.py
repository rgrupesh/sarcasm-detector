import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def clean_text(text):

  text = text.lower()

  pattern = re.compile('https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
  text = pattern.sub('', text)
  emoji = re.compile("["
                           u"\U0001F600-\U0001FFFF"
                           u"\U0001F300-\U0001F5FF"  
                           u"\U0001F680-\U0001F6FF"  
                           u"\U0001F1E0-\U0001F1FF"  
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
  text = emoji.sub('',text)
  text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)
  return text

def token_word(dframe):

  head_line = list() 
  lines = dframe['headline'].values.tolist()

  for line in lines:
    line = clean_text(line) 
    tokenize = word_tokenize(line) 
    pure_words = [word for word in tokenize if word.isalpha()] 
    stop_words = set(stopwords.words("english")) 
    filtered_words = [ word for word in pure_words if not word in stop_words] 
    head_line.append(filtered_words) 

  return head_line
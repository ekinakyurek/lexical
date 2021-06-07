from mosestokenizer import *
engtokenizer = MosesTokenizer('en')
import jieba
import jieba.posseg as pseg
import sys
import random
from tqdm import tqdm
import re
args = sys.argv

#https://www.tutorialspoint.com/natural_language_toolkit/natural_language_toolkit_word_replacement.htm
R_patterns = [
   (r'won\'t', 'will not'),
   (r'can\'t', 'can not'),
   (r'(\w+)\'m', '\g<1> am'),
   (r'(\w+)\'ll', '\g<1> will'),
   (r'(\w+)\'d like to', '\g<1> would like to'),
   (r'(\w+)n\'t', '\g<1> not'),
   (r'(\w+)\'ve', '\g<1> have'),
   (r'(\w+)\'s', '\g<1> is'),
   (r'(\w+)\'re', '\g<1> are'),
]

class REReplacer(object):
   def __init__(self, patterns = R_patterns):
      self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
   def replace(self, text):
      s = text
      for (pattern, repl) in self.patterns:
         s = re.sub(pattern, repl, s)
      return s

rep_word = REReplacer()

lines = []
with open(args[1],"r") as f:
    for line in tqdm(f.readlines()):
         eng, chn, *_ = line.split("\t")
         eng = rep_word.replace(eng)
         chn = chn.replace("，",", ").replace('。','. ')
         eng_tokens = engtokenizer(eng.strip())
         chn_tokens = [w.word for w in pseg.cut(chn.strip())]
         lines.append(" ".join(eng_tokens) + "\t" + " ".join(chn_tokens))

random.seed(10)
random.shuffle(lines)
random.shuffle(lines)
test_len  = len(lines) // 10
train_len = len(lines) - 2*test_len

data = {"train":lines[0:train_len], "dev":lines[train_len:train_len+test_len], "test":lines[train_len+test_len:]}
for split in ("train","dev","test"):
    with open(args[1] + f"_{split}_tokenized.tsv", 'w') as f:
        for item in data[split]:
            f.write("%s\n" % item)

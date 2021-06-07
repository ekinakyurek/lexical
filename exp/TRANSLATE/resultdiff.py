import sys
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu
ARGS = sys.argv
f1 = ARGS[1]
f2 = ARGS[2]
trn_file = "../../TRANSLATE/cmn.txt_train_tokenized.tsv"
whole_data = "../../TRANSLATE/cmn.tsv"
#10:49:34 [hlog.py:49] test evaluation (greedy)/2400/label False
#10:49:34 [hlog.py:49] test evaluation (greedy)/2400/tp 2
#10:49:34 [hlog.py:49] test evaluation (greedy)/2400/fp 4
#10:49:34 [hlog.py:49] test evaluation (greedy)/2400/fn 3
#10:49:34 [hlog.py:49] test evaluation (greedy)/2400/input ['Black', 'suits', 'you', '.']
#10:49:34 [hlog.py:49] test evaluation (greedy)/2400/gold ['黑色', '很', '衬', '你', '.']
#10:49:34 [hlog.py:49] test evaluation (greedy)/2400/pred ['把', '条', '条', '条', '你', '.']


with open(trn_file,"r") as handle:
    trn_data = handle.readlines()
    trn_data = [res.strip().split("\t") for res in trn_data]

print("trn_data")
with open(whole_data,"r") as handle:
    whole_data = handle.readlines()
    whole_data = [res.strip().split("\t") for res in whole_data]
print("whole data")

references = {}
for (inp,out) in whole_data:
    inp = inp.strip()
    out = out.strip().split(" ")
    if inp in references:
        references[inp].append(out)
    else:
        references[inp] = [out]
print("references")

counter = Counter()
for (inp,_) in trn_data:
    inp_words = inp.strip().split(" ")
    for w in inp_words:
        counter[w] += 1
print(counter)

with open(f1,"r") as handle:
    result1 = handle.readlines()
    result1 = [res.strip() for res in result1 if "test evaluation (greedy)" in res]
print("result1")

with open(f2,"r") as handle:
    result2 = handle.readlines()
    result2 = [res.strip() for res in result2 if "test evaluation (greedy)" in res]
print("result2")

ref_list = []
pred1_list = []
pred2_list = []
print("getting stats")
for i in range(0,len(result1),8):
    try:
        idx = eval(result1[i+4].split()[4].split("/")[1])
        label1 = eval(result1[i].split()[5])
        label2 = eval(result2[i].split()[5])
        out_words = eval("[" + result1[i+5].split(" [")[2])
        inp_words = eval("[" + result1[i+4].split(" [")[2])
        pred1_words = eval("[" + result1[i+6].split(" [")[2])
        pred2_words = eval("[" + result2[i+6].split(" [")[2])
        out = " ".join(out_words)
        inp = " ".join(inp_words)
        pred1  = " ".join(pred1_words)
        pred2  = " ".join(pred2_words)
    except:
        print("parsing error")

    cur_references = references[inp]
    if len([w for w in inp_words if counter[w] == 1]) > 0:
        ref_list.append(cur_references)
        pred1_list.append(pred1_words)
        pred2_list.append(pred2_words)

    isdifferent = label1 != label2

    if isdifferent and label1:
        if label1 and not label2:
            print(f"{idx}:+++ ")
        elif label2 and not label1:
             print(f"{idx}:--- ")
        print(f"inp: {inp}")
        print(f"out: {out}")
        print(f"pred1: {pred1}")
        print(f"pred2: {pred2}")






bleu_score1 = corpus_bleu(ref_list, pred1_list)
bleu_score2 = corpus_bleu(ref_list, pred2_list)
print("bleu_score1 ", bleu_score1)
print("bleu_score2: ", bleu_score2)
print(len(pred1_list))

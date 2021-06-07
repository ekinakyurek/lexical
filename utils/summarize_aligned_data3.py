import sys
from collections import Counter
import pdb
import pickle, json
EPS=3 #3
SPLIT_TOK=" ||| "
#SPLIT_TOK="\t"
def main(data_file, aligner_file):
    data = None
    with open(data_file, "r") as f:
        data = f.read().splitlines()

    word_alignment = {}
    inputs  = []
    outputs = []
    #with open(aligner_file, "r") as f:
    for k, line in enumerate(data):
        input, output = line.split(SPLIT_TOK)
        input, output = set(input.strip().split(" ")), set(output.strip().split(" "))
        inputs.append(input)
        outputs.append(output)
        for inp in input:
            if inp not in word_alignment:
                word_alignment[inp] = {}
            inpmap = word_alignment[inp]
            for out in output:
                if out not in inpmap:
                    inpmap[out] = 1
                else:
                    inpmap[out] = inpmap[out] + 1

    for i in range(len(inputs)):
        input = inputs[i]
        output = outputs[i]
        for k in input:
            for v in list(word_alignment[k].keys()):
                if v not in output:
                    del word_alignment[k][v]
            # else:
            #     for v in list(word_alignment[k].keys()):
            #         if v in outputs[i]:
            #             del word_alignment[k][v]

    incoming = {}
    for (k,mapped) in list(word_alignment.items()):
        for (v,_) in mapped.items():
            if v in incoming:
                incoming[v].add(k)
            else:
                incoming[v] = {k,}

        # if len(word_alignment[k]) == 0:
        #     del word_alignment[k]
    # print(incoming["9"])
    for (v, inset) in incoming.items():
        if len(inset) > EPS:
            # print(f"common word: v: {v}, inset: {inset}")
            # print("deleting ", v)
            for (k,mapped) in list(word_alignment.items()):
                if v in mapped:
                    print(f"since EPS deleting {k}->{v}")
                    del word_alignment[k][v]

    for (v,inset) in incoming.items():
        if len(inset) > 1:
            candidates = set([e for e in inset])
            for k, line in enumerate(data):
                if len(candidates) == 0:
                    break
                input, output = line.split(SPLIT_TOK)
                input, output = set(input.strip().split(" ")), set(output.strip().split(" "))
                if v in output:
                    for e in set(candidates):
                        if e not in input:
                            candidates.remove(e)
            if len(candidates) == 1:
                wrongs = inset-candidates
                for t in wrongs:
                    if v in word_alignment[t]:
                        print(f"in candidates deleting {t}->{v}")
                        del word_alignment[t][v]

    for (k,mapped) in list(word_alignment.items()):
        if len(word_alignment[k]) == 0:
            del word_alignment[k]
        else:
            if k in mapped:
                mapped[k] += 1





    # with open(aligner_file + '.v3.pickle', 'wb') as handle:
    #     pickle.dump(word_alignment, handle)
    with open(aligner_file + '.v3.json', 'w') as handle:
        json.dump(word_alignment, handle)



if __name__ == "__main__":
    # execute only if run as a script
    assert len(sys.argv) > 2
    main(sys.argv[1], sys.argv[2])

train_file=$1
nCodes=10000
echo $train_file
train_path=$(dirname $train_file)/bpe/
dev_file=${train_file//train/dev}
test_file=${train_file//train/test}
echo $train_path
mkdir -p $train_path

fast learnbpe $nCodes $train_file > ${train_path}/codes
fast applybpe ${train_path}/train${nCodes}.tsv $train_file ${train_path}/codes
fast getvocab ${train_path}/train${nCodes}.tsv > ${train_path}//vocab${nCodes}
fast applybpe ${train_path}/dev${nCodes}.tsv $dev_file ${train_path}/codes ${train_path}/vocab${nCodes}
fast applybpe ${train_path}/test${nCodes}.tsv $test_file ${train_path}/codes ${train_path}/vocab${nCodes}

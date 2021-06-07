train_file=$1
echo $train_file
train_path=$(dirname $train_file)/alignments/
echo $train_path
mkdir -p $train_path

python utils/summarize_aligned_data3.py $train_file ${train_path}/simple.align
python utils/pmi_align.py $train_file ${train_path}/pmi.align

for flag in "-d" "-o" "-v" "-d -o" "-d -v" "-o -v" "-d -o -v" ""; do
    str_flag="${flag// /}"
    str_flag="${str_flag//-/}"
    fast_align -i $train_file $flag > ${train_path}/forward.align.${str_flag}
    python utils/summarize_aligned_data.py $train_file ${train_path}/forward.align.${str_flag}
done

for flag in "-d" "-o" "-v" "-d -o" "-d -v" "-o -v" "-d -o -v" ""; do
    str_flag="${flag// /}"
    str_flag="${str_flag//-/}"
    fast_align -i $train_file -r $flag > ${train_path}/reverse.align.${str_flag}
    python utils/summarize_aligned_data.py $train_file ${train_path}/reverse.align.${str_flag}
done

for diag in "grow-diag-final-and" "grow-diag-final" "intersect" "grow-diag"; do
    for flag in "-d" "-o" "-v" "-d -o" "-d -v" "-o -v" "-d -o -v" ""; do
	str_flag="${flag// /}"
	str_flag="${str_flag//-/}"
	atools -i ${train_path}/forward.align.${str_flag} -j ${train_path}/reverse.align.${str_flag} -c ${diag} > ${train_path}/${diag}.align.${str_flag}
	python utils/summarize_aligned_data.py $train_file ${train_path}/${diag}.align.${str_flag}
    done
done

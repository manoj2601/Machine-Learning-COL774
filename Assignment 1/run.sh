if [[ $1 == "clean" ]];
then
	rm -rf output/q1/*.*
        rm -rf output/q2/*.*
        rm -rf output/q3/*.*
        rm -rf output/q4/*.*
fi
data_dir=$1
out_dir=$2
question=$3
part=$4
if [[ ${question}_${part} == "1_a" ]];
then
		python q1/q1a.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "1_b" ]];
then
        python q1/q1b.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "1_c" ]];
then
        python q1/q1c.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "1_d" ]];
then
        python q1/q1d.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "1_e" ]];
then
        python q1/q1e.py $data_dir $out_dir
fi

if [[ ${question}_${part} == "2_a" ]];
then
		python q2/q2a.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "2_b" ]];
then
        python q2/q2b.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "2_c" ]];
then
        python q2/q2c.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "2_d" ]];
then
        python q2/q2d.py $data_dir $out_dir
fi

if [[ ${question}_${part} == "3_a" ]];
then
        python q3/q3a.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "3_b" ]];
then
        python q3/q3b.py $data_dir $out_dir
fi

if [[ ${question}_${part} == "4_a" ]];
then
	python q4/q4a.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "4_b" ]];
then
        python q4/q4b.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "4_c" ]];
then
        python q4/q4c.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "4_d" ]];
then
        python q4/q4d.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "4_e" ]];
then
        python q4/q4e.py $data_dir $out_dir
fi
if [[ ${question}_${part} == "4_f" ]];
then
        python q4/q4e.py $data_dir $out_dir
fi
#!/bin/bash
# Cora
# step 1: get test_id index list where the baseline predict correct with the truth label
python baseline.py --dataset_name 'cora' --train_percent 0.75 --seed 0 --train_epochs 300 --attack_graph 'False'

INPUT=$(pwd)/results/target_attack/cora/train_percent_0.75_corrected_test_ID_res.csv

# step 2: attack the test_id list
#
OLDIFS=$IFS
IFS=','
declare -i cn=0
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
sed 1d $INPUT | while read -r id pred
do
	echo "ID : $id"
	echo "class : $pred"
	for i in 0 1 2 3 4 5 6
  do
     if [[ $pred -ne $i ]] then
      echo "attack"
      python target_attack.py --dataset_name 'cora' --attack_graph 'False' --node_idx $id --desired_class $i --feature_attack 'True' --fix_sparsity 'False' --train_percent 0.75 --sparsity 0.5 --feat_sparsity 0.5 --seed 0 --train_epochs 300 --added_node_num 20
     fi
  done
done < $INPUT
IFS=$OLDIFS

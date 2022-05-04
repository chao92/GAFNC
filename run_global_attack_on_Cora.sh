#################################################################### Global Attack ####################################################################
# V0 Baseline version
# result save at ./results/baseline
for ((i=0.1;i<0.99;i+=0.1)) # train percent
do
  for ((j=0;j<10;j+=1)) #  seed
  do
    echo -e "baseline train percent:${i} seed ${j}"
    python baseline.py --dataset_name 'cora' --train_percent $i --seed $j --train_epochs 300
  done
done

# DAGPA learned topology and features
for ((i = 0.1; i < 0.99; i += 0.1)) # train percent
do
  for ((j = 0; j < 10; j += 1)) # seed
    do
      echo -e " DAGPA train percent:${i}  seed ${j}"
      python attack.py --dataset_name 'cora' --model_name "DAGPA" --feature_attack True --random_generate 0 --train_percent $i --fix_sparsity 'False' --seed $j --train_epochs 300 --added_node_num 20 --random_structure False --random_feature False
    done
done

# V1 Purely Random attack
#for ((i = 0.1; i < 0.99; i += 0.1)) # train percent
#do
#  for ((k = 0.5; k < 0.6; k += 0.1)) # sparsity
#  do
#  for ((j = 0; j < 10; j += 1)) # seed
#    do
#      echo -e " V1 train percent:${i}  sparsity ${k} seed ${j}"
#      python attack.py --dataset_name 'cora' --model_name "V1" --feature_attack True --train_percent $i --structure_sparsity $k --feat_sparsity $k --seed $j --train_epochs 300 --added_node_num 20 --random_structure True --random_feature True
#    done
#  done
#done

# V2 Random feature attack with learned topology
#for ((i = 0.1; i < 0.99; i += 0.1)) # train percent
#do
#  for ((k = 0.5; k < 0.6; k += 0.1)) # sparsity
#  do
#  for ((j = 0; j < 10; j += 1)) # seed
#    do
#      echo -e " V2 train percent:${i}  sparsity ${k} seed ${j}"
#      python attack.py --dataset_name 'cora'  --model_name "V2" --feature_attack True --train_percent $i --structure_sparsity $k --feat_sparsity $k --seed $j --train_epochs 300 --added_node_num 20 --random_structure False --random_feature True
#    done
#  done
#done

# V3 Random topology attack with learned features
#for ((i = 0.1; i < 0.99; i += 0.1)) # train percent
#do
#  for ((k = 0.5; k < 0.6; k += 0.1)) # sparsity
#  do
#  for ((j = 0; j < 10; j += 1)) # seed
#    do
#      echo -e " V3 train percent:${i}  sparsity ${k} seed ${j}"
#      python attack.py --dataset_name 'cora' --model_name "V3" --feature_attack True --train_percent $i --structure_sparsity $k --feat_sparsity $k --seed $j --train_epochs 300 --added_node_num 20 --random_structure True --random_feature False
#    done
#  done
#done

# V4 Random topology attack with sampled features
#for ((i = 0.1; i < 0.99; i += 0.1)) # train percent
#do
#  for ((k = 0.5; k < 0.6; k += 0.1)) # sparsity
#  do
#  for ((j = 0; j < 10; j += 1)) # seed
#    do
#      echo -e " V4 train percent:${i}  sparsity ${k} seed ${j}"
#      python attack.py --dataset_name 'cora' --model_name "V4" --feature_attack False --random_generate 0 --train_percent $i --structure_sparsity $k --feat_sparsity $k --seed $j --train_epochs 300 --added_node_num 20 --random_structure True --random_feature False
#    done
#  done
#done

# V5 learned topology attack with sampled features
#for ((i = 0.1; i < 0.99; i += 0.1)) # train percent
#do
#  for ((k = 0.5; k < 0.6; k += 0.1)) # sparsity
#  do
#  for ((j = 0; j < 10; j += 1)) # seed
#    do
#      echo -e " V5 train percent:${i}  sparsity ${k} seed ${j}"
#      python attack.py --dataset_name 'cora' --model_name "V5" --feature_attack False --random_generate 0 --train_percent $i --structure_sparsity $k --feat_sparsity $k --seed $j --train_epochs 300 --added_node_num 20 --random_structure False
#    done
#  done
#done
















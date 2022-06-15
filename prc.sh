for i in 0 1 2 3 4 
do
    python eval_randnk.py \
        -tr uniref100/uniref100_train_split_${i} \
        -te uniref100/uniref100_test_split_${i}_curate \
        -p 0.00001 -up 0.005 -s 100 -EP True ; 

    python eval_randnk.py \
        -n uniref100/uniref100_split${i}_triplet \
        -tr uniref100/uniref100_train_split_${i} \
        -te uniref100/uniref100_test_split_${i}_curate \
        -p 0.00001 -up 0.005 -s 100
done
 
# ------------------ for promiscuous enzymes ------------------ #
# python -u .\eval_randnk.py \
#     -te uniref10_prom_split_test_curate \
#     -tr uniref10_prom_split_train \
#     -n uniref10_prom_triplet \
#     -p 0.0001 -up 0.05 -s 100

#python -u .\eval_randnk.py  -te uniref100_prom_split_test_curate  -tr uniref100_prom_split_train  -EP true  -p 0.0001 -up 0.05 -s 100
 
# python -u ./train-supcon.py -t uniref10/uniref10_train_split_3 -n uniref10_split3_supcon > out/uniref10_split3_supcon.out

#python eval_randnk.py  -n uniref100/uniref100_split_4_triplet  -tr uniref100/uniref100_train_split_4  -te uniref100/uniref100_test_split_4_curate  -p 0.00001 -up 0.005 -s 100
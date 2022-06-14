# for i in 0 1 2 3 4 
# do
#     python eval_randnk.py \
#         -tr uniref10_train_split_${i} \
#         -te uniref10_test_split_${i}_curate \
#         -p 0.00001 -up 0.005 -s 100 -EP True ; 
#     python eval_randnk.py \
#         -n uniref10/uniref10_split${i}_final \
#         -tr uniref10_train_split_${i} \
#         -te uniref10_test_split_${i}_curate \
#         -p 0.00001 -up 0.005 -s 100
# done

for i in 1 
do
    python eval_randnk.py \
        -n supcon_o256_b6000_a60_l5e-4_1680 \
        -tr uniref10_train_split_${i} \
        -te uniref10_test_split_${i}_curate \
        -o 256\
        -p 0.00001 -up 0.005 -s 100
done


# for i in 0 1 2 3 4 
# do
#     python eval_randnk.py -tr uniref100/uniref100_train_split_${i} \
#         -te uniref100/uniref100_test_split_${i}_curate \
#         -p 0.00001 -up 0.005 -s 100 -EP True ; 
#     python eval_randnk.py 
#         -n uniref10/uniref10_split${i}_final \
#         -tr uniref100/uniref100_train_split_${i} \
#         -te uniref100/uniref100_test_split_${i}_curate \
#         -p 0.00001 -up 0.005 -s 100
# done


 

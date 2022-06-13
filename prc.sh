for i in 0 1 2 3 4 
do
    
    python eval_randnk.py -tr uniref100/uniref100_train_split_${i} \
        -te uniref100/uniref100_test_split_${i}_curate \
        -p 0.00001 -up 0.005 -s 100 -EP True ; 
    python eval_randnk.py -tr uniref100/uniref100_train_split_${i} \
        -te uniref100/uniref100_test_split_${i}_curate \
        -p 0.00001 -up 0.005 -s 100
 
done






python eval_randnk.py -tr uniref100/uniref100_train_split_0 -te uniref100/uniref100_test_split_0_curate -p 0.00001 -up 0.005 -s 100 -EP True
python eval_randnk.py -tr uniref100/uniref100_train_split_1 -te uniref100/uniref100_test_split_1_curate -p 0.00001 -up 0.005 -s 100 -EP True
python eval_randnk.py -tr uniref100/uniref100_train_split_2 -te uniref100/uniref100_test_split_2_curate -p 0.00001 -up 0.005 -s 100 -EP True
python eval_randnk.py -tr uniref100/uniref100_train_split_3 -te uniref100/uniref100_test_split_3_curate -p 0.00001 -up 0.005 -s 100 -EP True
python eval_randnk.py -tr uniref100/uniref100_train_split_4 -te uniref100/uniref100_test_split_4_curate -p 0.00001 -up 0.005 -s 100 -EP True
python eval_randnk.py -tr uniref100/uniref100_train_split_0 -te uniref100/uniref100_test_split_0_curate -p 0.00001 -up 0.005 -s 100
python eval_randnk.py -tr uniref100/uniref100_train_split_1 -te uniref100/uniref100_test_split_1_curate -p 0.00001 -up 0.005 -s 100
python eval_randnk.py -tr uniref100/uniref100_train_split_2 -te uniref100/uniref100_test_split_2_curate -p 0.00001 -up 0.005 -s 100
python eval_randnk.py -tr uniref100/uniref100_train_split_3 -te uniref100/uniref100_test_split_3_curate -p 0.00001 -up 0.005 -s 100
python eval_randnk.py -tr uniref100/uniref100_train_split_4 -te uniref100/uniref100_test_split_4_curate -p 0.00001 -up 0.005 -s 100
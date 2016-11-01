import ensemble_boosting as boost
import updated_reader as reader

# print('Random Order Baseline (PTB) Small Const 5 Ensembles:')
# boost.baseline(data_path = 'simple-examples/data', num_ensembles = 5, model_name = 'small', train = False, random_training_order = True)

# print('Random Order FABIMBS (PTB) Small Dropout 0.5 Const 5 Ensembles:')
# boost.ABIMBS(data_path = 'simple-examples/data', num_ensembles = 5, model_name = 'smalldrop', forward = True, train = True, random_training_order = True)

# print('Random Order BABIMBS (PTB) Small Dropout 0.5 Const 5 Ensembles:')
# boost.ABIMBS(data_path = 'simple-examples/data', num_ensembles = 5, model_name = 'smalldrop', forward = False, train = True, random_training_order = True)

# print('Random Order Baseline (20 epochs) (PTB) Small Dropout 0.5 Const 5 Ensembles:')
# boost.baseline(data_path = 'simple-examples/data', num_ensembles = 5, model_name = 'smallextradrop', train = True, random_training_order = True)

# print('Random Order Sqrt ABISS (PTB) Small Dropout 0.5 Const 5 Ensembles Ordered:')
# boost.ABISS(data_path = 'simple-examples/data', num_ensembles = 5, model_name = 'smalldrop', method = 'sqrt', train = True, random_training_order = True)

# print('Random Order Std Dev ABISS (PTB) Small Dropout 0.5 Const 5 Ensembles Ordered:')
# boost.ABISS(data_path = 'simple-examples/data', num_ensembles = 5, model_name = 'smalldrop', method = 'stddev', train = True, random_training_order = True)

# print('Random Order Unigram Partition Small 5 Ensembles Ordered:')
# boost.unigram_partition(data_path = 'simple-examples/data', num_ensembles = 9, model_name = 'small', train = False, random_training_order = True)

# print('Reverse Random Order Baseline (PTB) Small Const 5 Ensembles:')
# boost.baseline(data_path = 'simple-examples/data', num_ensembles = 5, model_name = 'small', train = True, random_training_order = True, reverse_order = True)

# print('Reverse Random Order Unigram Partition Small 9 Ensembles:')
# boost.unigram_partition(data_path = 'simple-examples/data', num_ensembles = 9, model_name = 'small', train = True, random_training_order = True, reverse_order = True)

# print('Pretrain Small 17 Ensembles 0.1 LR:')
# boost.naive_pretrain_ensemble(data_path = 'simple-examples/data', num_ensembles = 17, model_name = 'small', train = True, random_training_order = False, reverse_order = False, new_lr = 0.1)
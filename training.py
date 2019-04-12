
train_X, val_X, test_X, train_y, val_y, word_index = load_data()

embedding_matrix_1 = load_glove(word_index)
#embedding_matrix_2 = load_fasttext(word_index)
embedding_matrix_3 = load_para(word_index)

embedding_matrix = np.mean([embedding_matrix_1,  embedding_matrix_3], axis = 0)


DATA_SPLIT_SEED = 2018
clr = CyclicLR(base_lr=0.001, max_lr=0.002,
               step_size=300., mode='exp_range',
               gamma=0.99994)

train_meta = np.zeros(train_y.shape)
test_meta = np.zeros(test_X.shape[0])
splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=DATA_SPLIT_SEED).split(train_X, train_y))
for idx, (train_idx, valid_idx) in enumerate(splits):
        X_train = train_X[train_idx]
        y_train = train_y[train_idx]
        X_val = train_X[valid_idx]
        y_val = train_y[valid_idx]
        model = model_lstm_atten(embedding_matrix)
        pred_val_y, pred_test_y, best_score = train_pred(model, X_train, y_train, X_val, y_val, epochs = 8, callback = [clr,])
        train_meta[valid_idx] = pred_val_y.reshape(-1)
        test_meta += pred_test_y.reshape(-1) / len(splits)
        
        
outputs = []
pred_val_y, pred_test_y, best_score = train_pred(model_lstm_atten(embedding_matrix_1), epochs = 3)
outputs.append([pred_val_y, pred_test_y, best_score, 'model_lstm_atten only Glove'])

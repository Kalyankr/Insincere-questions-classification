def load_data():

    #Load and preprocessing data

    train_df = pd.read_csv('../input/train.csv')
    test_df = pd.read_csv('../input/test.csv')
    
    train_df['question_text'] = train_df['question_text'].str.lower()
    test_df['question_text'] = test_df['question_text'].str.lower()
    train_df['question_text'] = train_df['question_text'].apply(lambda x: clean_text(x))
    test_df['question_text'] = test_df['question_text'].apply(lambda x: clean_text(x))
    train_df['question_text'] = train_df['question_text'].apply(lambda x: clean_numbers(x))
    test_df['question_text'] = test_df['question_text'].apply(lambda x: clean_numbers(x))
    train_df['question_text'] = train_df['question_text'].apply(lambda x: replace_typical_misspell(x))
    test_df['question_text'] = test_df['question_text'].apply(lambda x: replace_typical_misspell(x))
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    # Fill up the missing values
    train_X = train_df['question_text'].fillna('_na_').values
    val_X = val_df['question_text'].fillna('_na_').values
    test_X = test_df['question_text'].fillna('_na_').values
    
    # Tokenize the sentences
    tokenizer = Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    val_X = tokenizer.texts_to_sequences(val_X)
    test_X = tokenizer.texts_to_sequences(test_X)
    
    # Pad the sentences 
    train_X = pad_sequences(train_X, maxlen=MAXLEN)
    val_X = pad_sequences(val_X, maxlen=MAXLEN)
    test_X = pad_sequences(test_X, maxlen=MAXLEN)
    
    # Get the target values
    train_y = train_df['target'].values
    val_y = val_df['target'].values

    # Shuffling the data
    np.random.seed(42)
    trn_idx = np.random.permutation(len(train_X))
    val_idx = np.random.permutation(len(val_X))
    train_X = train_X[trn_idx]
    val_X = val_X[val_idx]
    train_y = train_y[trn_idx]
    val_y = val_y[val_idx]

    return train_X, val_X, test_X, train_y, val_y, tokenizer.word_index
    
 def f1(y_true, y_pred):

    def recall(y_true, y_pred):
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())

        return recall


    def precision(y_true, y_pred):
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))

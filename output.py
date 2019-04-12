
#predicting and submission file
pred_test_y = np.mean([outputs[i][1] for i in range(len(outputs))], axis = 0)
pred_test_y = (pred_test_y > best_thresh).astype(int)


sub = pd.read_csv('../input/sample_submission.csv')
out_df = pd.DataFrame({"qid":sub["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)

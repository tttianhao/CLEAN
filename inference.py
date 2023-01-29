from src.CLEAN.infer import infer_maxsep
train_data = "split100"
test_data = "new"
infer_maxsep(train_data, test_data, report_metrics=True, pretrained=True)
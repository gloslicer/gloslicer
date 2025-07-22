from transformers import LongformerTokenizer, LongformerModel, RobertaTokenizer, RobertaModel

model_name = "allenai/longformer-base-4096"
tokenizer = LongformerTokenizer.from_pretrained(model_name)
model = LongformerModel.from_pretrained(model_name)

tokenizer.save_pretrained("./model/base_model/longformer-base-4096")
model.save_pretrained("./model/base_model/longformer-base-4096")

model_2_name = "allenai/longformer-large-4096"
tokenizer = LongformerTokenizer.from_pretrained(model_2_name)
model = LongformerModel.from_pretrained(model_2_name)

tokenizer.save_pretrained("./model/base_model/longformer-large-4096")
model.save_pretrained("./model/base_model/longformer-large-4096")

model_name = "microsoft/codebert-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)

tokenizer.save_pretrained("./model/base_model/codebert-base")
model.save_pretrained("./model/base_model/codebert-base")

model_name = "microsoft/graphcodebert-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)

tokenizer.save_pretrained("./model/base_model/graphcodebert-base")
model.save_pretrained("./model/base_model/graphcodebert-base")
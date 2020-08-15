from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
sent1 = tokenizer("Hello world")
print(sent1, sent1['input_ids'])
sent2 = tokenizer(" Hello world")
print(sent2, sent2['input_ids'])

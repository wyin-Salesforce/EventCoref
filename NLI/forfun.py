from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
sent1 = tokenizer.tokenize("Hello world")
print(sent1)
sent2 = tokenizer.tokenize(" Hello world")
print(sent2)

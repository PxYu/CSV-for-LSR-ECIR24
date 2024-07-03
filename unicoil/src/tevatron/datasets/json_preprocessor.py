class Json_TrainPreProcessor:
    def __init__(self, tokenizer, query_max_length=32, text_max_length=256, separator=' '):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        # print(example['query'])
        # print(self.tokenizer.convert_tokens_to_ids(example['query']))
        # query = self.tokenizer.convert_tokens_to_ids(example['query'])[:self.query_max_length]
        
        query = self.tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        
        positives = []
        for pos in example['positive_passages']:
            text = pos['title'] + pos['text'] if 'title' in pos else pos['text']
            positives.append(self.tokenizer.convert_tokens_to_ids(text)[:self.text_max_length])
        
        negatives = []
        for neg in example['negative_passages']:
            text = neg['title'] + neg['text'] if 'title' in neg else neg['text']
            negatives.append(self.tokenizer.convert_tokens_to_ids(text)[:self.text_max_length])

        pos_teacher_scores = example.get("positive_teacher_scores", None)
        neg_teacher_scores = example.get("negative_teacher_scores", None)
        
        return {'query': query, 'positives': positives, 'negatives': negatives, 
                'positive_teacher_scores': pos_teacher_scores,
                'negative_teacher_scores': neg_teacher_scores
                }


class Json_QueryPreProcessor:
    def __init__(self, tokenizer, query_max_length=32):
        
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length

    def __call__(self, example):
        query_id = example['query_id']
        # query = self.tokenizer.convert_tokens_to_ids(example['query'])[:self.query_max_length]
        query = self.tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        return {'text_id': query_id, 'text': query}


class Json_CorpusPreProcessor:
    def __init__(self, tokenizer, text_max_length=256, separator=' '):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        docid = example['docid']
        text = example['title'] + example['text'] if 'title' in example else example['text']
        text = self.tokenizer.convert_tokens_to_ids(text)[:self.text_max_length]
        return {'text_id': docid, 'text': text}
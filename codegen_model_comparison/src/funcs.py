import numpy


def tokenize_function(example, tokenizer, seq_length):
    return tokenizer(
        example['text'],
        padding="max_length",
        truncation=True,
        max_length=seq_length,
    )


def add_labels(example):
    example['label'] = example['input_ids']
    return example


def compute_bleu_score(preds, tokenizer, metric):
    logits = preds.predictions[0]
    preds_tok = np.argmax(logits, axis=1)
    acts = preds.label_ids

    decode_predictions = tokenizer.batch_decode(preds_tok,
                                                skip_special_tokens=True)
    decode_labels = tokenizer.batch_decode(acts, skip_special_tokens=True)

    res = metric.compute(predictions=decode_predictions,
                         references=decode_labels)
    return {'bleu_score': res['bleu']}


def compute_chrf_score(preds, tokenizer, metric):
    logits = preds.predictions[0]
    preds_tok = np.argmax(logits, axis=1)
    acts = preds.label_ids

    decode_predictions = tokenizer.batch_decode(preds_tok,
                                                skip_special_tokens=True)
    decode_labels = tokenizer.batch_decode(acts, skip_special_tokens=True)

    res = metric.compute(predictions=decode_predictions,
                         references=decode_labels)
    return {'chrf_score': res['score']}

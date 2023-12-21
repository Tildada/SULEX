import json
import argparse
import os
import sys
import random
import logging
import time
# from tqdm import tqdm
from tqdm import tqdm
import numpy as np

from shared.data_structures import Dataset
from shared.const import task_ner_labels, get_labelmap
from entity.utils import convert_dataset_to_samples, convert_dataset_to_samples_drop_ents_neg_sample, batchify, NpEncoder
from entity.models import EntityModel

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import get_constant_schedule_with_warmup
import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('root')



def save_model(model, args):
    """
    Save the model to the output directory
    """
    logger.info('Saving model to %s...' % (args.output_dir))
    model_to_save = model.bert_model.module if hasattr(model.bert_model, 'module') else model.bert_model
    model_to_save.save_pretrained(args.output_dir)
    model.tokenizer.save_pretrained(args.output_dir)


def save_model_index(model, args, index=None):
    """
    Save the model to the output directory
    """
    if index != None:
        save_path = args.output_dir + "_epoch_" + str(index)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = args.output_dir
    logger.info('Saving model to %s...' % (save_path))
    model_to_save = model.bert_model.module if hasattr(model.bert_model, 'module') else model.bert_model
    model_to_save.save_pretrained(save_path)
    model.tokenizer.save_pretrained(save_path)


def output_ner_predictions(model, batches, dataset, output_file):
    """
    Save the prediction as a json file
    """
    ner_result = {}
    span_hidden_table = {}
    tot_pred_ett = 0
    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)

        pred_ner = output_dict['pred_ner']
        pred_prob = output_dict['ner_probs']
        hidden = output_dict['ner_last_hidden']

        for sample, preds, probs, hids in zip(batches[i], pred_ner, pred_prob, hidden):
            off = sample['sent_start_in_doc'] - sample['sent_start']
            k = sample['doc_key'] + '-' + str(sample['sentence_ix'])
            ner_result[k] = []
            for span, pred, prob, hid in zip(sample['spans'], preds, probs, hids):
                span_id = '%s::%d::(%d,%d)' % (sample['doc_key'], sample['sentence_ix'], span[0] + off, span[1] + off)
                if pred == 0 and prob[0] > 0.5:
                    continue
                ner_result[k].append([span[0] + off, span[1] + off, ner_id2label[pred], prob, hid])
            tot_pred_ett += len(ner_result[k])

    logger.info('Total pred entities: %d' % tot_pred_ett)

    js = dataset.js
    for i, doc in enumerate(js):
        doc["predicted_ner"] = []
        doc["predicted_relations"] = []
        for j in range(len(doc["sentences"])):
            k = doc['doc_key'] + '-' + str(j)
            if k in ner_result:
                doc["predicted_ner"].append(ner_result[k])
            else:
                logger.info('%s not in NER results!' % k)
                doc["predicted_ner"].append([])

            doc["predicted_relations"].append([])

        js[i] = doc

    logger.info('Output predictions to %s..' % (output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(doc, cls=NpEncoder) for doc in js))








def output_ner_predictions_0709(model, batches, dataset, output_file):
    """
    Save the prediction as a json file
    """
    ner_result = {}
    span_hidden_table = {}
    tot_pred_ett = 0
    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)

        pred_ner = output_dict['pred_ner']
        pred_prob = output_dict['ner_probs']
        hidden = output_dict['ner_last_hidden']

        for sample, preds, probs, hids in zip(batches[i], pred_ner, pred_prob, hidden):
            off = sample['sent_start_in_doc'] - sample['sent_start']
            k = sample['doc_key'] + '-' + str(sample['sentence_ix'])
            ner_result[k] = []
            for span, pred, prob, hid in zip(sample['spans'], preds, probs, hids):
                span_id = '%s::%d::(%d,%d)' % (sample['doc_key'], sample['sentence_ix'], span[0] + off, span[1] + off)
                if pred == 0 and prob[0] > 0.5:
                    continue
                ner_result[k].append([span[0] + off, span[1] + off, ner_id2label[pred], prob[pred], hid])
            tot_pred_ett += len(ner_result[k])

    logger.info('Total pred entities: %d' % tot_pred_ett)

    js = dataset.js
    for i, doc in enumerate(js):
        doc["predicted_ner"] = []
        doc["predicted_relations"] = []
        for j in range(len(doc["sentences"])):
            k = doc['doc_key'] + '-' + str(j)
            if k in ner_result:
                doc["predicted_ner"].append(ner_result[k])
            else:
                logger.info('%s not in NER results!' % k)
                doc["predicted_ner"].append([])

            doc["predicted_relations"].append([])

        js[i] = doc

    logger.info('Output predictions to %s..' % (output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(doc, cls=NpEncoder) for doc in js))








def evaluate(model, batches, tot_gold):
    """
    Evaluate the entity model
    """
    logger.info('Evaluating...')
    c_time = time.time()
    cor = 0
    tot_pred = 0
    l_cor = 0
    l_tot = 0

    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False, cur_epoch=0, tot_epoch=10)
        pred_ner = output_dict['pred_ner']
        for sample, preds in zip(batches[i], pred_ner):
            for gold, pred in zip(sample['spans_label'], preds):
                l_tot += 1
                if pred == gold:
                    l_cor += 1
                if pred != 0 and gold != 0 and pred == gold:
                    cor += 1
                if pred != 0:
                    tot_pred += 1

    acc = l_cor / l_tot
    logger.info('Accuracy: %5f' % acc)
    logger.info('Cor: %d, Pred TOT: %d, Gold TOT: %d' % (cor, tot_pred, tot_gold))
    p = cor / tot_pred if cor > 0 else 0.0
    r = cor / tot_gold if cor > 0 else 0.0
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0
    logger.info('P: %.5f, R: %.5f, F1: %.5f' % (p, r, f1))
    logger.info('Used time: %f' % (time.time() - c_time))
    return f1


def setseed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False



def dict_from_json_file(json_file: str):
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)


def main(args_file):
    model_config = dict_from_json_file(args_file)
    from argparse import Namespace
    args = Namespace(**model_config)

    # args.data_dir = args.data_dir + args.task

    args.train_data = os.path.join(args.data_dir, 'train.json')
    args.dev_data = os.path.join(args.data_dir, 'dev.json')
    args.test_data = os.path.join(args.data_dir, 'test.json')


    if 'roberta' in args.model:
        logger.info('Use Roberta: %s' % args.model)
        args.use_roberta = True

    setseed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))

    logger.info(sys.argv)
    logger.info(args)

    ner_label2id, ner_id2label = get_labelmap(task_ner_labels[args.task])
    num_ner_labels = len(task_ner_labels[args.task]) + 1
    model = EntityModel(args, num_ner_labels=num_ner_labels)

    dev_data = Dataset(args.dev_data)
    dev_samples, dev_ner = convert_dataset_to_samples(dev_data, args.max_span_length, ner_label2id=ner_label2id,
                                                      context_window=0)
    dev_batches = batchify(dev_samples, args.eval_batch_size)

    if args.do_train:
        train_data = Dataset(args.train_data)
        # train_samples, train_ner = convert_dataset_to_samples(train_data, args.max_span_length, ner_label2id=ner_label2id, context_window=args.context_window)
        train_samples, train_ner = convert_dataset_to_samples_drop_ents_neg_sample(train_data, args.max_span_length,
                                                                        drop_p=args.drop_percent,
                                                                        ner_label2id=ner_label2id,
                                                                        neg_sample = args.neg_sample)
        train_batches = batchify(train_samples, args.train_batch_size)
        best_result = 0.0

        param_optimizer = list(model.bert_model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if 'bert' in n]},
            {'params': [p for n, p in param_optimizer
                        if 'bert' not in n], 'lr': args.task_learning_rate}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=not (args.bertadam))
        t_total = len(train_batches) * args.num_epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total * args.warmup_proportion), t_total)

        tr_loss = 0
        tr_examples = 0
        global_step = 0
        eval_step = len(train_batches) // args.eval_per_epoch
        tot_step = args.num_epoch * len(train_batches)
        for _ in tqdm(range(args.num_epoch)):
            train_samples, train_ner = convert_dataset_to_samples_drop_ents_neg_sample(train_data, args.max_span_length,
                                                                                       drop_p=args.drop_percent,
                                                                                       ner_label2id=ner_label2id,
                                                                                       neg_sample=args.neg_sample)
            train_batches = batchify(train_samples, args.train_batch_size)
            for i in range(len(train_batches)):
                output_dict = model.run_batch(train_batches[i], training=True, tot_step=tot_step, cur_step=global_step,
                                              tot_epoch=args.num_epoch, cur_epoch=_)
                loss = output_dict['ner_loss']
                loss.backward()

                tr_loss += loss.item()
                tr_examples += len(train_batches[i])
                global_step += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if global_step % args.print_loss_step == 0:
                    logger.info('Epoch=%d, iter=%d, loss=%.5f' % (_, i, tr_loss / tr_examples))
                    tr_loss = 0
                    tr_examples = 0

                if global_step % eval_step == 0:
                    f1 = evaluate(model, dev_batches, dev_ner)
                    if f1 > best_result:
                        best_result = f1
                        logger.info('!!! Best valid (epoch=%d): %.2f' % (_, f1 * 100))
                        save_model(model, args)

    if args.do_eval:
        args.bert_model_dir = args.output_dir
        model = EntityModel(args, num_ner_labels=num_ner_labels)
        if args.eval_test:
            test_data = Dataset(args.test_data)
            prediction_file = os.path.join(args.output_dir, args.test_pred_filename)
        else:
            test_data = Dataset(args.dev_data)
            prediction_file = os.path.join(args.output_dir, args.dev_pred_filename)
        test_samples, test_ner = convert_dataset_to_samples(test_data, args.max_span_length, ner_label2id=ner_label2id,
                                                            context_window=0)
        test_batches = batchify(test_samples, args.eval_batch_size)
        evaluate(model, test_batches, test_ner)
        output_ner_predictions(model, test_batches, test_data, output_file=prediction_file)


if __name__ == '__main__':
    main(args_file="train.json")







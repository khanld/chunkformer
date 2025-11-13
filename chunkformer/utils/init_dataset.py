import copy
from typing import Optional

from chunkformer.dataset.dataset import Dataset
from chunkformer.text.base_tokenizer import BaseTokenizer


def init_asr_dataset(
    data_type, data_list_file, tokenizer: Optional[BaseTokenizer] = None, conf=None, partition=True
):
    return Dataset(data_type, data_list_file, tokenizer, conf, partition)


def init_dataset(
    dataset_type,
    data_type,
    data_list_file,
    tokenizer: Optional[BaseTokenizer] = None,
    conf=None,
    partition=True,
    split="train",
):
    assert dataset_type in ["asr", "ssl", "classification"]

    if split != "train":
        cv_conf = copy.deepcopy(conf)
        cv_conf["cycle"] = 1
        cv_conf["speed_perturb"] = False
        cv_conf["spec_aug"] = False
        cv_conf["spec_sub"] = False
        cv_conf["spec_trim"] = False
        cv_conf["shuffle"] = False
        cv_conf["list_shuffle"] = False
        conf = cv_conf

    # Add dataset_type to conf so Dataset function knows how to batch
    conf = copy.deepcopy(conf)
    conf["dataset_type"] = dataset_type

    if dataset_type == "asr":
        return init_asr_dataset(data_type, data_list_file, tokenizer, conf, partition)
    elif dataset_type == "classification":
        # Classification uses the same Dataset class but without tokenizer
        return init_asr_dataset(
            data_type, data_list_file, tokenizer=None, conf=conf, partition=partition
        )
    else:
        from chunkformer.ssl.init_dataset import init_dataset as init_ssl_dataset

        return init_ssl_dataset(data_type, data_list_file, conf, partition)

import functools, json

import seqio
import tensorflow as tf
import t5.data
from t5.data import postprocessors
from t5.data import preprocessors
from t5.evaluation import metrics
from seqio import FunctionDataSource, utils

import tensorflow_datasets as tfds

TaskRegistry = seqio.TaskRegistry
vocabulary = seqio.ByteVocabulary()

# The feature with streaming from HuggingFace is based on code made by Javier de la Rosa
# If you are using the code - please refer

DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(vocabulary=vocabulary, add_eos=True, required=False),
    "targets": seqio.Feature(vocabulary=vocabulary, add_eos=True),
}


def gen_dataset(split, column="text", limit_rec=None):
    recs_generated = 0
    while True and (limit_rec is None or recs_generated < limit_rec):
        with open(f"./wikitext_{split}.jsonl", "r") as dataset_fh:
            for line in dataset_fh:
                row = json.loads(line)
                recs_generated += 1
                if limit_rec is None or recs_generated < limit_rec:
                    yield row["text"]
                else:
                    break


def dataset_fn(split, shuffle_files, seed=None, dataset_params=None, limit_rec=None):
    return tf.data.Dataset.from_generator(
        functools.partial(gen_dataset, split, limit_rec=limit_rec),
        output_signature=tf.TensorSpec(shape=(), dtype=tf.string, name="wikitext"),
    )


@utils.map_over_dataset
def target_to_key(x, key_map, target_key):
    """Assign the value from the dataset to target_key in key_map"""
    return {**key_map, target_key: x}


# Final pretraining task used in Raffel et al., 2019 adaptated to NCC
TaskRegistry.add(
    "wikitext_span_corruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key,
            key_map={
                "inputs": None,
                "targets": None,
            },
            target_key="targets",
        ),
        seqio.preprocessors.tokenize,
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": DEFAULT_OUTPUT_FEATURES["targets"]},
    metric_fns=[],
)

# uniref100_task = t5.data.TaskRegistry.get("uniref504_span_corruption_500k_stream")
# ds = uniref100_task.get_dataset(split="validation", sequence_length={"inputs": 128, "targets": 32})

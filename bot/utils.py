import logging
import random

import numpy as np
import torch
import transformers
import yaml


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    class ColorCodes:
        grey = "\x1b[38;21m"
        green = "\x1b[1;32m"
        yellow = "\x1b[33;21m"
        red = "\x1b[31;21m"
        bold_red = "\x1b[31;1m"
        blue = "\x1b[1;34m"
        light_blue = "\x1b[1;36m"
        purple = "\x1b[1;35m"
        reset = "\x1b[0m"

    format = "%(message)s"

    FORMATS = {
        logging.DEBUG: ColorCodes.grey + format + ColorCodes.reset,
        logging.INFO: ColorCodes.light_blue + format + ColorCodes.reset,
        logging.WARNING: ColorCodes.yellow + format + ColorCodes.reset,
        logging.ERROR: ColorCodes.red + format + ColorCodes.reset,
        logging.CRITICAL: ColorCodes.bold_red + format + ColorCodes.reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logger(name):
    """Set up logger."""
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    return logger


# Set up logging
transformers.logging.set_verbosity_error()

logger = setup_logger(__name__)


def set_seed(seed):
    """Set seed globally."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except:
        pass


def read_yaml(path):
    with open(path, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return data


def load_pipeline(task, **kwargs):
    """Load a pipeline."""
    logger.info(f"Loading the pipeline '{kwargs.get('model')}'...")

    return transformers.pipeline(task, **kwargs)


def clean_text(txt):
    """Remove unnecessary spaces."""
    return ' '.join(txt.strip().split())


def generate_responses(prompt, pipeline, seed=None, debug=False, **kwargs):
    """Generate responses using a text generation pipeline."""
    if seed is not None:
        set_seed(seed)

    outputs = pipeline(prompt, **kwargs)
    responses = list(map(lambda x: clean_text(x['generated_text'][len(prompt):]), outputs))
    if debug:
        logger.debug(dict(responses=responses))
    return responses


def generate_scores(prompt, responses, pipeline, **kwargs):
    """Generate scores using a text classification pipeline."""
    responses = [prompt + response for response in responses]

    outputs = pipeline(responses, **kwargs)
    return [output['score'] for output in outputs]


def pick_best_response(prompt, responses, ranker_dict, debug=False):
    """Pick the best response according to the weighted average of scores."""
    if len(ranker_dict) == 0:
        return random.choice(responses)

    def _get_wa_group_scores(group_name):
        group_scores = 0
        group_weight_sum = 0
        for model_name, dct in ranker_dict.items():
            if dct['group'] == group_name:
                scores = np.array(generate_scores(
                    prompt,
                    responses,
                    dct['pipeline']
                ))
                if debug:
                    logger.debug(dict(
                        group=group_name,
                        model=model_name,
                        model_scores=scores,
                        model_weight=dct['weight']
                    ))
                group_scores += scores * dct['weight']
                group_weight_sum += dct['weight']
        group_scores /= group_weight_sum
        return group_scores

    group_names = list(map(lambda x: x['group'], ranker_dict.values()))
    if 'prior' in group_names:
        prior_scores = _get_wa_group_scores('prior')
        if debug:
            logger.debug(dict(prior_scores=prior_scores))
    else:
        prior_scores = 1
    if 'cond' in group_names:
        cond_scores = _get_wa_group_scores('cond')
        if debug:
            logger.debug(dict(cond_scores=cond_scores))
    else:
        cond_scores = 1
    final_scores = prior_scores * cond_scores
    if debug:
        logger.debug(dict(final_scores=final_scores))
    return responses[np.argmax(final_scores)]


def _load_state_dict_in_model(model, state_dict):
    load_result = model.load_state_dict(state_dict, strict=False)

    if len(load_result.missing_keys) != 0:
        if model._keys_to_ignore_on_save is not None and set(load_result.missing_keys) == set(
            model._keys_to_ignore_on_save
        ):
            model.tie_weights()
        else:
            print(f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.")
    if len(load_result.unexpected_keys) != 0:
        print(
            f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
        )


def load_weights(model, weights_path):
    state_dict = torch.load(weights_path, map_location="cpu")

    _load_state_dict_in_model(model, state_dict)

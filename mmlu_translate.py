import os
import time
import random
from os import environ
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set
from google.cloud import translate
from datasets import DatasetDict, load_dataset, Dataset
from test_translation.mmlu_translations.cloud_translate_mapping import (cloud_translate_lang_code_to_name,
                                                          cloud_translate_lang_name_to_code)

PROJECT_ID = environ.get("PROJECT_ID", "")
assert PROJECT_ID
PARENT = f"projects/{PROJECT_ID}"


## MMLU Google Translate Codes
CLOUD_TRANSLATE_LANG_CODES = ['hi','vi','ms','fil','id','te','si','ne', 'ar', 'fa', 'tr', 'ky', 'he', 'zh-CN', 'zh-TW', 'ja', 'ko',
'nl','fr', 'de', 'el', 'it','pl', 'ro', 'uk', 'ru', 'cs', 'pt', 'es', 'sv', 'lt', 'sr',
'mg', 'so', 'yo', 'ha', 'am', 'sn', 'ig', 'ny']


def cloud_translate(example: Dict[str, str],
                    target_lang_code: str,
                    keys_to_be_translated: List[str],
                    max_tries: int = 5) -> Dict[str, str]:
    """Translates the input example batch of texts using Google Cloud Translate
    API.

    Args:
        example (Dict[str, str]): A batch of inputs from the dataset for translation. Keys are the column names, values are the batch of text inputs
        target_lang_code (str): Language code for the target language
        keys_to_be_translated (List[str]): The keys/columns for the texts you want translated.cl
        max_tries (int, optional): _description_. Defaults to 5.

    Returns:
        Dict[str, str]: Translated outputs based on the example Dict
    """
    translate_client = translate.TranslationServiceClient()

    tries = 0

    while tries < max_tries:
        tries += 1
        try:
            for key in keys_to_be_translated:               
                results = translate_client.translate_text(
                            parent=PARENT,
                            contents=example[key],
                            target_language_code=target_lang_code,
                            mime_type="text/plain"
                )
                example[key] = [translation.translated_text for translation in results.translations]
                # print("results:", example[key])
                time.sleep(random.uniform(0.8, 1.5))
        except Exception as e:
            print(e)
            time.sleep(random.uniform(2, 5))

    return example


def translate_dataset_via_cloud_translate(
    dataset: DatasetDict,
    dataset_name: str,
    translator_name: str,
    splits: List[str],
    target_language: str,
    output_dir: str = "./datasets",
    source_language: str = "English",
    num_proc: int = 8,
    batch_size: int = 8,
) -> None:
    """This function takes an DatasetDict object and translates it using google translate API. 
    The function then ouputs the translations in both json and csv formats into a output directory 
    under the following naming convention:

    <output_dir>/<dataset_name>/<source_language_code>_to_<target_language_code>/<translator_name>/<date>/<split>.<file_type>

    Args:
        dataset (DatasetDict): A DatasetDict object of the original text dataset. Needs to have at least one split.
        dataset_name (str): Name of the dataset for storing output.
        translator_name (str): Name of the template for storing output.
        splits (List[str]): Split names in the dataset you want translated.
        target_language (str): the language you want translation to.
        output_dir (str, optional): Root directory of all datasets. Defaults to "./datasets".
        source_language (str, optional): Languague of the original text. Defaults to "English".
        num_proc (int, optional): Number of processes to use for processing the dataset. Defaults to 8. 
    """

    date = datetime.today().strftime('%Y-%m-%d')

    source_language_code = cloud_translate_lang_name_to_code[source_language]
    target_language_code = cloud_translate_lang_name_to_code[target_language]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for split in splits:
        split_time = time.time()
        ds = dataset[split]
        print(f"[{split}] {len(ds):}")

        ## extract options from "choices" column for mmlu
        ds = ds.map(lambda example: {"option_0": example["choices"][0],
                          "option_1": example["choices"][1],
                          "option_2": example["choices"][2],
                          "option_3": example["choices"][3]
                         })
        ## specify column names for which translation must be done.
        translate_keys = ['question', 'option_0','option_1','option_2','option_3']
        
        ds = ds.map(
            lambda x: cloud_translate(
                x,
                target_lang_code=target_language_code,
                keys_to_be_translated=translate_keys,
            ),
            batched=True,
            batch_size=batch_size,  # translate api has limit of 204800 bytes max at a time
            num_proc=num_proc,
        )
        print(f"[{split}] One example translated {ds[0]:}")
        print(f"[{split}] took {time.time() - split_time:.4f} seconds")

        translation_path = os.path.join(output_dir, dataset_name, f"{source_language_code}_to_{target_language_code}",
                                        translator_name, date)
        Path(translation_path).mkdir(exist_ok=True, parents=True)

        ds.to_csv(
            os.path.join(
                translation_path,
                f"{split}.csv",
            ),
            index=False,
        )
        ds.to_json(os.path.join(
            translation_path,
            f"{split}.jsonl"),
            force_ascii=False,
        )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.4f} seconds")

def translate_dataset_from_huggingface_hub(dataset_name: str,
                                           translator_name: str,
                                           splits: List[str],
                                           repo_id: str = "cais/mmlu",
                                           hf_data_config_name: str = "all",
                                           output_dir: str = "./datasets",
                                           source_language: str = "English",
                                           num_proc: int = 8,
                                           translation_lang_codes: List[str] = CLOUD_TRANSLATE_LANG_CODES,
                                           exclude_languages: Set[str] = {"English"},
                                           batch_size: int=8,) -> None:
                                           
    """
    Args:
        dataset_name (str): Name of the dataset for storing output.
        splits (List[str]): Split names in the dataset you want translated.
        repo_id (str, optional): Name of the dataset repo on Huggingface. Defaults to "bigscience/xP3".
        output_dir (str, optional): Root directory of all datasets. Defaults to "./datasets".
        source_language (str, optional): Languague of the original text. Defaults to "English".
        num_proc (int, optional): Number of processes to use for processing the dataset. Defaults to 8.
        translation_lang_codes (List[str], optional): List of Flores-200 language codes to translate to. Defaults to T5_LANG_CODES.
        exclude_languages (Set[str], optional): Set of languages to exclude. Defaults to {"English"}.
    """

    # temp_root = "temp_datasets"
    # temp_dir = f"{temp_root}/{dataset_name}"
    # Path(temp_dir).mkdir(exist_ok=True, parents=True)
    

    dataset = load_dataset(path=repo_id, name=hf_data_config_name) 

    # Make a copy of the source dataset inside translated datasets as well
    date = datetime.today().strftime('%Y-%m-%d')
    source_language_code = cloud_translate_lang_name_to_code[source_language]
    translation_path = os.path.join(output_dir, dataset_name, f"{source_language_code}_to_{source_language_code}",
                                    translator_name, date)
    Path(translation_path).mkdir(exist_ok=True, parents=True)
    for split in splits:
        ds = dataset[split]
        ds = ds.map(lambda example: {"option_0": example["choices"][0],
                          "option_1": example["choices"][1],
                          "option_2": example["choices"][2],
                          "option_3": example["choices"][3]
                         })
        ds.to_csv(
            os.path.join(
                translation_path,
                f"{split}.csv",
            ),
            index=False,
        )
        ds.to_json(os.path.join(
            translation_path,
            f"{split}.jsonl",
        ),
        force_ascii=False,
        )

    ## Start translation for all specified languages
    translation_lang_codes = CLOUD_TRANSLATE_LANG_CODES
    for code in translation_lang_codes:
        l = cloud_translate_lang_code_to_name[code]
        if l not in exclude_languages:
            print(f"Currently translating: {l}")
            translate_dataset_via_cloud_translate(
                dataset=dataset,
                dataset_name=dataset_name,
                translator_name=translator_name,
                splits=splits,
                target_language=l,
                output_dir=output_dir,
                source_language=source_language,
                num_proc=num_proc,
                batch_size=batch_size,
            )
            print("finished translation for :", l)
        break


if __name__=="__main__":
    translate_dataset_from_huggingface_hub(
        repo_id = "cais/mmlu", #"CohereForAI/mmlu_filtered_translation",
        hf_data_config_name="astronomy",
        dataset_name="mmlu_filtered_translation",
        translator_name="google_cloud_translate",
        splits=["validation"],
        output_dir= "/home/shivalikasingh/test_translation/datasets",
        source_language= "English",
        num_proc=8,
        batch_size=1,
    )
# mmlu_translations

Based on [this repo](https://github.com/for-ai/instruct-multilingual/blob/main/instructmultilingual). 

## Initial Setup

- Login with your account: gcloud auth application-default login
- gcloud config set project <GCP_PROJECT_ID>
- gcloud auth application-default set-quota-project <GCP_PROJECT_ID>  
- gcloud services enable translate.googleapis.com
- export PROJECT_ID=$(gcloud config get-value core/project)
- pip install -r requirements.txt

## Start Translations:
After doing above setup, run "mmlu_translate.py" to start translations

Things you may wanna modify before running the translation script:
- CLOUD_TRANSLATE_LANG_CODES (line 19)
- translate_keys (line 115)
- params for translate_dataset_from_huggingface_hub() called in main()
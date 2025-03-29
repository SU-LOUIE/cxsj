from huggingface_hub import create_repo
HUGGINGFACE_TOKEN = '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'
create_repo("klbj/medical-model2", token=HUGGINGFACE_TOKEN, exist_ok=True)
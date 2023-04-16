# AK_CLEANROOM

  AK_CLEANROOM is a script that trains a language model using the Hugging Face Transformers library. The script takes care of loading datasets, preprocessing them, training the model, and saving the trained model.  ## How to use  1. Make sure you have Python 3.6 or later installed.  2. Install the required dependencies using the `requirements.txt` file:

```bash pip install -r requirements.txt``

1.  Configure the `config/config.json` file with the required settings, such as:

*   `data`: Path to the training dataset file in JSON format.
*   `eval_data`: Path to the evaluation dataset file in JSON format.
*   `model`: Pretrained model to use for training.
*   `model_tokenizer`: Tokenizer associated with the pretrained model.
*   `MICRO_BATCH_SIZE`: Micro batch size for training.
*   `GRADIENT_ACCUMULATION_STEPS`: Number of gradient accumulation steps.
*   `EPOCHS`: Number of training epochs.
*   `LEARNING_RATE`: Learning rate for training.
*   `CUTOFF_LEN`: Maximum length of text sequences during training.
*   `MAX_STEP`: Maximum number of training steps.

4.  Set the `HUGGINGFACE_API_KEY` environment variable if you want to push the trained model to Hugging Face Hub:

bash

`export HUGGINGFACE_API_KEY=your_huggingface_api_key`

5.  Run the script:

bash

`python3 AK_CLEANROOM_ML.py`

During execution, the script will:

*   Load the datasets and perform preprocessing if necessary.
*   Load the pretrained model and tokenizer.
*   Train the model using the specified parameters.
*   Save the trained model and tokenizer.
*   Push the trained model to Hugging Face Hub (if `huggingface_access_token` is provided in the `config.json` file).

License
-------

This project is licensed under the MIT License. See the `LICENSE` file for details.

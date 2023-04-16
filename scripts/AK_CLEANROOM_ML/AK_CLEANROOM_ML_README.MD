ELIAI Training Script
=====================

This script is designed to train an AI model using the Hugging Face Transformers library. The main objective of this script is to train a model on a custom dataset, preprocess the dataset if needed, and save the model after training. Optionally, the trained model can be pushed to the Hugging Face Model Hub.

Prerequisites
-------------

*   Python 3.6 or higher
*   Install required libraries:

bash

`pip install -r requirements.txt`

Usage
-----

1.  Prepare your dataset in JSON format with the following structure:

json

`[   {     "instruction": "instruction_text",     "input": "input_text",     "output": "output_text"   },   ... ]`

2.  Edit the `config/config.json` file to specify the necessary parameters, such as model, tokenizer, dataset paths, training parameters, and Hugging Face credentials.

3.  Run the script:


bash

`python train_script.py`

4.  Follow the prompts in the terminal to continue with the training or fix any issues in the dataset.

5.  After the training is completed, the model and tokenizer will be saved to the specified output directory.

6.  If the Hugging Face credentials are provided, the model will be pushed to the Hugging Face Model Hub.


Config File Format
------------------

The `config/config.json` file should contain the following parameters:

json

`[   {     "data": "path/to/your/dataset.json",     "eval_data": "path/to/your/eval_dataset.json",     "model": "huggingface/model_name",     "model_tokenizer": "huggingface/tokenizer_name",     "MICRO_BATCH_SIZE": 1,     "GRADIENT_ACCUMULATION_STEPS": 1,     "EPOCHS": 1,     "LEARNING_RATE": 5e-5,     "CUTOFF_LEN": 256,     "MAX_STEP": 10,     "PreProcessedData?": false,     "Load_Checkpoint": "",     "out_dir": "path/to/save/trained_model",     "load_best_model_at_end": true,     "save_steps": 1,     "eval_steps": 1,     "CPU_MODE": false,     "huggingface_access_token": "your_huggingface_access_token"   } ]`

Contributing
------------

Feel free to open issues or submit pull requests if you have any suggestions or improvements for this script. Your contributions are always welcome.

License
-------

This project is licensed under the MIT License.

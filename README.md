# SciBERT for Uncertainty Prediction

The repository contains the codes to fine-tune SciBERT for uncertainty prediction.

## How to Use

Our trained model can predict the *number of uncertainty expressions* given *a sentence* and *some meta-information about the economic journal, including gender of the authors.*

### Step 1) Download the trained model

Our trained model can be downloaded from this [google drive link](https://drive.google.com/file/d/1xJ_5myOvjGKGSH3hGRLd9hu5s50ydhTO/view?usp=share_link).

To download it to your server using command line, you can use the following command:

```bash
cd models/
file_id="1xJ_5myOvjGKGSH3hGRLd9hu5s50ydhTO"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${file_id}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${file_id}" -O trained_scibert_uncertainty.pt && rm -rf /tmp/cookies.txt
```

### Step 2) Run the inference mode of the trained model

We added 10 example data points in [`data/example_data.csv`](data/example_data.csv). 

You can run our trained model on this example data using the following command:

```bash
cd ..
python eval.py --splits_filename data/example_data.csv
```

## Training Procedure

### Model Architecture

Our model follows the following pipeline: [Use the image from overleaf]

### Training

To train a model for classification/regression tasks, use [`train.py`](train.py):

```bash
python train.py --dataset_df_dir data/ --splits_filename train.csv val.csv test.csv \
    --text_col input --y_col label --class_weight automatic \
    --model_save_dir models/ \
    --log_dir log/ --iter_time_span 1000 \
    --pretrained_model roberta-large --lr 1e-5 --max_length 512 --csv_output_path output/roberta_large_output.csv \
    --n_epochs 5
```

Note: To obtain the entire training data, please contact the correspondence author of ["Editing a Woman’s Voice" (2021)](https://arxiv.org/abs/2212.02581).

### Evaluation 

To evaluate the trained model on new data, use [`eval.py`](eval.py):

```bash
python eval.py --splits_filename test.csv --text_col input --y_col label --num_numeric_features 0 --numeric_features_col \
        --model_load_path models/roberta_large.pt \
        --log_dir log/ \
        --csv_output_path output/roberta_large_output.csv \
        --output_type binary --max_length 512 --pretrained_model roberta-large \
        --batch_size 8 --dropout_rate 0.1
```




# Training and Evaluation Scheme: AI Scholar Uncertainty Prediction.

The repository contains the code of fine-tuning pretrained language models for uncertainty prediction.

To train a model for classification/regression tasks, use `train.py`:

```bash
python train.py --dataset_df_dir data/ --splits_filename train.csv val.csv test.csv \
    --text_col input --y_col label --class_weight automatic --seed 42 \
    --model_save_dir models/ \
    --log_dir log/ --iter_time_span 1000 \
    --pretrained_model roberta-large --lr 1e-5 --max_length 512 --csv_output_path output/roberta_large_output.csv \
    --n_epochs 5

```

To evaluate the trained model on new data, use `eval.py`:

```bash
python eval.py --dataset_df_dir data/ \
        --splits_filename test.csv test.csv test.csv --text_col input --y_col label --num_numeric_features 0 \
        --model_load_path models/roberta_large.pt \
        --log_dir log/ \
        --csv_output_path output/roberta_large_output.csv \
        --output_type binary --max_length 512 --pretrained_model roberta-large \
        --batch_size 8 --dropout_rate 0.1 --img_output_dir img/ 
```
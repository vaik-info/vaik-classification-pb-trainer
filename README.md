# vaik-classification-pb-trainer

Train classification pb model

## train_pb.py

### Usage

```shell
pip install -r requirements.txt
python train_pb.py --train_input_dir_path ~/.vaik-mnist-classification-dataset/train \
                --valid_input_dir_path ~/.vaik-mnist-classification-dataset/valid \
                --classes_txt_path ~/.vaik-mnist-classification-dataset/classes.txt \
                --model_type efficient_net_v2_b0_model \
                --epochs 10 \
                --step_size 100 \
                --batch_size 8 \
                --test_max_sample_per_classes 100 \
                --image_height 224 \
                --image_width 224 \
                --output_dir_path '~/output_model'        
```

### Output

![vaik-classification-pb-trainer-output-train1](https://user-images.githubusercontent.com/116471878/200271108-3b485be9-be4d-48f3-b185-855be8651cf6.png)

![vaik-classification-pb-trainer-output-train2](https://user-images.githubusercontent.com/116471878/200271111-f21fc130-02f1-4d6d-b609-26884ebb9c59.png)
 
-----

## dump_dataset.py

### Usage

```shell
pip install -r requirements.txt
python dump_dataset.py --input_image_dir_path ~/.vaik-mnist-classification-dataset/train \
                --classes_txt_path ~/.vaik-mnist-classification-dataset/classes.txt \
                --sample_num 25000 \
                --image_height 224 \
                --image_width 224 \
                --output_dir_path ~/.vaik-mnist-classification-dataset/dump
```

### Output

![vaik-classification-pb-trainer-output-dump](https://user-images.githubusercontent.com/116471878/200271097-7a024ef7-d4a9-4d95-9809-ea4607a2c2dd.png)
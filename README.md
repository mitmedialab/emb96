```bash
sudo docker build -t medialab/emb96:latest .

sudo docker run \
  -v /dataset:/app/emb96/dataset \
  -v /t_dataset:/app/emb96/t_dataset \
  -v /generated:/app/emb96/generated \
  -v /experience:/app/emb96/experience \
  -it \
  -p 0.0.0.0:6006:6006 \
  medialab/emb96:latest \

python3 main.py \
  --dataset_dir ../dataset/ \
  --t_dataset_dir ../t_dataset/ \
  --generated_dir ../generated/ \
  --epochs 250 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --weight_decay 0. \
  --beta 4. \
  --num_workers 4 \
  --experience_name experience \
  --saving_rate 1 \
  --train \
```

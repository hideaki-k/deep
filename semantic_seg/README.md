# フロー

1. MATLAB DEM\iterator.mでランダムDEMを生成
2. semantic_img_loc.csv senabtic_eval_loc.csvに変更を書き込む
3. python hazard_detect.pyにてhazard_labelを作成
4. DEM_autoencoder_segmentation.pyにて対象CSVを指定して実行

# dem_autoencoder_segmentation.py
64pixの領域から長方形を塗りつぶすタスク

Run ```python train.py -b 128 -e 100 -t 20 -r True```
(batch, epoch, simulation time steps, recurrent conection)

教師データ
(C:\Users\aki\Documents\GitHub\deep\semantic_seg\semantic_data)

各エポックごとに/modelsに.pthとしてsaveされる。  

教師データはrectangle_builder.pyで生成される.


# dem_conv_classification.py

64pixの二値分類タスク

教師データ  
(
    ラベル1
    C:\Users\aki\Documents\GitHub\deep\DEM\crater  
    ラベル2
    C:\Users\aki\Documents\GitHub\deep\DEM\terrain_generation\perlin_bolder
)

![image](https://user-images.githubusercontent.com/56909755/118626459-dc948400-b805-11eb-8c57-f77c62fbeb1e.png)

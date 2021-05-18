# dem_autoencoder_segmentation.py
64pixの領域から長方形を塗りつぶすタスク

教師データ
(C:\Users\aki\Documents\GitHub\deep\semantic_seg\semantic_data)

各エポックごとに/modelsに.pthとしてsaveされる。  

教師データはrectangle_builder.pyで生成される・
# dem_conv_classification.py

64pixの二値分類タスク

教師データ  
(
    ラベル1
    C:\Users\aki\Documents\GitHub\deep\DEM\crater  
    ラベル2
    C:\Users\aki\Documents\GitHub\deep\DEM\terrain_generation\perlin_bolder
)


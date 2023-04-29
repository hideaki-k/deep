# About

仮想の月面DEM(Digital elevation model)を生成するツールです。\
パーリングノイズによるフラクタル地形をベースとし、必要に応じてクレータやボルダー、傾斜等を付与することができます。



# Udage

## Run, `iterator.m`  

普通のDEMを生成したい場合は、iterator.m内で`fractal_terrain_generation`を呼びましょう　　  

## モード  
・is_evaluate : 評価用DEM生成モード  (evaluate_terrain_generation)  
・is_double_terrain : 二つの傾斜角をもったDEM生成モード   (double_terrain_generation)  
・`指定なし : 通常のdem生成モード(fractal_terrain_generation) `

## fractal_terrain_generation(k,mode,pix,angle,folder_name,is_noise,is_boulder)  

・k : サンプルインデックス  
・mode : 保存モード(0:mat,png保存,1:avi,2:三次元プロット)  
・pix : データサイズ  
・angle : base DEMの傾斜角  
・folder_name : 　保存先フォルダ  
・is_noise :　パーリンノイズ有無  
・is_boulder : ボルダー有無  

### その他　アングルを複数種類作りたいとき
iterator.mでis_mix_angle=1とし
max_angle（最大の傾斜角)を指定してください



import numpy as np

def rectangle(img, img_ano, centers, max_side):
    """
    img 四角形の線のみの二次元画像
    img_ano アノテーション画像
    centers center座標のlist
    max_side 辺の最大長さの1/2
    """
    if max_side < 3: # max_side が小さすぎるとき
        max_side = 4
    # 辺の長さの1/2を定義
    side_x = np.random.randint(3, int(max_side))
    side_y = np.random.randint(3, int(max_side))

    # 中心の座標(x, y)を定義
    x = np.random.randint(max_side + 1, img.shape[0] - (max_side + 1))
    y = np.random.randint(max_side + 1, img.shape[1] - (max_side + 1))

    #過去の中心位置と近い位置が含まれた場合,inputデータをそのまま返す
    for center in centers:
        if np.abs(center[0] - x ) < (2 * max_side + 1):
            if np.abs(center[1] - y) < (2 * max_side + 1):
                return img, img_ano, centers
    
    img[x - side_x : x + side_x, y - side_y] = 1.0      #上辺
    img[x - side_x : x + side_x, y + side_y] = 1.0      #下辺
    img[x - side_x, y - side_y : y + side_y] = 1.0      #左辺
    img[x + side_x, y - side_y : y + side_y + 1] = 1.0  #右辺
    img_ano[x - side_x : x + side_x + 1, y - side_y : y + side_y + 1] = 1.0
    centers.append([x, y])
    return img, img_ano, centers

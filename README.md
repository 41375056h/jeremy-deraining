# Meta-Auxiliary Deraining：專案設計說明

這份 README 只說明研究目標、資料、方法與實作規格，不記錄進度或實驗結果。

本專案把 Chi 等人（CVPR 2021）用於動態場景去模糊的 meta-auxiliary learning，移植到單張影像雨絲去除。原論文處理的是模糊，不是雨；本文也不宣稱精確重現作者未公開的去雨實作。每個決定都應區分「論文明示」、「依公式解讀」或「暫定實作」。

規格優先順序：[TODO.md](./TODO.md) > [架構圖 v2](./Meta_Auxiliary_Deraining_Architecture_v2.png) > [論文](./Chi_Test-Time_Fast_Adaptation_for_Dynamic_Scene_Deblurring_via_Meta-Auxiliary_Learning_CVPR_2021_paper.pdf)。使用者後續的明確修正優先於較早的候選設計。若有衝突，先列出並確認，不自行取捨。架構圖的舊 H6 已過時，依 TODO 分成 H6a（meta-gradient）、H6b（內迴圈 optimizer 與學習率）、H6c（外迴圈 optimizer 與學習率）。

## 核心概念

同一張帶雨圖有兩個任務。主任務預測去雨殘差 R，計算「去雨圖 = 帶雨圖 + R」。輔助任務利用共享特徵 F1 與未截斷梯度的 R，重建原帶雨圖。輔助分支的直接入口是 concat(F1, R)，不是原帶雨圖或去雨圖；原帶雨圖只作重建目標。

主、輔任務不能只是各自訓練。輔助損失的梯度必須能經由 R 到達主分支，也要到達共享主幹。測試時沒有乾淨 GT，仍可用當張帶雨圖計算輔助重建損失，做少量更新。Meta-training 的目的，是學到讓「輔助更新後的去雨」也變好的起始權重。重建損失下降不等於去雨改善，必須另外檢查去雨損失與影像品質。

## 模型怎麼實作

輸入為 RGB，像素縮放到 [-1, 1]。從原圖各自直接產生四分之一、二分之一、原尺寸三個尺度，由粗到細處理；乾淨 GT 也使用同一金字塔函式。原圖高寬 H、W 的半尺度取 floor(H/2)、floor(W/2)，四分之一尺度取 floor(H/4)、floor(W/4)。暫定用抗鋸齒雙線性插值及 align_corners=False；這是去雨實作假設，不是論文完整指定的程序。

三尺度各有獨立參數。對任一尺度高寬 h、w，定義 u_h=ceil(h/2)、u_w=ceil(w/2)；v_h=ceil(u_h/2)、v_w=ceil(u_w/2)；t_h=ceil(v_h/2)、t_w=ceil(v_w/2)。預設主幹：

| 層或特徵 | 輸出尺寸 | 說明 |
| --- | --- | --- |
| Encoder Conv 1 | h x w x 32 | stride 1；較細尺度在這層後接收 F1 |
| Encoder Conv 2 | u_h x u_w x 64 | stride 2；在這層後接收 F2 |
| Encoder Conv 3 | v_h x v_w x 80 | stride 2；在這層後接收 F3 |
| Encoder Conv 4 與瓶頸 ResBlocks | t_h x t_w x 80 | Conv 4 預設 stride 2 |
| Decoder Deconv 3、skip 融合與 ResBlocks | v_h x v_w x 80 | 輸出 F3，最深 |
| Decoder Deconv 2、skip 融合與 ResBlocks | u_h x u_w x 64 | 輸出 F2，中間 |
| Decoder Deconv 1、skip 融合與 ResBlocks | h x w x 32 | 輸出 F1，最淺 |

F3、F2、F1 都取自對應 decoder ResBlocks 之後，絕不是「F1 最深」。粗尺度傳給中尺度，中尺度再傳給細尺度。傳送前先以雙線性插值對齊接收端的實際高寬，再 concat：

| 跨尺度特徵 | 下一尺度的接收位置 | concat 後通道 | 融合後通道 |
| --- | --- | --- | --- |
| F3，80 通道 | Encoder Conv 3 後 | 160 | 80 |
| F2，64 通道 | Encoder Conv 2 後 | 128 | 64 |
| F1，32 通道 | Encoder Conv 1 後 | 64 | 32 |

主分支從 F1 預測三通道 R，輸出去雨圖。輔助分支的直接入口恰為 concat(F1, R)，共 35 通道，輸出三通道的重建帶雨圖。不得把原帶雨圖、去雨圖或乾淨 GT 直接接到輔助入口；正常模式也不得對 R 做 detach。「以零張量遮斷 R」只供診斷，不是正常前向。

全部參數必須分成互不重疊、合起來涵蓋所有參數的三組：

| 群組 | 模組 | 用途 |
| --- | --- | --- |
| theta_S | 三尺度 encoder、decoder、跨尺度融合、skip 融合、主幹 ResBlocks | 兩任務共用特徵 |
| theta_Pri | 三尺度主分支投影、ResBlocks、殘差輸出層 | 去雨；透過 R 接收輔助梯度 |
| theta_Aux | 三尺度輔助融合、ResBlocks、重建輸出層 | 重建帶雨圖 |

以下都是暫定、可消融的實作細節，不能稱為論文作者的精確層序：3 x 3 有 bias 卷積、padding 1；encoder stride 為 1/2/2/2；decoder 用轉置卷積；奇數尺寸只裁掉 decoder 多出的底邊或右邊，最多一格；skip concat 後以卷積恢復通道。瓶頸與 decoder 的 F3/F2/F1 各放 4 個 ResBlock；主、輔分支各用寬度 32 和 4 個 ResBlock。預設 ResBlock 是「輸入 + 第二層卷積(LeakyReLU(第一層卷積(輸入)))」，相加後不活化；LeakyReLU 斜率 0.1。沒有 normalization 或輸出活化。ResBlock 位置、分支尾端層序與寬度、Conv 4 stride 等要記在可序列化設定中，載入 checkpoint 時核對。

## Dataset 與資料規則

資料根目錄由 [data_config.json](../deraining/data_config.json) 指定，預設 /home/intern/jeremy/dataset，不能寫死在程式中。以下路徑相對於資料根目錄：

| 獨立模型 | 帶雨圖資料夾 | 乾淨 GT 資料夾 |
| --- | --- | --- |
| rain200h | Rain200/Rain200H/train/input | Rain200/Rain200H/train/target |
| rain200l | Rain200/Rain200L/train/input | Rain200/Rain200L/train/target |
| hq_rain | HQ_RAIN/Training/input | HQ_RAIN/Training/target |

三個來源各自從新模型開始訓練，不混合資料、不互用 checkpoint，也不共用 optimizer 或 scheduler 狀態。三個來源採同檔名配對；缺少唯一對應時報錯，不用排序猜測。帶雨圖與 GT 必須尺寸相同。每個來源固定種子，依乾淨圖內容雜湊分組，留約 10% 組別驗證；同一乾淨圖的不同帶雨版本不能跨越訓練與驗證。清單需保存來源、種子、配對與分組規則、抽樣規則和逐組路徑。

資料工具只報告訓練來源與測試集之間的內容雜湊重複，不自動刪除；HQ_RAIN 固定抽 20 組帶雨與乾淨拼圖供人工確認。訓練時，對同一對 RGB 影像同步做 256 x 256 隨機裁切和水平、垂直翻轉，然後分別建立三尺度金字塔。原圖太小時，對帶雨圖與 GT 做一致的反射填充；無法反射填充時明確報錯。驗證使用完整影像、batch size 1、不做隨機增強。資料順序及 Python、NumPy、PyTorch、DataLoader worker 的種子都要可設定、可重現。

共同測試範圍：Rain100H、Rain100L、HQ_RAIN Testing、LHP_rain、RealRain_1K H/L、RainDS_real rainstreak、Unirain 03_N_RS 與 01_D_RS。Rain100H/L 和 HQ_RAIN Testing 的候選路徑列在資料設定；其餘測試來源須先確認真實資料夾與檔名配對規則，不能假設。AGAN、RaindropClarity、UAV-Rain1k 等雨滴資料集不納入雨絲基準。

## 方法論

### Stage 1：聯合預訓練

每個尺度各自對所有像素、通道、batch 元素取平均絕對誤差（L1），再將三尺度等權相加：

    L_pri = 三尺度 mean(abs(去雨輸出 - 乾淨 GT)) 的總和
    L_aux = 三尺度 mean(abs(重建帶雨圖 - 帶雨輸入)) 的總和
    L_total = primary_weight * L_pri + auxiliary_weight * L_aux

兩個權重預設皆為 1。L_aux 的目標只能是帶雨輸入，乾淨 GT 不得進入 L_aux。此階段更新 theta_S、theta_Pri、theta_Aux 全部參數。基準使用 Adam、初始學習率 1e-4；驗證 L_pri 停滯時用 ReduceLROnPlateau 將學習率減半。訓練長度以迭代數指定，不固定以 epoch 計；batch size、精度、驗證與 checkpoint 間隔、梯度裁剪均由設定控制。三個模型各自保存設定、曲線、日誌、驗證拼圖、最新及最佳 checkpoint。非有限 loss 應保存診斷狀態並停止，不能覆蓋最佳 checkpoint。

### Stage 2：Meta-auxiliary training

從同一來源的 Stage 1 權重初始化，每組帶雨／乾淨影像視為一個 task：

1. 內迴圈只看該張帶雨圖，用 L_aux 對 theta_S、theta_Pri、theta_Aux 全部參數做 5 步可微分更新，得到該 task 的暫時參數。
2. 用適應後的模型與該 task 的乾淨 GT 算 L_pri；跨 task 的外迴圈 L_pri 依論文公式加總，不默默改成平均。
3. 外迴圈只更新基礎 theta_S 與 theta_Pri。基礎 theta_Aux 另依 Algorithm 1 逐 task 依序更新，不以「平均暫時的 theta_Aux」代替。

額外的基礎 theta_Aux 更新採哪一步梯度，以及與五步內更新、外更新的精確先後，是需明確記錄的公式解讀／暫定實作。基準先檢查完整高階 meta-gradient、內外迴圈 SGD、內外學習率各 2.5e-5、內迴圈 5 步；這些不是去雨任務保證最佳的數值。若改用一階近似、Adam 或其他學習率，須另標成變體，且測試適應的內迴圈設定要與對應 meta-training 匹配。完整二階更新與全解析度五步適應都要實測顯存與時間，不能默默改成 patch、切塊或近似運算。

### 測試時適應與比較

每張測試圖都從相同 checkpoint 重設模型與 optimizer 狀態，只用這張帶雨圖的 L_aux 更新 0 到 5 步，再由主分支輸出去雨圖。測試 GT 只用於輸出後的 PSNR、SSIM 與視覺化，絕對不能用於適應、checkpoint 選擇或步數調整。共同測試集應比較 Stage 1 的 0/5 步和 Stage 2 的 0/5 步，並保存中間 1 到 4 步的診斷。RGB 指標範圍、裁邊規則、像素轉換與視覺化抽樣，都必須在比較前固定。

## 不能變的核心概念

1. 測試時只能用當張雨圖適應；GT 不能洩漏到輔助損失或適應梯度。
2. 共享主幹、主分支殘差 R 與輔助分支必須相連；正常模式下 L_aux 的梯度須到達 theta_S 和 theta_Pri。
3. Meta 目標是輔助更新後的去雨表現；不能以 L_aux 下降直接宣稱去雨改善。
4. Rain200H、Rain200L、HQ_RAIN 分別訓練成三個模型；相同乾淨內容不能跨越訓練與驗證。
5. 每張測試圖都重設至相同 checkpoint；不能把上一張適應後的狀態帶到下一張。
6. 三尺度、雙分支、R 傳到輔助分支、三組參數分工、兩階段目標與 rainy-only 測試適應，是本研究基準的身份。若更改，須另命名為變體／消融，不可仍稱同一基準。

卷積細節、ResBlock 數與位置、分支寬度、學習率、optimizer、batch size、precision 和 patch 協定可以研究，但每次都須記錄設定並分開比較。「全圖適應」與「patch 適應後在全圖評估」是不同協定。

## 程式分工與必要檢查

| 範圍 | 檔案與必要檢查 |
| --- | --- |
| 資料盤點、配對、切分、清單 | [data_tools.py](../deraining/data_tools.py)；明確配對、分組無洩漏、固定種子可重現 |
| 模型、跨尺度特徵、參數分組 | [model.py](../deraining/model.py)；尺寸、F3/F2/F1 方向、35 通道入口、參數群完整互斥 |
| 成對影像與同步增強 | [stage1_data.py](../deraining/stage1_data.py)；尺寸一致、裁切翻轉對齊、恢復後順序一致 |
| 去雨與輔助損失 | [stage1_loss.py](../deraining/stage1_loss.py)；手算小張量驗證尺度平均與加總、GT 不進 L_aux |
| 三個獨立 Stage 1 入口 | [train_stage1.py](../deraining/train_stage1.py) 與 [設定目錄](../deraining/configs/)；來源、輸出目錄、checkpoint、非有限 loss |
| Meta 更新 | [meta_auxiliary.py](../deraining/meta_auxiliary.py)；內外更新範圍、task 順序、高階梯度 |
| 測試適應 | 每張圖重設；GT 隔離、0 到 5 步輸出、指標與顯存時間 |

另外應測試：預設 ResBlock 在第二層卷積歸零時等於輸入；主損失不更新輔助參數；輔助損失正常會更新主分支，但遮斷 R 後不會；批次與單張前向一致；奇數尺寸正確；訓練中斷後的狀態與資料順序可按定義恢復。所有數值比較都須附設定、資料清單、checkpoint 和種子。

論文來源：[Chi et al., CVPR 2021 官方頁面](https://openaccess.thecvf.com/content/CVPR2021/html/Chi_Test-Time_Fast_Adaptation_for_Dynamic_Scene_Deblurring_via_Meta-Auxiliary_Learning_CVPR_2021_paper.html)。本專案的去雨實作邊界以 [TODO.md](./TODO.md) 為準。

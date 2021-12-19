import numpy as np
import warnings
warnings.filterwarnings("ignore", category=Warning)

learningRate = 0.01										# 學習率
tau = 0.1											# 容錯率
epoch = 30											# 最大世代數
num_InputX = 784										# 輸入層包含784個節點
num_hidden_layer = 1										# 隱藏層的層數
num_hidden_neuron = 30										# 隱藏層的神經元個數
num_OutputY = 4											# 輸出層包含四個神經元, 辨識 0, 3, 8, 9 四個類別
num_train_img = 12800										# 要train的img的資料筆數
num_validate = 3200										# 用來驗證的img的資料筆數
num_test = 4000											# 無類別測試資料
output_model = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) 			# One-Hot Encoding
fault_tolerance = np.array([[tau], [tau], [tau], [tau]]) 						# 容錯率
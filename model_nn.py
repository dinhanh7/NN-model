import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
import matplotlib as plt

#đọc data
df = pd.read_csv("dataset.csv")
X = df.drop(columns=["Outcome"])  #X là ma trận, ko lấy cột outcome
y = df["Outcome"]  #Y là cột kết quả (outcome)

#chuẩn hóa data
scaler = StandardScaler() #so sánh các giá trị khác nhau và đơn vị đo lường khác nhau
X = scaler.fit_transform(X) #tạo tham số mô hình học tập từ data theo y/c của mô hình

#chia data , data huấn luyện là 80%, data kiểm tra là 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#random state là chia dữ liệu, mỗi lần train là 42 mẫu

# xây dựng mô hình
def build_model(hp):
    model = Sequential()
    #input layer
    model.add(Dense(
        #units_input là số neuron của tầng đầu vào
        units=hp.Int('units_input', min_value=8, max_value=128, step=8),
        #sử dụng hàm kích hoạt relu
        activation='relu',
        input_dim=X_train.shape[1]
    ))
    #hidden layer
    for i in range(hp.Int('num_layers', 1, 3)):  #đây là số tầng ẩn
        model.add(Dense(
            units=hp.Int(f'units_{i}', min_value=8, max_value=128, step=8),#units{i} là số neuron trong mỗi tầng ẩn
            activation='relu'
        ))
    #ouput layer
    model.add(Dense(1, activation='sigmoid'))  #một neuron với hàm kích hoạt sigmoid

    # Compile
    model.compile(
        #adam là bộ tối ưu hóa
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy', #hàm mất mát loss function
        metrics=['accuracy']#theo dõi độ chính xác trong quá trình huấn luyện
    )
    return model

# khỏi tạo keras-tunner
tuner = kt.Hyperband( #Hyperband là pp tìm kiếm siêu tham số nhanh chóng bằng cách giảm dần số lượng thử nghiệm qua các bước

    build_model,
    objective='val_accuracy', #tối ưu hóa độ chính xác trên tập validation
    max_epochs=20, #mỗi epọch là một lần duyệt qua hết data trong tập huấn luyện, đây giới hạn là 20
    factor=3,
    directory='my_dir', #lưu data cho tập train
    project_name='diabetes_tuning'
)

# hàm tìm kiếm các siêu tham số tốt nhất sau khi train
tuner.search(X_train, y_train, epochs=20, validation_split=0.2, verbose=1)

# lấy mô hình tối ưu
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
Số tầng ẩn: {best_hps.get('num_layers')}
Số neuron mỗi tầng: {[best_hps.get(f'units_{i}') for i in range(best_hps.get('num_layers'))]}
Learning rate: {best_hps.get('learning_rate')}
""")

# Huấn luyện mô hình với siêu tham số tối ưu
model = tuner.hypermodel.build(best_hps) #xd mô hình với siêu tham số tối ưu
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)

#đánh giá mô hình
test_loss, test_acc = model.evaluate(X_test, y_test) #evaluate là hàm đánh giá mô hình trên tập kiểm tra
print(f"Độ chính xác trên tập kiểm tra: {test_acc:.4f}")

from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model.stochastic_gradient import SGDRegressor  # 随机梯度下降
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 生成数据
def generate_data(num_samples):
    X = np.array(range(num_samples))
    y = 3.65*X + 10
    return X, y

if __name__ == '__main__':

    # 参数
    args = Namespace( # Simple object for storing attributes.
        seed=1234,
        data_file="sample_data.csv",
        num_samples=100,
        train_size=0.75,
        test_size=0.25,
        num_epochs=100,
    )

    print(args)

    # 设置随机种子来保证实验结果的可重复性。
    np.random.seed(args.seed)

    # 生成随机数据
    X, y = generate_data(args.num_samples)
    # print(X,y)
    data = np.vstack([X, y]).T  # 通过垂直的形式把数组堆叠起来
    df = pd.DataFrame(data, columns=['X', 'y'])
    print(df.head())

    # 画散点图
    plt.title("Generated data")
    plt.scatter(x=df["X"], y=df["y"])
    plt.show()

    # 划分数据到训练集和测试集
    # print(df["X"].values)
    # print(df["X"].values.reshape(-1, 1)) # 行不作限制，对列进行限制
    X_train, X_test, y_train, y_test = train_test_split(
        df["X"].values.reshape(-1, 1), df["y"], test_size=args.test_size,
        random_state=args.seed)
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    # 标准化训练集数据 (mean=0, std=1)
    X_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.values.reshape(-1, 1))

    # 在训练集和测试集上进行标准化操作
    standardized_X_train = X_scaler.transform(X_train)
    standardized_y_train = y_scaler.transform(y_train.values.reshape(-1, 1)).ravel()
    standardized_X_test = X_scaler.transform(X_test)
    standardized_y_test = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

    # 检查
    print("mean:", np.mean(standardized_X_train, axis=0),
          np.mean(standardized_y_train, axis=0))  # mean 应该是 ~0
    print("std:", np.std(standardized_X_train, axis=0),
          np.std(standardized_y_train, axis=0))  # std 应该是 1

    # 初始化模型
    lm = SGDRegressor(loss="squared_loss", penalty="none", max_iter=args.num_epochs)

    # 训练
    res = lm.fit(X=standardized_X_train, y=standardized_y_train)
    print(res)

    # 预测 (还未标准化)
    pred_train = (lm.predict(standardized_X_train) * np.sqrt(y_scaler.var_)) + y_scaler.mean_
    pred_test = (lm.predict(standardized_X_test) * np.sqrt(y_scaler.var_)) + y_scaler.mean_


    # 训练和测试集上的均方误差 MSE
    train_mse = np.mean((y_train - pred_train) ** 2)
    test_mse = np.mean((y_test - pred_test) ** 2)
    print("train_MSE: {0:.2f}, test_MSE: {1:.2f}".format(train_mse, test_mse))


    # 图例大小
    plt.figure(figsize=(15, 5))

    # 画出训练数据
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plt.scatter(X_train, y_train, label="y_train")
    plt.plot(X_train, pred_train, color="red", linewidth=1, linestyle="-", label="lm")
    plt.legend(loc='lower right')

    # 画出测试数据
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plt.scatter(X_test, y_test, label="y_test")
    plt.plot(X_test, pred_test, color="red", linewidth=1, linestyle="-", label="lm")
    plt.legend(loc='lower right')

    # 显示图例
    plt.show()

    # 推论
    # 传入我们自己的输入值
    X_infer = np.array((0, 1, 2), dtype=np.float32)
    standardized_X_infer = X_scaler.transform(X_infer.reshape(-1, 1))
    pred_infer = (lm.predict(standardized_X_infer) * np.sqrt(y_scaler.var_)) + y_scaler.mean_
    print(pred_infer)
    df.head(3)
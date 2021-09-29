import os
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from binascii import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns', None)

local_path = "../data/housing/"
url_path = "https://github.com/ageron/handson-ml/blob/master/datasets/housing/housing.tgz"
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None): #无参数
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X [:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# 第一步获取数据
def fetch_hoursing_data():
    os.makedirs(local_path, exist_ok=True)
    # housing_path = os.path.join(local_path, "housing.tgz")
    # urllib.request.urlretrieve(url_path, housing_path)
    hoursing_tgz = tarfile.open("%s/housing.tgz"%local_path)
    hoursing_tgz.extractall(path=local_path)
    hoursing_tgz.close()
# 第二步，快速看数据
def look_up_data():
    housing = pd.read_csv("%s/housing.csv"%local_path)
    print(housing.head()) #看头部
    print(housing.info()) #看简介
    print(housing["ocean_proximity"].value_counts()) #文本类型的分布
    print(housing.describe()) #数值类特征的分布情况
    housing.hist(bins=50, figsize=(20, 15)) #图形化展示分布
    plt.show()
# 第三步，创建测试数据
def split_train_test(housing, test_ratio):
    # 缺陷，每次运行会重新生成，导致测试集会变动，如此可以看到整个数据集；
    # 解决方案1：运行一次后，保存，后续读取即可
    # 解决方案2：permutation的随机性通过随机种子设定，导致重复生成，会固定
    # 解决后，缺陷：获取更新的数据集后(新增的量少),再做切分，此时原训练数据会进入测试集,导致测试集不够稳定
    # 解决方案：通过增加样本唯一标示，来判断是否进入测试集
    np.random.seed(42)
    shuffle_index = np.random.permutation(len(housing))
    test_size = int(test_ratio * len(housing))
    test_data_index = shuffle_index[:test_size]
    train_data_index = shuffle_index[test_size:]
    test_data = housing.iloc[test_data_index]
    train_data = housing.iloc[train_data_index]
    return train_data,test_data
def split_train_test_by_id(housing, test_ratio, id_column):
    def test_set_check(id, test_ratio):
        return crc32(np.int64(id)) & 0xffffffff < test_ratio*2**32
    ids = housing[id_column]
    in_test_set = ids.apply(lambda id:test_set_check(id, test_ratio))
    return housing.loc[~in_test_set],housing.loc[in_test_set]
def creat_test_data():
    housing = pd.read_csv("%s/housing.csv" % local_path)
    # 自行构造测试数据集
    test_ratio = 0.2
    train_data, test_data = split_train_test(housing, test_ratio)
    print(len(test_data), len(train_data), len(housing))
    print(test_data.head())
    # 行为id缺陷：增数据必然是不删只增，显然不合理
    # 位置id：可能存在不同区域，同id，因为出现粗糙估计
    housing_with_index = housing.reset_index()
    housing_with_index["id"] = housing["longitude"] * 1000 + housing["latitude"]
    train_data, test_data = split_train_test_by_id(housing_with_index, test_ratio, "id")
    print(len(test_data), len(train_data), len(housing))
    # print(test_data.head())
    # sklearn的方法
    train_data, test_data = train_test_split(housing, test_size=0.2, random_state=42)
    print(len(test_data), len(train_data), len(housing))
    print(test_data.head())
def look_up_diff_sample_way():
    # 纯随机采样 VS 分层采样
    # 大数据集合下，二者没有差别
    # 小数据集合下，纯随机采样会出现属性/标签等抽样偏差，应该采用分层采样

    # 连续属性的数据，如何分段观察数据？均等间隔分段时需要保证每段的数据量比较充足
    housing = pd.read_csv("%s/housing.csv" % local_path)
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                   labels=[1, 2, 3, 4, 5])
    # housing["income_cat"].hist()
    # plt.show()

    # n_splits形成多少个分组也就是重复操作多少次，便于做多次, test_size设置测试集占比
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_data = housing.loc[train_index]
        strat_test_data = housing.loc[test_index]
    train_data, test_data = train_test_split(housing, test_size=0.2, random_state=42)
    strat_test_df = strat_test_data["income_cat"].value_counts(normalize=True).to_frame(name="StratifiedTest")
    random_test_df = test_data["income_cat"].value_counts(normalize=True).to_frame(name="RandomTest")
    strat_train_df = strat_train_data["income_cat"].value_counts(normalize=True).to_frame(name="StratifiedTrain")
    random_train_df = train_data["income_cat"].value_counts(normalize=True).to_frame(name="RandomTrain")
    all_df = housing["income_cat"].value_counts(normalize=True).to_frame(name="Overall")
    res_df = pd.concat([all_df, strat_test_df, random_test_df, strat_train_df, random_train_df], axis=1).sort_index()
    res_df["StratifiedTestErr(%)"] = (res_df["StratifiedTest"] - res_df["Overall"]) / res_df["Overall"] * 100.0
    res_df["RandomTestErr(%)"] = (res_df["RandomTest"] - res_df["Overall"]) / res_df["Overall"] * 100.0
    res_df["StratifiedTrainErr(%)"] = (res_df["StratifiedTrain"] - res_df["Overall"]) / res_df["Overall"] * 100.0
    res_df["RandomTrainErr(%)"] = (res_df["RandomTrain"] - res_df["Overall"]) / res_df["Overall"] * 100.0
    # 可以观察到两个现象
    # 1. 分层抽样明显比随机采样要优
    # 2. 数据量多时，纯随机采样的偏差会小
    print(res_df)

    pass
# 第四步，深度看训练数据
def look_up_data_deep_info():
    # 采样数据
    housing = pd.read_csv("%s/housing.csv" % local_path)
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                   labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_data = housing.loc[train_index]
        strat_test_data = housing.loc[test_index]
    # 开始观察
    train_data = strat_train_data.drop(["income_cat"], axis=1)
    housing = train_data.copy()
    # housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4
    #              ,s=housing["population"]/100,label="population",figsize=(10, 7)
    #              ,c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
    # plt.legend()
    # 观察方式，看位置与人口关联，看房价与位置关联，看房价与人口关联，也就是3者的两两关联，以及三者共同的关联等信息
    # plt.show()

    # 相关性分析
    corr_matrix = housing.corr() #线性相关性，另外有个点，相关性值不是斜率的含义
    print(corr_matrix["median_house_value"].sort_values(ascending=False))
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8)) #可观察图中的散点图观察非线性的关系
    housing.plot(kind="scatter", x="median_income", y="median_house_value",alpha=0.1)
    plt.show()

    pass
# 第五步，根据观察的数据，尝试生产新特征
def cross_diff_featrues():
    # 采样数据
    housing = pd.read_csv("%s/housing.csv" % local_path)
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                   labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_data = housing.loc[train_index]
        strat_test_data = housing.loc[test_index]
    # 针对训练数据操作
    train_data = strat_train_data.drop(["income_cat"], axis=1)
    housing = train_data.copy()

    # 根据业务，由经验可以简单组合，通过线性关系简单获取到额外的组合特征
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"] #平均每个家庭的房间数
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"] #平均每个房间的卧室数
    housing["population_per_household"] = housing["population"] / housing["households"] #平均每个家庭的人数
    corr_matrix = housing.corr() #线性相关性，另外有个点，相关性值不是斜率的含义
    print(type(corr_matrix["median_house_value"]))
    print(corr_matrix["median_house_value"].apply(lambda x:abs(x)).sort_values(ascending=False))
# 第六步，开始准备数据
def perpare_data():
    # 采样数据
    housing = pd.read_csv("%s/housing.csv" % local_path)
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                   labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_data = housing.loc[train_index]
        strat_test_data = housing.loc[test_index]
    train_data = strat_train_data.drop(["income_cat"], axis=1)

    housing = train_data.drop("median_house_value", axis=1)
    housing_label = train_data["median_house_value"].copy() #

    #数据清理
    # housing.dropna(subset=["total_bedrooms"], inplace=True) # 指定列上删除某缺失值的行
    # housing.drop("total_bedrooms", axis=1, inplace=True) # 删除默认值的列
    # median = housing["total_bedrooms"].median() #中位数填充默认值
    # print(median)
    # print(any(housing["total_bedrooms"].isnull()))
    # housing["total_bedrooms"].fillna(median, inplace=True)
    # print(any(housing["total_bedrooms"].isnull()))
    # sklearn 搞定中位数
    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop(["ocean_proximity"], axis=1)
    imputer.fit(housing_num)
    print(imputer.statistics_)
    X = imputer.transform(housing_num) #numpy
    housing_tr = pd.DataFrame(X, columns=housing_num.columns,index=housing_num.index)
    print(housing_tr.head())
    # 尝试处理字符串类型
    housing_cat = housing[["ocean_proximity"]]
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded= ordinal_encoder.fit_transform(housing_cat)
    print(ordinal_encoder.categories_)
    one_hot_encoder = OneHotEncoder()
    housing_cat_1hot = one_hot_encoder.fit_transform(housing_cat)
    print(housing_cat_1hot.toarray()[:3])

    # 尝试自定的转化器
    attr_adder = CombinedAttributesAdder()
    print(type(housing.values))
    household_extra_attribs = attr_adder.transform(housing.values)
    # print(household_extra_attribs)

    # max-min特征缩放
    # max_min_scaler = MinMaxScaler()
    # X = max_min_scaler.fit_transform(housing_tr)
    # housing_tr_norm = pd.DataFrame(X, columns=housing_tr.columns,index=housing_tr.index)
    # print(housing_tr_norm.head())
    # 标注化，受异常值影响小，但值非0~1之间
    stand_scaler = StandardScaler()
    X = stand_scaler.fit_transform(X)
    print(X[:3])

    #pip流操作
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler())
    ])
    housing_num_tr = num_pipeline.fit_transform(housing_num)
    print(housing_num_tr[:3])

    print(type(housing_num))
    num_attribs = list(housing_num)
    print(num_attribs)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs)
    ]) #返回中存在稀疏比，是否满足阈值，满足可能返回稀疏矩阵
    housing_prepared = full_pipeline.fit_transform(housing)
    print(housing_prepared[:3])
# 第七步，训练数据
def train_data():
    # 采样数据
    housing = pd.read_csv("%s/housing.csv" % local_path)
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                   labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_data = housing.loc[train_index]
        strat_test_data = housing.loc[test_index]
    train_data = strat_train_data.drop(["income_cat"], axis=1)

    housing = train_data.drop("median_house_value", axis=1)
    housing_label = train_data["median_house_value"].copy() #

    #pip流操作
    # 数值处理
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler())
    ])
    housing_num = housing.drop(["ocean_proximity"], axis=1)
    housing_num_tr = num_pipeline.fit_transform(housing_num)
    print(housing_num_tr[:3])
    #文字处理
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs)
    ]) #返回中存在稀疏比，是否满足阈值，满足可能返回稀疏矩阵
    housing_prepared = full_pipeline.fit_transform(housing)
    print(housing_prepared[:3])

    # 开始训练
    # lin_reg = LinearRegression()
    # lin_reg = DecisionTreeRegressor()
    lin_reg = RandomForestRegressor()
    lin_reg.fit(housing_prepared, housing_label)

    housing_predict = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_label, housing_predict)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)

    # 线性回归，欠拟合
    tree_reg = LinearRegression()
    scores = cross_val_score(tree_reg, housing_prepared, housing_label, scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    print(tree_rmse_scores)
    print(tree_rmse_scores.mean())
    print(tree_rmse_scores.std())

    # 决策数回归，过拟合
    tree_reg = DecisionTreeRegressor()
    scores = cross_val_score(tree_reg, housing_prepared, housing_label, scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    print(tree_rmse_scores)
    print(tree_rmse_scores.mean())
    print(tree_rmse_scores.std())

    # 随机森林回归，依然过拟合
    tree_reg = RandomForestRegressor()
    scores = cross_val_score(tree_reg, housing_prepared, housing_label, scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    print(tree_rmse_scores)
    print(tree_rmse_scores.mean())
    print(tree_rmse_scores.std())
# 第八步，微调模型参数
def search_model_params():
    pass

search_model_params()







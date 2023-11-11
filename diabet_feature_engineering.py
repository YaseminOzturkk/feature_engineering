#########################################
# Diyabet Feature Engineering
#########################################

###########################################
# İş Problemi
###########################################
"""Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin
edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir. Modeli
geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını
gerçekleştirmeniz beklenmektedir"""

#############################################
# Veri seti Hikayesi
#############################################

"""Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin 
parçasıdır. ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde 
olan Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir."""

# Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu,
# 0 ise negatif oluşunu belirtmektedir.


"""Değişkenler

- Pregnancies: Hamilelik sayısı
- Glucose: Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu
- Blood Pressure: Kan Basıncı (Küçük tansiyon) (mm Hg)
- SkinThickness: Cilt Kalınlığı
- Insulin: 2 saatlik serum insülini (mu U/ml)
- DiabetesPedigreeFunction: Fonksiyon (Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu)
- BMI: Vücut kitle endeksi
- Age: Yaş (yıl)
- Outcome: Hastalığa sahip (1) ya da değil (0)"""



# Kütüphanelerin Import Edilmesi

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


def load_dataset():
    data = pd.read_csv("C:/Users/yasmi/Desktop/feature_engineering/feature_engineering/diabetes.csv")
    return data

df = load_dataset()

###################################
# GÖREV 1: Keşifçi Veri Analizi
###################################

# Adım 1: Genel resmi inceleyiniz.

def check_df(df, head=5):
    print("###################### SHAPE #########")
    print(df.shape)
    print("###################### TYPES #########")
    print(df.dtypes)
    print("###################### HEAD #########")
    print(df.head())
    print("###################### TAIL #########")
    print(df.tail())
    print("###################### NA #########")
    print(df.isnull().sum())
    print("###################### SUMMARY STATISTICS #########")
    print(df.describe().T)
check_df(df)

df.columns = [col.upper() for col in df.columns]
df.head()

# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız


# Kategorik değişken analizi
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Raio": 100 * dataframe[col_name].value_counts() / len(dataframe),
                        "IS_NULL": dataframe[col_name].isnull().any()}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)



for col in cat_cols:
    cat_summary(df, col, plot=True)

# Sayısal değişken analizi

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)

# Not: # Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu,
# # 0 ise negatif oluşunu belirtmektedir.

def target_summary_with_num(dataframe, target, numerical_cols):
    print(dataframe.groupby(target).agg({numerical_cols: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "OUTCOME", col)


# Adım 5: Aykırı gözlem analizi yapınız.

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(check_outlier(df, col))

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index
for col in num_cols:
    print(grab_outliers(df, col, index=True))


# Adım 6: Eksik gözlem analizi yapınız.

# eksik gözlem var mı yok mu?

df.isnull().values.any()

# degiskenlerdeki eksik deger sayısı
df.isnull().sum()

# degiskenlerdeki tam deger sayısı
df.notnull().sum()

# veri setindeki toplam eksik deger sayısı
df.isnull().sum().sum()

# Adım 7: Korelasyon analizi yapınız.

corr = df[num_cols].corr()
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show(block=True)

###################################
# Görev 2: Feature Engineering
###################################


# Adım 1: Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0
# olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
# değerlere işlemleri uygulayabilirsiniz.


# İlgili değişkenlerdeki 0 değerini NaN ile değiştirelim.
cols_to_replace_na = ['GLUCOSE', 'INSULIN', 'BLOODPRESSURE', 'SKINTHICKNESS', 'BMI', 'DIABETESPEDIGREEFUNCTION', 'AGE']
df[cols_to_replace_na] = df[cols_to_replace_na].replace(0, np.nan)

df.isnull().sum()

# Aykırı değer problemini çözelim

for col in num_cols:
    print(col, check_outlier(df, col))

# Aykırı değerleri tresholds değerler ile baskılayalım.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    replace_with_thresholds(df, col)

# Eksik değer problemini çözelim

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)
df.shape
df["INSULIN"].fillna(df["INSULIN"].median(), inplace=True)
df["SKINTHICKNESS"].fillna(df["SKINTHICKNESS"].median(), inplace=True)
df["BLOODPRESSURE"].fillna(df["BLOODPRESSURE"].median(), inplace=True)
df["BMI"].fillna(df["BMI"].median(), inplace=True)
df["GLUCOSE"].fillna(df["GLUCOSE"].median(), inplace=True)
df.dropna()

df.isnull().sum()
# Adım 2: Yeni değişkenler oluşturunuz.

# BMI_CAT Feature

"""18,5 kg/m2 ve daha düşük değerler = Zayıf
18,5 ve 24,9 kg/m2 arasındaki değerler = Normal ağırlıkta
25,0 ve 29,9 kg/m2 arasındaki değerler = Kilolu
30,0 ve 34,9 kg/m2 arasındaki değerler = 1. derece obezite
35,0 ve 39,9 kg/m2 arasındaki değerler = 2. derece obezite
40 kg/m2 ve üzerindeki değerler = 3. derece obezite"""


df.loc[(df["BMI"] < 18.5), "BMI_CAT"] = "weak"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] <= 24.9),  "BMI_CAT")] = "normal weight"
df.loc[((df["BMI"] >= 25.0) & (df["BMI"] <= 29.9),  "BMI_CAT")] = "overweight"
df.loc[((df["BMI"] >= 30.0) & (df["BMI"] <= 34.9),  "BMI_CAT")] = "1st degree obesity"
df.loc[((df["BMI"] >= 35.0) & (df["BMI"] <= 39.9),  "BMI_CAT")] = "2nd degree obesity"
df.loc[((df["BMI"] >= 40.0),  "BMI_CAT")] = "3rd degree obesity"


# AGE_CAT Feature

df['AGE_CAT'] = pd.qcut(df['AGE'], q=3, labels=['young', 'mature', 'senior'])

# PREGNANCY_CAT Feature

df.loc[(df['PREGNANCIES'] == 0), 'PREGNANCY_CAT'] = 'no_pregnancy'
df.loc[(df['PREGNANCIES'] == 1), 'PREGNANCY_CAT'] = 'one_pregnancy'
df.loc[(df['PREGNANCIES'] > 1), 'PREGNANCY_CAT'] = 'multi_pregnancy'

df["PREGNANCIES"].isnull().sum()

# GLUCOSE_CAT Feature

df.loc[(df['GLUCOSE'] >= 170), 'GLUCOSE_CAT'] = 'diabetes'
df.loc[(df['GLUCOSE'] >= 105) & (df['GLUCOSE'] < 170), 'GLUCOSE_CAT'] = 'prediabetes'
df.loc[(df['GLUCOSE'] < 105) & (df['GLUCOSE'] > 70), 'GLUCOSE_CAT'] = 'normal'
df.loc[(df['GLUCOSE'] <= 70), 'GLUCOSE_CAT'] = 'hypoglycemia'

df["GLUCOSE"].isnull().sum()

# BLOODPRESSURE_CAT Feature

df.loc[(df['BLOODPRESSURE'] >= 110), 'BLOODPRESSURE_CAT'] = 'hypersensitive crisis'
df.loc[(df['BLOODPRESSURE'] >= 90) & (df['BLOODPRESSURE'] < 110), 'BLOODPRESSURE_CAT'] = 'hypertension'
df.loc[(df['BLOODPRESSURE'] < 90) & (df['BLOODPRESSURE'] > 70), 'BLOODPRESSURE_CAT'] = 'normal'
df.loc[(df['BLOODPRESSURE'] <= 70), 'BLOODPRESSURE_CAT'] = 'low'

df["BLOODPRESSURE"].isnull().sum()

# INSULIN_CAT Feature

df.loc[(df['INSULIN'] >= 160), 'INSULIN_CAT'] = 'high'
df.loc[(df['INSULIN'] < 160) & (df['INSULIN'] >= 16), 'INSULIN_CAT'] = 'normal'
df.loc[(df['INSULIN'] < 16), 'INSULIN_CAT'] = 'low'

df["INSULIN"].isnull().sum()
df.head()


# Adım 3: Encoding işlemlerini gerçekleştiriniz.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Label Encoding
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

# Rare Encoding

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "OUTCOME", cat_cols)

# One - Hot Encoding

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, ohe_cols)

df.isnull().sum()
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
rare_analyser(df, "OUTCOME", cat_cols)

df.shape
# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


# Adım 5: Model oluşturunuz.

df.isnull().sum()

y = df["OUTCOME"]
X = df.drop(["OUTCOME"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)



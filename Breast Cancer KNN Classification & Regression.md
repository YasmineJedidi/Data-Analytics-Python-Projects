1. [Importing Libraries](#Importing-Libraries)
2. [Processing Data](#Processing-Data)
    * [Dataset Description](#Dataset-Description)
    * [Data Cleaning](#Cleaning-Data) 
3. [KNN Classification](#KNN-Classification)
    * [Hyperparameter Tuning](#Hyperparameter-Tuning-Using-GridSearchCV) 
    * [Interpretation](#Interpretation)
4. [KNN Regression](#KNN-Regression)
    * [Interpretation](#Regression-Interpretation)

# Importing Libraries


```python
import pandas as pd
import numpy as np
```


```python
##### Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
#####sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
```


```python
#####sklearn KNN classification
import sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier
```

# Processing Data


```python
df = pd.read_csv(r'C:\Users\yesmi\OneDrive\Desktop\Junior Spring Semester\Data Mining\Tutorial\Breast Cancer KNN Classification and Regression\Breast Cancer Wisconsin (Diagnostic).csv')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>...</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842302</td>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>842517</td>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84300903</td>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84348301</td>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84358402</td>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>



## Dataset Description

This dataset contains 570 medical recods, each case is assiged an ID. 
Data is on:
1. The dignosis: (M = malignant, B = benign)

*The means, standard errors, and worst cases of:*

2. radius
3. texture
4. perimeter
5. area
6. smoothness
7. compactness
8. concavity
9. concave points
10. symmetry
11. fractal_dimension


```python
df.shape
```




    (569, 32)




```python
df.info( )
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 569 entries, 0 to 568
    Data columns (total 32 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   id                       569 non-null    int64  
     1   diagnosis                569 non-null    object 
     2   radius_mean              569 non-null    float64
     3   texture_mean             569 non-null    float64
     4   perimeter_mean           569 non-null    float64
     5   area_mean                569 non-null    float64
     6   smoothness_mean          569 non-null    float64
     7   compactness_mean         569 non-null    float64
     8   concavity_mean           569 non-null    float64
     9   concave points_mean      569 non-null    float64
     10  symmetry_mean            569 non-null    float64
     11  fractal_dimension_mean   569 non-null    float64
     12  radius_se                569 non-null    float64
     13  texture_se               569 non-null    float64
     14  perimeter_se             569 non-null    float64
     15  area_se                  569 non-null    float64
     16  smoothness_se            569 non-null    float64
     17  compactness_se           569 non-null    float64
     18  concavity_se             569 non-null    float64
     19  concave points_se        569 non-null    float64
     20  symmetry_se              569 non-null    float64
     21  fractal_dimension_se     569 non-null    float64
     22  radius_worst             569 non-null    float64
     23  texture_worst            569 non-null    float64
     24  perimeter_worst          569 non-null    float64
     25  area_worst               569 non-null    float64
     26  smoothness_worst         569 non-null    float64
     27  compactness_worst        569 non-null    float64
     28  concavity_worst          569 non-null    float64
     29  concave points_worst     569 non-null    float64
     30  symmetry_worst           569 non-null    float64
     31  fractal_dimension_worst  569 non-null    float64
    dtypes: float64(30), int64(1), object(1)
    memory usage: 142.4+ KB
    


```python
df.nunique()
```




    id                         569
    diagnosis                    2
    radius_mean                456
    texture_mean               479
    perimeter_mean             522
    area_mean                  539
    smoothness_mean            474
    compactness_mean           537
    concavity_mean             537
    concave points_mean        542
    symmetry_mean              432
    fractal_dimension_mean     499
    radius_se                  540
    texture_se                 519
    perimeter_se               533
    area_se                    528
    smoothness_se              547
    compactness_se             541
    concavity_se               533
    concave points_se          507
    symmetry_se                498
    fractal_dimension_se       545
    radius_worst               457
    texture_worst              511
    perimeter_worst            514
    area_worst                 544
    smoothness_worst           411
    compactness_worst          529
    concavity_worst            539
    concave points_worst       492
    symmetry_worst             500
    fractal_dimension_worst    535
    dtype: int64




```python
df.diagnosis.unique()
```




    array(['M', 'B'], dtype=object)



## Cleaning Data


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>symmetry_mean</th>
      <th>...</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5.690000e+02</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>...</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.037183e+07</td>
      <td>14.127292</td>
      <td>19.289649</td>
      <td>91.969033</td>
      <td>654.889104</td>
      <td>0.096360</td>
      <td>0.104341</td>
      <td>0.088799</td>
      <td>0.048919</td>
      <td>0.181162</td>
      <td>...</td>
      <td>16.269190</td>
      <td>25.677223</td>
      <td>107.261213</td>
      <td>880.583128</td>
      <td>0.132369</td>
      <td>0.254265</td>
      <td>0.272188</td>
      <td>0.114606</td>
      <td>0.290076</td>
      <td>0.083946</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.250206e+08</td>
      <td>3.524049</td>
      <td>4.301036</td>
      <td>24.298981</td>
      <td>351.914129</td>
      <td>0.014064</td>
      <td>0.052813</td>
      <td>0.079720</td>
      <td>0.038803</td>
      <td>0.027414</td>
      <td>...</td>
      <td>4.833242</td>
      <td>6.146258</td>
      <td>33.602542</td>
      <td>569.356993</td>
      <td>0.022832</td>
      <td>0.157336</td>
      <td>0.208624</td>
      <td>0.065732</td>
      <td>0.061867</td>
      <td>0.018061</td>
    </tr>
    <tr>
      <th>min</th>
      <td>8.670000e+03</td>
      <td>6.981000</td>
      <td>9.710000</td>
      <td>43.790000</td>
      <td>143.500000</td>
      <td>0.052630</td>
      <td>0.019380</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.106000</td>
      <td>...</td>
      <td>7.930000</td>
      <td>12.020000</td>
      <td>50.410000</td>
      <td>185.200000</td>
      <td>0.071170</td>
      <td>0.027290</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.156500</td>
      <td>0.055040</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.692180e+05</td>
      <td>11.700000</td>
      <td>16.170000</td>
      <td>75.170000</td>
      <td>420.300000</td>
      <td>0.086370</td>
      <td>0.064920</td>
      <td>0.029560</td>
      <td>0.020310</td>
      <td>0.161900</td>
      <td>...</td>
      <td>13.010000</td>
      <td>21.080000</td>
      <td>84.110000</td>
      <td>515.300000</td>
      <td>0.116600</td>
      <td>0.147200</td>
      <td>0.114500</td>
      <td>0.064930</td>
      <td>0.250400</td>
      <td>0.071460</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9.060240e+05</td>
      <td>13.370000</td>
      <td>18.840000</td>
      <td>86.240000</td>
      <td>551.100000</td>
      <td>0.095870</td>
      <td>0.092630</td>
      <td>0.061540</td>
      <td>0.033500</td>
      <td>0.179200</td>
      <td>...</td>
      <td>14.970000</td>
      <td>25.410000</td>
      <td>97.660000</td>
      <td>686.500000</td>
      <td>0.131300</td>
      <td>0.211900</td>
      <td>0.226700</td>
      <td>0.099930</td>
      <td>0.282200</td>
      <td>0.080040</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.813129e+06</td>
      <td>15.780000</td>
      <td>21.800000</td>
      <td>104.100000</td>
      <td>782.700000</td>
      <td>0.105300</td>
      <td>0.130400</td>
      <td>0.130700</td>
      <td>0.074000</td>
      <td>0.195700</td>
      <td>...</td>
      <td>18.790000</td>
      <td>29.720000</td>
      <td>125.400000</td>
      <td>1084.000000</td>
      <td>0.146000</td>
      <td>0.339100</td>
      <td>0.382900</td>
      <td>0.161400</td>
      <td>0.317900</td>
      <td>0.092080</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.113205e+08</td>
      <td>28.110000</td>
      <td>39.280000</td>
      <td>188.500000</td>
      <td>2501.000000</td>
      <td>0.163400</td>
      <td>0.345400</td>
      <td>0.426800</td>
      <td>0.201200</td>
      <td>0.304000</td>
      <td>...</td>
      <td>36.040000</td>
      <td>49.540000</td>
      <td>251.200000</td>
      <td>4254.000000</td>
      <td>0.222600</td>
      <td>1.058000</td>
      <td>1.252000</td>
      <td>0.291000</td>
      <td>0.663800</td>
      <td>0.207500</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 31 columns</p>
</div>




```python
df.isnull().values.any()
```




    False



There are no missing values.


```python
#####checking for duplicates

df.id.value_counts()
```




    842302     1
    90250      1
    901315     1
    9013579    1
    9013594    1
              ..
    873885     1
    873843     1
    873701     1
    873593     1
    92751      1
    Name: id, Length: 569, dtype: int64



There are no duplicate values.

# KNN Classification


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>...</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842302</td>
      <td>1</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>842517</td>
      <td>1</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84300903</td>
      <td>1</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84348301</td>
      <td>1</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84358402</td>
      <td>1</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
ctypes ={'M' : 1, 'B' : 0}

df['diagnosis'] = df['diagnosis'].map(ctypes) 
```


```python
#####Splitting the dataset into training and testing sets
x = df.iloc[:,2:]
y = df.iloc[:, 1]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.2)
```


```python
##### Scaling the data (min-max)
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
```


```python
#####optimal number of nearest neighbours
import math
math.sqrt(len(y_test))
```




    10.677078252031311




```python
KNN = KNeighborsClassifier(n_neighbors = 10, p = 2, metric = 'euclidean')
KNN.fit(x_train,y_train)
```




    KNeighborsClassifier(metric='euclidean', n_neighbors=10)




```python
y_pred = KNN.predict(x_test)
y_pred
```

    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    




    array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1,
           0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0,
           0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0,
           1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0,
           1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
           0, 1, 1, 0], dtype=int64)




```python
#####confusion matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
```

    [[66  1]
     [ 5 42]]
    


```python
#####accuracy score
print(accuracy_score(y_test,y_pred))
```

    0.9473684210526315
    


```python
print(classification_report(y_test,y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.93      0.99      0.96        67
               1       0.98      0.89      0.93        47
    
        accuracy                           0.95       114
       macro avg       0.95      0.94      0.94       114
    weighted avg       0.95      0.95      0.95       114
    
    

## Hyperparameter Tuning Using GridSearchCV


```python
from sklearn.model_selection import GridSearchCV
k_range = list(range(1, 31))
grid_params = { 'n_neighbors' : k_range,
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}

gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 3)
##### fitting the model on our train set
g_res = gs.fit(x_train,y_train)
```

    Fitting 5 folds for each of 180 candidates, totalling 900 fits
    [CV 1/5] END metric=minkowski, n_neighbors=1, weights=uniform;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=1, weights=uniform;, score=0.956 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=1, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=1, weights=uniform;, score=0.956 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=1, weights=uniform;, score=0.945 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=1, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=1, weights=distance;, score=0.956 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=1, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=1, weights=distance;, score=0.956 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=1, weights=distance;, score=0.945 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=2, weights=uniform;, score=0.945 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=2, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=2, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=2, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=2, weights=uniform;, score=0.945 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=2, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=2, weights=distance;, score=0.956 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=2, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=2, weights=distance;, score=0.956 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=2, weights=distance;, score=0.945 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=3, weights=uniform;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=3, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=3, weights=uniform;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=3, weights=uniform;, score=1.000 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=3, weights=uniform;, score=0.956 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=3, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=3, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=3, weights=distance;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=3, weights=distance;, score=1.000 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=3, weights=distance;, score=0.956 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=4, weights=uniform;, score=0.945 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=4, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=4, weights=uniform;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=4, weights=uniform;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=4, weights=uniform;, score=0.956 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=4, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=4, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=4, weights=distance;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=4, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=4, weights=distance;, score=0.956 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=5, weights=uniform;, score=0.945 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=5, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=5, weights=uniform;, score=0.956 total time=   0.0s
    

    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    

    [CV 4/5] END metric=minkowski, n_neighbors=5, weights=uniform;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=5, weights=uniform;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=5, weights=distance;, score=0.945 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=5, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=5, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=5, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=5, weights=distance;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=6, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=6, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=6, weights=uniform;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=6, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=6, weights=uniform;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=6, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=6, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=6, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=6, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=6, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=7, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=7, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=7, weights=uniform;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=7, weights=uniform;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=7, weights=uniform;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=7, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=7, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=7, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=7, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=7, weights=distance;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=8, weights=uniform;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=8, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=8, weights=uniform;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=8, weights=uniform;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=8, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=8, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=8, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=8, weights=distance;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=8, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=8, weights=distance;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=9, weights=uniform;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=9, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=9, weights=uniform;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=9, weights=uniform;, score=0.989 total time=   0.0s
    

    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    

    [CV 5/5] END metric=minkowski, n_neighbors=9, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=9, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=9, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=9, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=9, weights=distance;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=9, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=10, weights=uniform;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=10, weights=uniform;, score=0.956 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=10, weights=uniform;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=10, weights=uniform;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=10, weights=uniform;, score=0.956 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=10, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=10, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=10, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=10, weights=distance;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=10, weights=distance;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=11, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=11, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=11, weights=uniform;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=11, weights=uniform;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=11, weights=uniform;, score=0.956 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=11, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=11, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=11, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=11, weights=distance;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=11, weights=distance;, score=0.956 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=12, weights=uniform;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=12, weights=uniform;, score=0.956 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=12, weights=uniform;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=12, weights=uniform;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=12, weights=uniform;, score=0.956 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=12, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=12, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=12, weights=distance;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=12, weights=distance;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=12, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=13, weights=uniform;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=13, weights=uniform;, score=0.956 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=13, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=13, weights=uniform;, score=0.978 total time=   0.0s
    

    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    

    [CV 5/5] END metric=minkowski, n_neighbors=13, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=13, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=13, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=13, weights=distance;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=13, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=13, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=14, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=14, weights=uniform;, score=0.956 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=14, weights=uniform;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=14, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=14, weights=uniform;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=14, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=14, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=14, weights=distance;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=14, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=14, weights=distance;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=15, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=15, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=15, weights=uniform;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=15, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=15, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=15, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=15, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=15, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=15, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=15, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=16, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=16, weights=uniform;, score=0.956 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=16, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=16, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=16, weights=uniform;, score=0.956 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=16, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=16, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=16, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=16, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=16, weights=distance;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=17, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=17, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=17, weights=uniform;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=17, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=17, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=17, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=17, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=17, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=17, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=17, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=18, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=18, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=18, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=18, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=18, weights=uniform;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=18, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=18, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=18, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=18, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=18, weights=distance;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=19, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=19, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=19, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=19, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=19, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=19, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=19, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=19, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=19, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=19, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=20, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=20, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=20, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=20, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=20, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=20, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=20, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=20, weights=distance;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=20, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=20, weights=distance;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=21, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=21, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=21, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=21, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=21, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=21, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=21, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=21, weights=distance;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=21, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=21, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=22, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=22, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=22, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=22, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=22, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=22, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=22, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=22, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=22, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=22, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=23, weights=uniform;, score=0.901 total time=   0.0s
    

    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    

    [CV 2/5] END metric=minkowski, n_neighbors=23, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=23, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=23, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=23, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=23, weights=distance;, score=0.901 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=23, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=23, weights=distance;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=23, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=23, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=24, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=24, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=24, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=24, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=24, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=24, weights=distance;, score=0.901 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=24, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=24, weights=distance;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=24, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=24, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=25, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=25, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=25, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=25, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=25, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=25, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=25, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=25, weights=distance;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=25, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=25, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=26, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=26, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=26, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=26, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=26, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=26, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=26, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=26, weights=distance;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=26, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=26, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=27, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=27, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=27, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=27, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=27, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=27, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=27, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=27, weights=distance;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=27, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=27, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=28, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=28, weights=uniform;, score=0.967 total time=   0.0s
    

    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    

    [CV 3/5] END metric=minkowski, n_neighbors=28, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=28, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=28, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=28, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=28, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=28, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=28, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=28, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=29, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=29, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=29, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=29, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=29, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=29, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=29, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=29, weights=distance;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=29, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=29, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=30, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=30, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=30, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=30, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=30, weights=uniform;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=minkowski, n_neighbors=30, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=minkowski, n_neighbors=30, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=minkowski, n_neighbors=30, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=minkowski, n_neighbors=30, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=minkowski, n_neighbors=30, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=1, weights=uniform;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=1, weights=uniform;, score=0.956 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=1, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=1, weights=uniform;, score=0.956 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=1, weights=uniform;, score=0.945 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=1, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=1, weights=distance;, score=0.956 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=1, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=1, weights=distance;, score=0.956 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=1, weights=distance;, score=0.945 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=2, weights=uniform;, score=0.945 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=2, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=2, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=2, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=2, weights=uniform;, score=0.945 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=2, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=2, weights=distance;, score=0.956 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=2, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=2, weights=distance;, score=0.956 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=2, weights=distance;, score=0.945 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=3, weights=uniform;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=3, weights=uniform;, score=0.978 total time=   0.0s
    

    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    

    [CV 3/5] END metric=euclidean, n_neighbors=3, weights=uniform;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=3, weights=uniform;, score=1.000 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=3, weights=uniform;, score=0.956 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=3, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=3, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=3, weights=distance;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=3, weights=distance;, score=1.000 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=3, weights=distance;, score=0.956 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=4, weights=uniform;, score=0.945 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=4, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=4, weights=uniform;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=4, weights=uniform;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=4, weights=uniform;, score=0.956 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=4, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=4, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=4, weights=distance;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=4, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=4, weights=distance;, score=0.956 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=5, weights=uniform;, score=0.945 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=5, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=5, weights=uniform;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=5, weights=uniform;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=5, weights=uniform;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=5, weights=distance;, score=0.945 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=5, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=5, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=5, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=5, weights=distance;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=6, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=6, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=6, weights=uniform;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=6, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=6, weights=uniform;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=6, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=6, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=6, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=6, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=6, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=7, weights=uniform;, score=0.923 total time=   0.0s
    

    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    

    [CV 2/5] END metric=euclidean, n_neighbors=7, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=7, weights=uniform;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=7, weights=uniform;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=7, weights=uniform;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=7, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=7, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=7, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=7, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=7, weights=distance;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=8, weights=uniform;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=8, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=8, weights=uniform;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=8, weights=uniform;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=8, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=8, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=8, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=8, weights=distance;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=8, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=8, weights=distance;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=9, weights=uniform;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=9, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=9, weights=uniform;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=9, weights=uniform;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=9, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=9, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=9, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=9, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=9, weights=distance;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=9, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=10, weights=uniform;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=10, weights=uniform;, score=0.956 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=10, weights=uniform;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=10, weights=uniform;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=10, weights=uniform;, score=0.956 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=10, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=10, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=10, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=10, weights=distance;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=10, weights=distance;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=11, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=11, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=11, weights=uniform;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=11, weights=uniform;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=11, weights=uniform;, score=0.956 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=11, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=11, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=11, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=11, weights=distance;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=11, weights=distance;, score=0.956 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=12, weights=uniform;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=12, weights=uniform;, score=0.956 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=12, weights=uniform;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=12, weights=uniform;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=12, weights=uniform;, score=0.956 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=12, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=12, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=12, weights=distance;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=12, weights=distance;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=12, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=13, weights=uniform;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=13, weights=uniform;, score=0.956 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=13, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=13, weights=uniform;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=13, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=13, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=13, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=13, weights=distance;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=13, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=13, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=14, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=14, weights=uniform;, score=0.956 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=14, weights=uniform;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=14, weights=uniform;, score=0.967 total time=   0.0s
    

    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    

    [CV 5/5] END metric=euclidean, n_neighbors=14, weights=uniform;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=14, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=14, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=14, weights=distance;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=14, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=14, weights=distance;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=15, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=15, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=15, weights=uniform;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=15, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=15, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=15, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=15, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=15, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=15, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=15, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=16, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=16, weights=uniform;, score=0.956 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=16, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=16, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=16, weights=uniform;, score=0.956 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=16, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=16, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=16, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=16, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=16, weights=distance;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=17, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=17, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=17, weights=uniform;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=17, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=17, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=17, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=17, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=17, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=17, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=17, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=18, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=18, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=18, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=18, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=18, weights=uniform;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=18, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=18, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=18, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=18, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=18, weights=distance;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=19, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=19, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=19, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=19, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=19, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=19, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=19, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=19, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=19, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=19, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=20, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=20, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=20, weights=uniform;, score=0.934 total time=   0.0s
    

    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    

    [CV 4/5] END metric=euclidean, n_neighbors=20, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=20, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=20, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=20, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=20, weights=distance;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=20, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=20, weights=distance;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=21, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=21, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=21, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=21, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=21, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=21, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=21, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=21, weights=distance;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=21, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=21, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=22, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=22, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=22, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=22, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=22, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=22, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=22, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=22, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=22, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=22, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=23, weights=uniform;, score=0.901 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=23, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=23, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=23, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=23, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=23, weights=distance;, score=0.901 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=23, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=23, weights=distance;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=23, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=23, weights=distance;, score=0.978 total time=   0.0s
    

    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    

    [CV 1/5] END metric=euclidean, n_neighbors=24, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=24, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=24, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=24, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=24, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=24, weights=distance;, score=0.901 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=24, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=24, weights=distance;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=24, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=24, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=25, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=25, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=25, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=25, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=25, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=25, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=25, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=25, weights=distance;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=25, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=25, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=26, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=26, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=26, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=26, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=26, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=26, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=26, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=26, weights=distance;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=26, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=26, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=27, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=27, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=27, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=27, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=27, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=27, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=27, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=27, weights=distance;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=27, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=27, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=28, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=28, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=28, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=28, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=28, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=28, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=28, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=28, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=28, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=28, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=29, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=29, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=29, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=29, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=29, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=29, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=29, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=29, weights=distance;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=29, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=29, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=30, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=30, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=30, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=30, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=30, weights=uniform;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=euclidean, n_neighbors=30, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=euclidean, n_neighbors=30, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=euclidean, n_neighbors=30, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=euclidean, n_neighbors=30, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=euclidean, n_neighbors=30, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=1, weights=uniform;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=1, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=1, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=1, weights=uniform;, score=0.945 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=1, weights=uniform;, score=0.956 total time=   0.0s
    

    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    

    [CV 1/5] END metric=manhattan, n_neighbors=1, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=1, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=1, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=1, weights=distance;, score=0.945 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=1, weights=distance;, score=0.956 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=2, weights=uniform;, score=0.956 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=2, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=2, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=2, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=2, weights=uniform;, score=0.956 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=2, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=2, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=2, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=2, weights=distance;, score=0.945 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=2, weights=distance;, score=0.956 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=3, weights=uniform;, score=0.956 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=3, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=3, weights=uniform;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=3, weights=uniform;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=3, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=3, weights=distance;, score=0.956 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=3, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=3, weights=distance;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=3, weights=distance;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=3, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=4, weights=uniform;, score=0.945 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=4, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=4, weights=uniform;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=4, weights=uniform;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=4, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=4, weights=distance;, score=0.945 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=4, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=4, weights=distance;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=4, weights=distance;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=4, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=5, weights=uniform;, score=0.956 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=5, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=5, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=5, weights=uniform;, score=0.989 total time=   0.0s
    

    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    

    [CV 5/5] END metric=manhattan, n_neighbors=5, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=5, weights=distance;, score=0.956 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=5, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=5, weights=distance;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=5, weights=distance;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=5, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=6, weights=uniform;, score=0.956 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=6, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=6, weights=uniform;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=6, weights=uniform;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=6, weights=uniform;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=6, weights=distance;, score=0.945 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=6, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=6, weights=distance;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=6, weights=distance;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=6, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=7, weights=uniform;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=7, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=7, weights=uniform;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=7, weights=uniform;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=7, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=7, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=7, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=7, weights=distance;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=7, weights=distance;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=7, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=8, weights=uniform;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=8, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=8, weights=uniform;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=8, weights=uniform;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=8, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=8, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=8, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=8, weights=distance;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=8, weights=distance;, score=0.989 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=8, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=9, weights=uniform;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=9, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=9, weights=uniform;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=9, weights=uniform;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=9, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=9, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=9, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=9, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=9, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=9, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=10, weights=uniform;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=10, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=10, weights=uniform;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=10, weights=uniform;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=10, weights=uniform;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=10, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=10, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=10, weights=distance;, score=0.956 total time=   0.0s
    

    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    

    [CV 4/5] END metric=manhattan, n_neighbors=10, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=10, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=11, weights=uniform;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=11, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=11, weights=uniform;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=11, weights=uniform;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=11, weights=uniform;, score=1.000 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=11, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=11, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=11, weights=distance;, score=0.956 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=11, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=11, weights=distance;, score=1.000 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=12, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=12, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=12, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=12, weights=uniform;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=12, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=12, weights=distance;, score=0.934 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=12, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=12, weights=distance;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=12, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=12, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=13, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=13, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=13, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=13, weights=uniform;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=13, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=13, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=13, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=13, weights=distance;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=13, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=13, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=14, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=14, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=14, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=14, weights=uniform;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=14, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=14, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=14, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=14, weights=distance;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=14, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=14, weights=distance;, score=0.967 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=15, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=15, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=15, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=15, weights=uniform;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=15, weights=uniform;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=15, weights=distance;, score=0.923 total time=   0.0s
    

    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    

    [CV 2/5] END metric=manhattan, n_neighbors=15, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=15, weights=distance;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=15, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=15, weights=distance;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=16, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=16, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=16, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=16, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=16, weights=uniform;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=16, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=16, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=16, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=16, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=16, weights=distance;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=17, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=17, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=17, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=17, weights=uniform;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=17, weights=uniform;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=17, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=17, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=17, weights=distance;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=17, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=17, weights=distance;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=18, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=18, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=18, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=18, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=18, weights=uniform;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=18, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=18, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=18, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=18, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=18, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=19, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=19, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=19, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=19, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=19, weights=uniform;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=19, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=19, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=19, weights=distance;, score=0.945 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=19, weights=distance;, score=0.978 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=19, weights=distance;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=20, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=20, weights=uniform;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=20, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=20, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=20, weights=uniform;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=20, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=20, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=20, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=20, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=20, weights=distance;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=21, weights=uniform;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=21, weights=uniform;, score=0.978 total time=   0.0s
    

    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    

    [CV 3/5] END metric=manhattan, n_neighbors=21, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=21, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=21, weights=uniform;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=21, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=21, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=21, weights=distance;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=21, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=21, weights=distance;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=22, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=22, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=22, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=22, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=22, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=22, weights=distance;, score=0.912 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=22, weights=distance;, score=0.978 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=22, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=22, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=22, weights=distance;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=23, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=23, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=23, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=23, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=23, weights=uniform;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=23, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=23, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=23, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=23, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=23, weights=distance;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=24, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=24, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=24, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=24, weights=uniform;, score=0.956 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=24, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=24, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=24, weights=distance;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=24, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=24, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=24, weights=distance;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=25, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=25, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=25, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=25, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=25, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=25, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=25, weights=distance;, score=0.956 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=25, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=25, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=25, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=26, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=26, weights=uniform;, score=0.967 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=26, weights=uniform;, score=0.923 total time=   0.0s
    

    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    

    [CV 4/5] END metric=manhattan, n_neighbors=26, weights=uniform;, score=0.956 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=26, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=26, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=26, weights=distance;, score=0.956 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=26, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=26, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=26, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=27, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=27, weights=uniform;, score=0.956 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=27, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=27, weights=uniform;, score=0.956 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=27, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=27, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=27, weights=distance;, score=0.956 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=27, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=27, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=27, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=28, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=28, weights=uniform;, score=0.945 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=28, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=28, weights=uniform;, score=0.956 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=28, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=28, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=28, weights=distance;, score=0.956 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=28, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=28, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=28, weights=distance;, score=0.989 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=29, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=29, weights=uniform;, score=0.945 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=29, weights=uniform;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=29, weights=uniform;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=29, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=29, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=29, weights=distance;, score=0.956 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=29, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=29, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=29, weights=distance;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=30, weights=uniform;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=30, weights=uniform;, score=0.945 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=30, weights=uniform;, score=0.934 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=30, weights=uniform;, score=0.956 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=30, weights=uniform;, score=0.978 total time=   0.0s
    [CV 1/5] END metric=manhattan, n_neighbors=30, weights=distance;, score=0.923 total time=   0.0s
    [CV 2/5] END metric=manhattan, n_neighbors=30, weights=distance;, score=0.945 total time=   0.0s
    [CV 3/5] END metric=manhattan, n_neighbors=30, weights=distance;, score=0.923 total time=   0.0s
    [CV 4/5] END metric=manhattan, n_neighbors=30, weights=distance;, score=0.967 total time=   0.0s
    [CV 5/5] END metric=manhattan, n_neighbors=30, weights=distance;, score=0.978 total time=   0.0s
    

    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    


```python
##### the best score
print("Best Score: ",g_res.best_score_)
```

    Best Score:  0.9692307692307693
    


```python
##### hyperparameters with the best score
print("Best Hyperparameters", g_res.best_params_)
```

    Best Hyperparameters {'metric': 'manhattan', 'n_neighbors': 11, 'weights': 'uniform'}
    


```python
KNN = KNeighborsClassifier(n_neighbors = 11, metric = 'manhattan',weights = 'uniform' )
KNN.fit(x_train,y_train)
```




    KNeighborsClassifier(metric='manhattan', n_neighbors=11)




```python
y_pred = KNN.predict(x_test)
y_pred
```

    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    




    array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1,
           0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0,
           0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0,
           1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0,
           1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1,
           0, 1, 1, 0], dtype=int64)




```python
print(accuracy_score(y_test,y_pred))
```

    0.9473684210526315
    


```python
cm = confusion_matrix(y_test,y_pred)
print(cm)
```

    [[65  2]
     [ 4 43]]
    


```python
y_pred = KNN.predict(x_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
```

    C:\Users\yesmi\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    




    Text(0.5, 23.52222222222222, 'Predicted Labels')




    
![png](output_39_2.png)
    


## Interpretation

The best Hyperparameters for KNN classification in this case are:manhattan as a distance, n_neighbors= 11,and uniform weights.

In the context of the breast cancer dataset, accuracy represents the percentage of correctly classified samples (i.e., how often the model correctly predicts whether a tumor is malignant or benign)

With a **0.947% accuracy score**, the model classifies 65 Malignant cancer cases correctly, and only 2 incorrectly (2 Patients have Benign breast cancer but have dignosed with Malignant cancer). And 4 patients with Malignant cancer have been dignosed with Benign breast cancer.

# KNN Regression


```python
#####libraries 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>...</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842302</td>
      <td>1</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>842517</td>
      <td>1</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84300903</td>
      <td>1</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84348301</td>
      <td>1</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84358402</td>
      <td>1</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 12)
x_train.shape, x_test.shape
```




    ((455, 30), (114, 30))




```python
#####Creating holders to store the model performance results
ML_Model = []
acc_train = []
acc_test = []
rmse_train = []
rmse_test = []
```


```python
def storeResults(model, a,b,c,d):
    ML_Model.append(model)
    acc_train.append(round(a, 3))
    acc_test.append(round(b, 3))
    rmse_train.append(round(c, 3))
    rmse_test.append(round(d, 3))
```


```python
knn = KNeighborsRegressor()
```


```python
k_range = list(range(1, 31))
weight_options = ['uniform', 'distance']
metric_options = ['minkowski', 'euclidean', 'manhattan']
param_grid = {'n_neighbors': k_range,
              'weights': weight_options,
              'metric': metric_options}
```


```python
##### instantiating the grid
knn_grid = GridSearchCV(knn, param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)

```


```python
knn_grid.fit(x_train, y_train)
```




    GridSearchCV(cv=10, estimator=KNeighborsRegressor(), n_jobs=-1,
                 param_grid={'metric': ['minkowski', 'euclidean', 'manhattan'],
                             'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                             13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                             23, 24, 25, 26, 27, 28, 29, 30],
                             'weights': ['uniform', 'distance']},
                 scoring='neg_mean_squared_error')




```python
#####Checking the best parameters for the model
knn_para = knn_grid.best_params_
print(knn_para)
```

    {'metric': 'manhattan', 'n_neighbors': 9, 'weights': 'distance'}
    


```python
knn_best.fit(x_train, y_train)
y_train_knn = knn_best.predict(x_train)
y_test_knn = knn_best.predict(x_test)
```


```python
acc_train_knn = knn_grid.score(x_train, y_train) 
acc_test_knn = knn_grid.score(x_test, y_test)
```


```python
acc_train_knn = knn_best.score(x_train, y_train)
acc_test_knn = knn_best.score(x_test, y_test)
```


```python
rmse_train_knn = np.sqrt(mean_squared_error(y_train, y_train_knn))
rmse_test_knn = np.sqrt(mean_squared_error(y_test, y_test_knn))
```


```python
storeResults('KNN', acc_train_knn, acc_test_knn, rmse_train_knn, rmse_test_knn)
```


```python
print("KNN: Accuracy on training Data: {:.3f}".format(acc_train_knn))
print("KNN: Accuracy on test Data: {:.3f}".format(acc_test_knn))
```

    KNN: Accuracy on training Data: 1.000
    KNN: Accuracy on test Data: 0.715
    


```python
training_accuracy = []
test_accuracy = []
##### try n_neighbors from 1 to 20
neighbors_settings = range(1, 31)
for n in neighbors_settings:
    ##### fit the model
    knn = KNeighborsRegressor(n_neighbors=n)
    knn.fit(x_train, y_train)
    ##### record training set accuracy
    training_accuracy.append(knn.score(x_train, y_train))
    ##### record generalization accuracy
    test_accuracy.append(knn.score(x_test, y_test))

#####plotting the training & testing accuracy for n_neighbours from 1 to 30
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")  
plt.xlabel("n_neighbors")
plt.legend()
```




    <matplotlib.legend.Legend at 0x21ce922b610>




    
![png](output_58_1.png)
    



```python
print("MSE:",mean_squared_error(y_test, y_test_knn))
```

    MSE: 0.07130882171721656
    

## Regression Interpretation
* The best hyperparameters found by the grid search are printed using the best_params_ attribute of the grid search object.
* The KNN model is fitted on the training data

KNN regression is not suitable for this dataset because it is a binary classification, and the accuracy is low compared to KNN classification. 


```python

```

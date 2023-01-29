#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Ignore warning
import warnings
warnings.filterwarnings("ignore")


# Normality test
from scipy.stats import chi2_contingency
from scipy.stats import shapiro
from scipy.stats import normaltest
import scipy.stats as st

#Feature Encoding 
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#CrossValidate 
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import cross_validate 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, KFold

#Modeling 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier

#Hypertuning Parameter 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# # Load Data

# In[2]:


df = pd.read_csv('loan_data_2007_2014.csv')


# In[3]:


pd.set_option('display.max_columns', None)
df.head(3)


# In[4]:


df.info()


# In[5]:


var = df.dtypes.reset_index()
var.columns = ['Kolom', 'Tipe_Data']


# In[6]:


fitur = df.columns.tolist()
list = []
list2 = []
for i in range(0, len(fitur)):
    x = df[fitur[i]].nunique()
    y = df[fitur[i]].unique()
    list.append(x)
    list2.append(y)
var['Nunique'] = list
var['Unique'] = list2


# In[7]:


var = var.sort_values(by='Nunique', ascending=False)


# In[8]:


pd.set_option('display.max_rows', None)
var


# In[9]:


print('Variabel bertipe object :', var[var['Tipe_Data']=='object'].shape)
print('Variabel bertipe numerik :', var[var['Tipe_Data']!='object'].shape)


# In[10]:


var_object = var[var['Tipe_Data']=='object']
var_object


# - Beberapa variabel/kolom yang memiliki cardinality tinggi akan didrop
# - Beberapa variabel juga seluruhnya missing value dan tidak mengandung informasi sama sekali akan didrop

# In[11]:


list_drop = ['Unnamed: 0', 'id', 'member_id','il_util', 'total_cu_tl', 'inq_fi', 'all_util','max_bal_bc', 'open_rv_24m',              'open_rv_12m', 'open_il_24m','total_bal_il', 'mths_since_rcnt_il', 'open_il_12m', 'open_il_6m', 'open_acc_6m',             'verification_status_joint','dti_joint','annual_inc_joint', 'inq_last_12m','url', 'emp_title', 'desc', 'title', 'zip_code']


# In[12]:


df = df.drop(list_drop, axis=1)


# In[13]:


num = df.loc[:, df.dtypes!=np.object]
cat = df.loc[:, df.dtypes==np.object]


# In[14]:


num.describe()


# In[15]:


cat.describe()


# In[16]:


df.duplicated().sum()


# - Tidak ada data duplikat

# # Exploratory Data Analysis

# ## Univariate Analysis

# In[17]:


features = num.columns


for i in range(0, len(features)):
    plt.rcParams['figure.figsize'] = (30,70)
    sns.set(font_scale=2)
    plt.subplot(10,5,i+1)
    sns.distplot(x=df[features[i]], color='Darkblue')
    plt.xlabel(features[i])
    plt.tight_layout()


# In[18]:


# features = ['TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER']
features = ['addr_state', 'sub_grade', 'purpose','emp_length', 'loan_status', 'grade', 'home_ownership','verification_status',            'term', 'initial_list_status', 'pymnt_plan','application_type']

for i in range(0, len(features)):
    plt.rcParams['figure.figsize'] = (20, 40)
    sns.set(font_scale=1)
    plt.subplot(6,2,i+1)
    sns.countplot(x=df[features[i]], color='Steelblue')
    plt.xlabel(features[i], fontsize=20)
    plt.ylabel('count', fontsize=20)
    plt.xticks(rotation = 90)
    plt.tight_layout()


# In[19]:


# features = ['TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER']
features = ['earliest_cr_line', 'last_credit_pull_d', 'next_pymnt_d', 'last_pymnt_d', 'issue_d']

for i in range(0, len(features)):
    plt.rcParams['figure.figsize'] = (20, 40)
    sns.set(font_scale=1)
    plt.subplot(5,1,i+1)
    sns.countplot(x=df[features[i]], color='Steelblue')
    plt.xlabel(features[i], fontsize=20)
    plt.ylabel('count', fontsize=20)
    plt.xticks(rotation = 90)
    plt.tight_layout()


# # Multivariate 

# - Sebelum melakukan Multivariate Analysis terlebih dahulu melakukan feature engineering pada variabel target

# In[20]:


df_train = df.copy()


# Untuk fitur target terlebih dahulu akan ditentukan melalui feature engineering
# 
# Pada variabel 'loan_status' terdapat informasi status pinjaman nasabah saat ini
# - Current : Nasabah telah melakukan pembayaran angsuran 
# - Fully Paid : Nasabah melakukan pelunasan (Asumsi bahwa pelunasan dilakukan karena nasabah memang melakukan pelunasan dengan - status kredit yang baik/lancar/ Kontrak sudah selesai)
# - Charged Off : Status kredit nasabah dihapus bukukan (Bad Customer)
# - Default : Nasabah melakukan wanprestasi/ nasabah gagal melaksanakan kewajiban hutang-pihutang
# - Late (16-30 days) : Nasabah lancar dalam melakukan pembayaran angsuran namun Past Due 16-30 hari 
# - Late (31-120 days) : Nasabah lancar dalam melakukan pembayaran angsuran namun Past Due 31-120 hari
# - Does not meet the credit policy. Status:Fully Paid : Nasabah yang melakukan pelunasan namun saat pengajuan pinjaman, tidak memenuhi suatu persayaratan credit policy namun customer melakukan pelunasan
# - In Grace Period : kelonggaran waktu dalam melakukan pengembalian pinjaman pokok dan atau bunganya selama jangka waktu tertentu (https://www.kamusbesar.com/grace-period)
# - Selain itu pada variabel term akan diubah menjadi numerik data yaitu dengan menghapus string 'year'

# In[21]:


df_train['target'] = np.where(df_train['loan_status']=='Fully Paid',0,
                              np.where(df_train['loan_status']=='Current',0,
                                       np.where(df_train['loan_status']=='Late (16-30 days)',0,
                                               np.where(df_train['loan_status']=='Does not meet the credit policy. Status:Fully Paid',0,1))))
df_train['target'].value_counts()


# In[22]:


from dython.nominal import associations
data = df_train.loc[:, df_train.dtypes!=np.object]
fig, ax = plt.subplots(figsize =(30,20))
sns.set(font_scale=1.2)
plt.text(x=0,y=-0.4,s="Korelasi Variabel Numerik dengan Target",fontsize=20,weight='bold')
# Estimate and generate Cramer's V association plot
cramers_v = associations(data, nom_nom_assoc = 'cramer',cmap = "vlag_r", ax=ax)
plt.show()


# - Kolom/variabel yang memiliki korelasi >0.7 dan mendekati 0.7 (kecuali dengan target) akan didrop, karena terjadi multikolinearitas

# In[23]:


delete = ['collection_recovery_fee','funded_amnt_inv', 'funded_amnt','last_pymnt_amnt', 'loan_amnt','out_prncp_inv',          'installment','total_pymnt','total_pymnt_inv','total_rec_int', 'revol_bal','policy_code','open_acc','mths_since_last_record']
df_train = df_train.drop(delete, axis=1)


# In[24]:


from dython.nominal import associations
data = df_train.loc[:, df_train.dtypes!=np.object]
fig, ax = plt.subplots(figsize =(20,10))
sns.set(font_scale=1)
plt.text(x=0,y=-0.4,s="Korelasi Variabel Numerik dengan Target setelah handling multikolinearity",fontsize=15,weight='bold')
# Estimate and generate Cramer's V association plot
cramers_v = associations(data, nom_nom_assoc = 'cramer',cmap = "vlag_r", ax=ax)
plt.show()


# In[25]:


from dython.nominal import associations
data = df_train.loc[:, df_train.dtypes==np.object]
data['target'] = df_train['target']
fig, ax = plt.subplots(figsize =(20,10))
sns.set(font_scale=1.2)
plt.text(x=0,y=-0.4,s="Korelasi Variabel Kategorik dan Target",fontsize=20,weight='bold')
# Estimate and generate Cramer's V association plot
cramers_v = associations(data, nom_nom_assoc = 'cramer',cmap = "vlag_r", ax=ax)
plt.show()


# - Variabel grade dan subgrade memiliki hubungan multikolinearitas, sehingga akan didrop salah satu. Dalam hal ini variabel sub_grade meskipun memiliki korelasi dengan target lebih rendah dibandingkan grade. Namun cardinalitynya sangat tinggi
# - Variabel application_type juga akan didrop/dihapus karena tidak memiliki hubungan sama sekali dengan semua variabel kategorik
# - Loan Status nantinya akan didrop karena variabel target merupakan fitur extraction dari variabel loan status

# In[26]:


df_train = df_train.drop(['application_type', 'sub_grade','loan_status'], axis=1)


# In[27]:


from dython.nominal import associations
data = df_train.loc[:, df_train.dtypes==np.object]
data['target'] = df_train['target']
fig, ax = plt.subplots(figsize =(20,10))
sns.set(font_scale=1.2)
plt.text(x=0,y=-0.4,s="Korelasi Variabel Kategorik dan Target setelah handling multikolinearity",fontsize=15,weight='bold')
# Estimate and generate Cramer's V association plot
cramers_v = associations(data, nom_nom_assoc = 'cramer',cmap = "vlag_r", ax=ax)
plt.show()


# In[28]:


from dython.nominal import associations
data = df_train
fig, ax = plt.subplots(figsize =(30,20))
sns.set(font_scale=1)
plt.text(x=0,y=-0.4,s="Korelasi antar Variabel",fontsize=20,weight='bold')
# Estimate and generate Cramer's V association plot
cramers_v = associations(data, nom_nom_assoc = 'cramer',cmap = "vlag_r", ax=ax)
plt.show()


# - Akan dilakukan drop pada variabel grade karena terjadi multikolinearity dengan int_rate, variabel grade memiliki korelasi yang lebih rendah dengan target dibandingkan int_rate
# - Variabel last_pymnt_d dan next_pymnt_d juga akan didrop karena memiliki multikolinearity dengan out_prncp

# In[29]:


df_train = df_train.drop(['grade','last_pymnt_d','next_pymnt_d'], axis=1)


# In[30]:


from dython.nominal import associations
data = df_train
fig, ax = plt.subplots(figsize =(30,20))
sns.set(font_scale=1)
plt.text(x=0,y=-0.4,s="Korelasi antar variabel Variabel",fontsize=15,weight='bold')
# Estimate and generate Cramer's V association plot
cramers_v = associations(data, nom_nom_assoc = 'cramer',cmap = "vlag_r", ax=ax)
plt.show()


# # Data Preprocessing

# ## Handling Missing Value

# In[31]:


var_null = df_train.isnull().sum()/df_train.shape[0]*100
var_null = var_null.reset_index()
var_null.columns = ['Variabel/Kolom', '% NA']
var_null = var_null.sort_values(by='% NA', ascending=False)
var_null = var_null[var_null['% NA']>0]
list = []
fitur = var_null['Variabel/Kolom'].unique()
for i in range(0, len(var_null['Variabel/Kolom'])):
    x = df[fitur[i]].dtypes
    list.append(x)
var_null['Tipe Data'] = list
var_null


# - Variabel 'mths_since_last_major_derog' dan 'mths_since_last_delinq' akan didrop karena memiliki missing value mendekati dan lebih dari 50% 
# - persentase Missing Value < 1% pada variabel/kolom akan didrop
# - Missing Value pada variabel/kolom 'total_rev_hi_lim'/ batas kredit tertinggi, tot_cur_bal dan 'tot_coll_amt' akan diinput dengan median
# - Missing Value pada variabel/kolom 'emp_length' akan diinput dengan '<1 year'

# In[32]:


df_train = df_train.drop(['mths_since_last_major_derog', 'mths_since_last_delinq'], axis=1)
df_train['total_rev_hi_lim'].fillna(df_train['total_rev_hi_lim'].quantile(0.5), inplace=True)
df_train['tot_coll_amt'].fillna(df_train['tot_coll_amt'].quantile(0.5), inplace=True)
df_train['tot_cur_bal'].fillna(df_train['tot_cur_bal'].quantile(0.5), inplace=True)
df_train['emp_length'].fillna('< 1 year', inplace=True)


# In[33]:


df_train = df_train.dropna(how='any',axis=0)


# In[34]:


df_train.isnull().sum()


# In[35]:


print('Total baris sebelum dihapus missing value:', df.shape)
print('Total baris setelah dihapus missing value:', df_train.shape)


# - Sudah tidak ada missing value

# ## Feature Selection

# ### Anova Test

# In[36]:


yes_target = df_train[df_train["target"]==1]
no_target = df_train[df_train["target"]==0]
list=[]
list_kolom = []
fitur = df_train.select_dtypes(exclude=[object]).columns
for i in fitur:
    stat, p= st.f_oneway(yes_target[i], no_target[i])
    list.append(p)
list
print('Hasil Uji Statistik ANOVA :')
for i in range(0,len(fitur)):
    if list[i] > 0.05:
        result =  "Terima H0"
        print(result,fitur[i],'p-value',list[i])
        kolom = fitur[i]
        list_kolom.append(kolom)
    else:
        result =  "Terima H1"
        print(result,fitur[i],'p-value',list[i])
print("")
print('Kolom numerik yang akan didrop berdasarkan uji statistik')
print(list_kolom)


# - Variabel 'acc_now_delinq', 'tot_coll_amt' tidak berpengaruh terhadap target sedemikian sehingga akan didrop
# - Selain itu variabel recoveries akan didrop karena variabel tersebut terkait rencana pembayaran, dan asumsi ketika calon nasabah akan mengajukan pinjaman belum ada rencana pembayaran saat hari itu
# - Selanjutnya, akan dilakukan drop pada variabel numerik yang tidak memiliki pengaruh dengan variabel target

# In[37]:


df_train = df_train.drop(['acc_now_delinq', 'tot_coll_amt','recoveries'], axis=1)


# ### Chi-Square

# In[38]:


from scipy.stats import chi2_contingency
list=[]
categorical = df_train.loc[:, df_train.dtypes==np.object].columns.tolist()

for i in categorical:
    ctab = pd.crosstab(df_train[i], df_train['target'])
    stat, p, dof, expected = chi2_contingency(ctab)
    list.append(p)
list

print()
print('Hasil Uji Statistik Chi-Square :')
for i in range(0,9):
    if list[i] > 0.05:
        result =  "Terima H0"
        print(result,categorical[i],'p-value',list[i])

    else:
        result =  "Terima H1"
        print(result,categorical[i],'p-value',list[i])


# - Semua variabel kategorik memiliki hubungan dengan variabel target

# In[39]:


fitur = df_train.loc[:, df_train.dtypes==np.object].columns.tolist()
list = []
list2= []
list3 = []
for i in range(0, len(fitur)):
    x = df_train[fitur[i]].nunique()
    y = df_train[fitur[i]].unique()
    z = fitur[i]
    list.append(x)
    list2.append(z)
    list3.append(y)
    
var = pd.DataFrame({
        'Fitur': list2,
        'Nunique' : list,
        'Unique' : list3})
var = var.sort_values(by = 'Nunique', ascending=False)
var


# - Variabel addr_state memiliki cardinality yang sangat tinggi oleh karena itu akan didrop
# 
# Beberapa informasi terkait variabel berikut :
# - issue_d merupakan bulan dimana pinjaman tersebut didanai oleh perusahaan. 
# - last_credit_pull_d terakhir perusahaan menge'check' credit history.
# - pymnt_plan merupakan rencana pembayaran terhadap angsuran
# 
# Fitur-fitur tersebut akan didrop dengan alasan bahwa asumsinya ketika ada pengajuan/aplikasi masuk, belum diketahui apakah pengajuan tersebut akan disetujui atau tidak sebelum diaplikasikan ke model.

# In[40]:


df_train = df_train.drop(['addr_state','issue_d', 'last_credit_pull_d', 'pymnt_plan'], axis=1)


# ## Handling Outlier

# In[41]:


features = df_train.loc[:, df_train.dtypes!=np.object].columns.tolist()
features.remove('target')

for i in range(0, len(features)):
    plt.rcParams['figure.figsize'] = (20,10)
    sns.set(font_scale=1.2)
    plt.subplot(2,9,i+1)
    sns.boxplot(y=df_train[features[i]], color='Steelblue')
    plt.ylabel(features[i])
    plt.xticks(rotation = 45)
    plt.tight_layout()


# - Beberapa variabel memiliki outlier, sebelum dilakukan drop terlebih dahulu akan dilakukan cek apakah outlier tersebut bertipe 'Global Outlier'. Jika outlier bertipe Global Outlier maka akan dilakukan drop

# In[42]:


import statsmodels.api as sm
features = df_train.loc[:, df_train.dtypes!=np.object].columns.tolist()
features.remove('target')

fig, axes = plt.subplots(ncols=3, nrows=5, sharex=True, figsize=(20, 20))
for i, ax in zip(features, np.ravel(axes)):
    sm.qqplot(df_train[i], line='s', ax=ax)
    ax.set_title(f'{i} QQ Plot')
    plt.tight_layout()


# In[43]:


train = df_train.copy()


# In[44]:


train = train[train['annual_inc']<2500000]
train = train[train['delinq_2yrs']<25]
train = train[train['pub_rec']<20]
train = train[train['revol_util']<175]
train = train[train['total_acc']<140]
train = train[train['total_rec_late_fee']<300]
train = train[train['collections_12_mths_ex_med']<5]
train = train[train['tot_cur_bal']<4500000]
train = train[train['total_rev_hi_lim']<1200000]


# In[45]:


import statsmodels.api as sm
features = train.loc[:, train.dtypes!=np.object].columns.tolist()
features.remove('target')

fig, axes = plt.subplots(ncols=3, nrows=5, sharex=True, figsize=(20, 20))
for i, ax in zip(features, np.ravel(axes)):
    sm.qqplot(train[i], line='s', ax=ax)
    ax.set_title(f'{i} QQ Plot')
    plt.tight_layout()


# ## Feature Engineering

# - Akan dilakukan feature engineering dengan tujuan menyederhanakan beberapa tipe variabel pada variabel kategorik dan melakukan extraction pada variabel bertipe datetime

# In[46]:


fitur = train.loc[:, train.dtypes==np.object].columns.tolist()
list = []
list2= []
list3 = []
for i in range(0, len(fitur)):
    x = train[fitur[i]].nunique()
    y = train[fitur[i]].unique()
    z = fitur[i]
    list.append(x)
    list2.append(z)
    list3.append(y)
    
var = pd.DataFrame({
        'Fitur': list2,
        'Nunique' : list,
        'Unique' : list3})
var = var.sort_values(by = 'Nunique', ascending=False)
var


# - Pada variabel home_ownership terdapat beberapa kategori yaitu Rent, Own, Mortgage, None, Any dan Other. Pada kategori/kelompok None dan Any akan dijadikan satu kategori dengan 'Other'

# In[47]:


train['home_ownership'] = np.where(train['home_ownership']=='RENT', 'RENT',
                                  np.where(train['home_ownership']=='OWN', 'OWN',
                                          np.where(train['home_ownership']=='MORTGAGE','MORTGAGE', 'OTHER')))


# - Pada variabel/kolom purpose akan direduce menjadi 2 kategori yaitu purpose untuk produktif dan multiguna. Dimana purpose credit yang masih terkait dengan usaha/ bisnis akan masuk pada kategori 'produktif' dan selain untuk kegiatan usaha/ bisnis akan masuk pada kategori 'multiguna'

# In[48]:


train['purpose'] = np.where(train['purpose']=='small_business', 'produktif',
                           np.where(train['purpose']=='renewable_energy', 'produktif', 'multiguna'))


# - Pada variabel/kolom 'term' yang merupakan tenor calon nasabah mengajukan pinjaman sebelumnya memiliki unique value '36 months' dan '60 months' akan diubah menjadi format tahun. 

# In[49]:


train['term'] = np.where(train['term']==' 36 months', 3, 5)


# - Pada variabel/kolom 'earliest_cr_line' yang merupakan informasi dimana nasabah pertama kali melakukan pengajuan credit akan dilakukan extraction yaitu dengan mengambil informasi tahunnya 

# In[50]:


train = train.reset_index()
train = train.drop('index', axis=1)


# In[51]:


from datetime import datetime
list = []
fitur = train.shape[0]
for i in range(0, fitur):
    x = datetime.strptime(train['earliest_cr_line'][i], '%b-%y').strftime('%b-%Y')
    list.append(x)


# In[52]:


train['earliest_date'] = list
train['earliest_date'] = pd.to_datetime(train['earliest_date'])


# In[53]:


train['earliest_year'] = datetime.today().year - train['earliest_date'].dt.year -                         (datetime.today().month < train['earliest_date'].dt.month)


# In[54]:


train = train.drop('earliest_date', axis=1)


# - Setelah dilakukan feature extraction pada variabel earliest_cr_line, maka variabel tersebut akan didrop

# In[55]:


train = train.drop('earliest_cr_line', axis=1)


# In[56]:


train = train[train['earliest_year']>=0]


# In[57]:


train.head()


# In[58]:


train = train.reset_index()
train = train.drop('index', axis=1)
train.shape[0]


# In[59]:


train.duplicated().sum()


# # Top Insight

# In[60]:


train.describe()


# In[61]:


train.loc[:, train.dtypes=='object'].describe()


# In[62]:


df_bi = train.copy()


# In[63]:


df_bi['out_prncp_bin'] = np.where(train['out_prncp']<=5000, '0-5000',
                                       np.where(train['out_prncp']<10000,'5001-10000',
                                           np.where(train['out_prncp']<=15000,'10001-15000',
                                                np.where(train['out_prncp']<=20000,'15001-20000',
                                                        np.where(train['out_prncp']<=25000,'20001-25000',
                                                                 np.where(train['out_prncp']<=30000,'25001-30000','>30000'))))))


# In[64]:


df_bi['total_rec_prncp_bin'] = np.where(train['total_rec_prncp']<=5000, '0-5.000',
                                       np.where(train['total_rec_prncp']<10000,'5.001-10.000',
                                           np.where(train['total_rec_prncp']<=15000,'10.001-15.000',
                                                np.where(train['total_rec_prncp']<=20000,'15.001-20.000',
                                                        np.where(train['total_rec_prncp']<=25000,'20.001-25.000',
                                                                 np.where(train['total_rec_prncp']<=30000,'25.001-30.000','>30.000'))))))


# In[65]:


df_bi['int_rate_bin'] = np.where(train['int_rate']<4, '0-3%',
                                 np.where(train['int_rate']<7, '4%-6%',
                                          np.where(train['int_rate']<10, '7%-9%',
                                                   np.where(train['int_rate']<13, '10%-12%',
                                                            np.where(train['int_rate']<16, '13%-15%',
                                                                     np.where(train['int_rate']<19, '16%-18%',
                                                                              np.where(train['int_rate']<22, '19%-21%',
                                                                                       np.where(train['int_rate']<25, '22%-24%','>25%'))))))))


# In[66]:


df_rate = df_bi[['total_rec_prncp_bin','target']]
df_rate['target'] = np.where(df_rate['target']==1,'Difficulties Payment','No Difficulties Payment')

cross_tab_prop = pd.crosstab(index=df_rate['total_rec_prncp_bin'],
                             columns=df_rate['target'],
                             normalize="index")
cross_tab_prop = cross_tab_prop[['No Difficulties Payment','Difficulties Payment']]

#Agar Kolom Kategorikal terurut
cross_tab_prop['sort_bin'] = np.where(cross_tab_prop.index=='0-5.000',1,
                              np.where(cross_tab_prop.index=='5.001-10.000',2,
                                      np.where(cross_tab_prop.index=='10.001-15.000',3,
                                              np.where(cross_tab_prop.index=='15.001-20.000',4,
                                                      np.where(cross_tab_prop.index=='20.001-25.000',5,
                                                              np.where(cross_tab_prop.index=='25.001-30.000',6,7))))))
                                     
cross_tab_prop = cross_tab_prop.sort_values(by='sort_bin', ascending=True)
cross_tab_prop = cross_tab_prop.drop('sort_bin', axis=1)
cross_tab_prop

cross_tab_prop.plot(kind='bar', 
                    stacked=True, 
                    color=['Steelblue','Indianred'],
                    figsize=(20, 12))

plt.legend(loc="upper center",ncol=5,title='Target', fontsize = 15)
sns.set(font_scale=1.5)
plt.axhline(y = df_bi['target'].value_counts(True)[0], color ="Green", linestyle ="--")
plt.text(x=-0.5,y=1.42,s="Smoothness payment rate analysis based on Total Principal received",fontsize=25,weight='bold')
plt.text(x=0.27,y=cross_tab_prop['No Difficulties Payment'].mean()-0.13,s='Average Smoothness \npayment rate',fontsize=15,weight='bold')
plt.xlabel("Total Principal received",fontsize = 20,weight='bold')
plt.ylabel("Proportion",fontsize = 20,weight='bold')
plt.ylim(0,1.4)
plt.xticks(rotation = 0)
plt.show()


# - Semakin banyak total principal received/ total pokok hutang yang telah dibayarkan customer, proporsi keberhasilan pembayarannya juga cenderung tinggi dibandingkan total pokok hutang yang nominalnya lebih kecil. 

# In[67]:


df_rate = df_bi[['int_rate_bin','target']]
df_rate['target'] = np.where(df_rate['target']==1,'Difficulties Payment','No Difficulties Payment')

cross_tab_prop = pd.crosstab(index=df_rate['int_rate_bin'],
                             columns=df_rate['target'],
                             normalize="index")
cross_tab_prop = cross_tab_prop[['No Difficulties Payment','Difficulties Payment']]

#Agar Kolom Kategorikal terurut
cross_tab_prop['sort_bin'] = np.where(cross_tab_prop.index=='0-3%',1,
                              np.where(cross_tab_prop.index=='4%-6%',2,
                                      np.where(cross_tab_prop.index=='7%-9%',3,
                                              np.where(cross_tab_prop.index=='10%-12%',4,
                                                      np.where(cross_tab_prop.index=='13%-15%',5,
                                                              np.where(cross_tab_prop.index=='16%-18%',6,
                                                                       np.where(cross_tab_prop.index=='19%-21%',7,
                                                                                np.where(cross_tab_prop.index=='22%-24%',8,9))))))))
                                     
cross_tab_prop = cross_tab_prop.sort_values(by='sort_bin', ascending=True)
cross_tab_prop = cross_tab_prop.drop('sort_bin', axis=1)
cross_tab_prop

cross_tab_prop.plot(kind='bar', 
                    stacked=True, 
                    color=['Steelblue','Indianred'],
                    figsize=(20, 12))

plt.legend(loc="upper center",ncol=5,title='Target', fontsize = 15)
sns.set(font_scale=1.5)
plt.axhline(y = df_bi['target'].value_counts(True)[0], color ="Green", linestyle ="--")
plt.text(x=-0.5,y=1.42,s="Smoothness payment rate analysis based on Interest Rate",fontsize=25,weight='bold')
plt.text(x=0.27,y=cross_tab_prop['No Difficulties Payment'].mean()-0.05,s='Average Smoothness \npayment rate',fontsize=12,weight='bold')
plt.xlabel("Interest Rate",fontsize = 20,weight='bold')
plt.ylabel("Proportion",fontsize = 20,weight='bold')
plt.ylim(0,1.4)
plt.xticks(rotation = 0)
plt.show()


# - Semakin besar interest rate atau bunga pinjaman, customer cenderung mengalami kesulitan bayar. Terlihat dari proporsi keberhasilan pembayaran. Dimana bunga pinjaman lebih dari 22%, proporsi customer mengalami kesulitan bayar lebih dari 20%.

# In[68]:


df_rate = df_bi[['purpose','target']]
df_rate['target'] = np.where(df_rate['target']==1,'Difficulties Payment','No Difficulties Payment')

cross_tab_prop = pd.crosstab(index=df_rate['purpose'],
                             columns=df_rate['target'],
                             normalize="index")
cross_tab_prop = cross_tab_prop[['No Difficulties Payment','Difficulties Payment']]


cross_tab_prop.plot(kind='bar', 
                    stacked=True, 
                    color=['Steelblue','Indianred'],
                    figsize=(10, 8))

plt.legend(loc="upper center",ncol=5,title='Target', fontsize = 15)
sns.set(font_scale=1.5)
plt.axhline(y = df_bi['target'].value_counts(True)[0], color ="Green", linestyle ="--")
plt.text(x=-0.5,y=1.42,s="Smoothness payment rate analysis based on Purpose",fontsize=20,weight='bold')
plt.text(x=0.27,y=cross_tab_prop['No Difficulties Payment'].mean()-0.05,s='Average Smoothness \npayment rate',fontsize=12,weight='bold')
plt.xlabel("Purpose",fontsize = 20,weight='bold')
plt.ylabel("Proportion",fontsize = 20,weight='bold')
plt.ylim(0,1.4)
plt.xticks(rotation = 0)
plt.show()


# - Customer yang melakukan pengajuan pinjaman dana dengan tujuan produktif, proporsi mengalami kesulitan bayar cenderung lebih tinggi dibandingkan dengan tujuan multiguna (kebutuhan konsumtif). 
# - Oleh karena itu perlu dilakukan analisis terhadap jenis bisnis customer, agar diketahui jenis bisnis seperti apa yang memiliki potensi  mengakibatkan customer mengalami kesulitan pembayaran credit.

# ## Split Data

# In[69]:


X = train.drop('target', axis=1).copy()
y = train['target']


# In[70]:


from sklearn.model_selection import train_test_split 


# In[71]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.2, random_state=42)


# In[72]:


X_train_ori, X_test_ori, y_train_ori, y_test_ori = train_test_split(X, y, stratify = y, test_size=0.2, random_state=42)


# ## Feature Encoding

# ### **Strategi encoding**
# 
# - Label/Ordinal Encoding : Feature yang memiliki 2 kategori dan feature>2 kategori namun dapat diurutkan
# - One Hot Encoding : Feature yang memiliki >2 kategori namun tidak dapat diurutkan
# - StandardScaler : Semua feature numerik, meskipun variabel/feature dti sebarannya cenderung mendekati normal tetap akan digunakan StandardScaler

# In[73]:


print(X_train.loc[:, X_train.dtypes!=np.object].columns.tolist())


# In[74]:


ord = ['purpose', 'emp_length', 'initial_list_status']
ohe = ['home_ownership', 'verification_status']
num = ['term', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'revol_util', 'total_acc',        'out_prncp', 'total_rec_prncp', 'total_rec_late_fee', 'collections_12_mths_ex_med', 'tot_cur_bal',        'total_rev_hi_lim', 'earliest_year']


# In[75]:


list_ord = [['multiguna', 'produktif'], 
            ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'],
           ['f','w']]


# In[76]:


#Ordinal Encoder
ordinal_encoder = OrdinalEncoder(categories=list_ord)
ord_pipe = Pipeline([('ordinal_encoder', ordinal_encoder)])

#OneHot Encoder
onehot_encoder = OneHotEncoder(handle_unknown='ignore')
ohe_pipe = Pipeline([('onehot_encoder', onehot_encoder)])

#Standard Scaler
scaler = StandardScaler()
num_pipe = Pipeline([('scaler', scaler)])


# In[77]:


preprocessor = ColumnTransformer([
    ('num_pipe', num_pipe, num),
    ('ord_pipe', ord_pipe, ord),
    ('ohe_pipe', ohe_pipe, ohe)
])


# In[78]:


preprocessor


# In[79]:


all_columns = num + ord + pd.get_dummies(X_train[ohe]).columns.tolist()


# In[80]:


X_train.head()


# In[81]:


X_train_encod = preprocessor.fit_transform(X_train)
X_test_encod = preprocessor.transform(X_test)


# In[82]:


X_train_encoding = pd.DataFrame(X_train_encod, columns=all_columns)
X_train_encoding.head()


# In[83]:


X_train_ori[num+ord+ohe].head()


# In[84]:


X_test_encoding = pd.DataFrame(X_test_encod, columns=all_columns)
X_test_encoding.head()


# ## Handle Class Imbalance
# Untuk handle class imbalance akan digunakan metode Class Weight yang parameternya ada pada model Machine Learning, Class Weight digunakan karena ketika di dalam algoritma modelnya akan memberikan kesempatan lebih kepada kelas minoritas sehingga dapat memberikan penalti yang lebih tinggi kepada kelas minoritas dan algoritma dapat fokus pada pengurangan kesalahan untuk kelas minoritas.<br>
# [Terkait Class Weight Parameter](https://www.analyticsvidhya.com/blog/2020/10/improve-class-imbalance-class-weights/)

# # Modeling

# - Pada case ini berkaitan dengan resiko sebuah pengajuan kredit oleh calon customer, dimana sebelum dilakukan approval kredit maka terlebih dahulu dilakukan analisa kelayakan calon customer tersebut untuk mendapatkan kredit atau tidak dari pihak pemberi kredit 
# - Secara manual biasanya analisa tersebut dilakukan oleh seorang Credit Analyst atau pihak lainnya yang bertanggung jawab dalam menganalisa sebuah pengajuan kredit dan memberikan rekomendasi terkait kondisi pengajuan kredit tersebut 
# - Namun, semakin banyaknya pengajuan/aplikasi yang masuk tentu hal ini akan memakan banyak waktu apabila dilakukan analisa secara manual untuk semua aplikasi yang masuk
# - Oleh karena itu, dengan pendekatan Data Science melalui model pada Machine Learning, kita dapat melakukan prediksi untuk mengetahui cutsomer yang mengalami kesulitan pembayaran dan lancar dalam pembayaran dengan menggunakan probability
# - Metric yang akan digunakan pada model ini adalah AUC_ROC yang bertujuan mengevaluasi kinerja dari model klasifikasi yang digunakan dalam membedakan kelas 1 dan 0, dimana 1 merupakan nasabah yang mengalami kesulitan pembayaran angsuran dan 0 merupakan nasabah yang lancar dalam pembayaran.
# - Selain itu, untuk evalusi performa model akan dilihat nilai dari Koefisien Gini dan KS atau Kolmogorov Smirnov, dimana koefisien Gini dapat digunakan untuk mengukur performa dari hasil model klasifikasi. Semakin tinggi nilai statistik dari KS semakin baik model membedakan antara calon customer yang kesulitan bayar dan lancar dalam pembayaran

# ## Cross Validation

# In[85]:


#modelling with scoring metrics
def cross_validation(model):
    model_name = []
  
    cv_roc_auc_mean = []
    cv_roc_auc_std = []
    training_roc_auc = []

    
    for name, model in models:
    
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ]) 
        model_name.append(name)

        #scoring
        scoring= ['roc_auc']
        
        #cross_validate 
        cv_score = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=5, n_jobs = -1)
        
        # training
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_train)

        
        training_roc_auc.append(roc_auc_score(y_train, y_pred))
        cv_roc_auc_mean.append(abs(cv_score['test_roc_auc']).mean())
        cv_roc_auc_std.append(abs(cv_score['test_roc_auc']).std())
   
    return pd.DataFrame({
        'Model': model_name,
        'Training AUC_ROC' : training_roc_auc,
        'CV AUC_ROC (mean)' : cv_roc_auc_mean,
        'CV AUC_ROC (std)' : cv_roc_auc_std,
    })


# In[86]:


#assign model ke dalam variabel
models = [
    ['Logistic Regression', LogisticRegression(class_weight='balanced', random_state=42)],
    ['XGB', XGBClassifier(verbosity=0)],
    ['Decision Tree', DecisionTreeClassifier(class_weight='balanced', random_state=42)],
    ['Random Forest', RandomForestClassifier(class_weight='balanced', random_state=42)],
    ['Naive Bayes', GaussianNB()],
    ['LGBM', LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1)]
]


# In[87]:


get_ipython().run_cell_magic('time', '', 'cv_result = cross_validation(models)\ncv_result')


# In[88]:


cv_result['Gap AUC ROC'] = abs(cv_result['CV AUC_ROC (mean)'] - cv_result['Training AUC_ROC'])
cv_result


# - Model yang digunakan adalah Light Gradient Boosting Machine/ LGBM dikarenakan gap antara score AUC_ROC data training dan cross validation test cenderung lebih kecil dibandingkan model lain
# - Selain itu standard deviasi pada model LGBM adalah yang paling kecil, dengan nilai standard deviasi yang kecil maka performa modelnya cenderung lebih konsisten
# - Untuk selanjutnya akan dilakukan hypertunning parameter

# In[89]:


cv_result.to_excel('cv_result.xlsx')


# ## Hypertunning Parameter

# In[90]:


get_ipython().run_cell_magic('time', '', "#Hyptun Logistic Regression\n\nlgbm = LGBMClassifier(class_weight = 'balanced', random_state=42)\n\npipeline = Pipeline([\n        ('preprocessor', preprocessor),\n        ('algo', lgbm)\n    ])\n\nparam_lgbm = {'algo__boosting_type': ['gbdt', 'dart', 'goss'],\n                'algo__num_leaves': (10, 50),\n                'algo__max_depth': (5, 10),\n                'algo__lambda_l2' : (0, 5),\n                'algo__lambda_l1' : (0, 5),\n                'algo__min_gain_to_split' : (0.001, 0.1),\n                'algo__min_data_in_leaf': (10, 120),\n                'algo__bagging_fraction': (0.5, 1),\n                'algo__feature_fraction': (0.1, 0.8)}\n\n\nrs_lgbm= RandomizedSearchCV(estimator=pipeline, param_distributions=param_lgbm, scoring='roc_auc', \n                               random_state=42, cv=5, n_jobs=-1, verbose=1)\nrs_lgbm.fit(X_train, y_train)\n\nprint(rs_lgbm.best_params_)\nprint(rs_lgbm.score(X_train, y_train), rs_lgbm.best_score_)")


# ## Model Evaluation

# ### Compare model before and after hypertuning parameter ini Data Training

# In[91]:


plt.rcParams['figure.figsize'] = (30,10)
sns.set(font_scale=2)
fig, ax = plt.subplots(1,2)
ax[0].set_title("Data Train before tuning LGBM Model")
ax[1].set_title("Data Train after tuning hyperparameter LGBM Model")

lgbm = LGBMClassifier(class_weight = 'balanced', random_state=42, n_jobs=-1)
lgbm.fit(X_train_encod, y_train)
y_pred_train_lgbm = lgbm.predict(X_train_encod)

lgbm_rs = LGBMClassifier(n_jobs=-1, class_weight = 'balanced', random_state=42, num_leaves=50, min_gain_to_split=0.001, min_data_in_leaf=10, 
                         max_depth=10, lambda_l2=5, lambda_l1=0, feature_fraction=0.8, boosting_type='gbdt', bagging_fraction=1)
lgbm_rs.fit(X_train_encod, y_train)
y_pred_train_lgbm_rs = lgbm_rs.predict(X_train_encod)

print("Score Before Hyperparameter Tuning Use LGBM")
print(metrics.classification_report(y_train,y_pred_train_lgbm))
metrics.ConfusionMatrixDisplay(
confusion_matrix = metrics.confusion_matrix(y_train, y_pred_train_lgbm), 
display_labels = [False, True]).plot(ax=ax[0], cmap='Reds')

print("Score After Hyperparameter Tuning Use LGBM")
print(metrics.classification_report(y_train,y_pred_train_lgbm_rs))
metrics.ConfusionMatrixDisplay(
confusion_matrix = metrics.confusion_matrix(y_train, y_pred_train_lgbm_rs), 
display_labels = [False, True]).plot(ax=ax[1], cmap='Reds');


# ### Model Evaluation

# - Model akan dievaluasi menggunakan nilai AUC, Gini dan KS atau Kolmogorov Smirnov
# 
# [Credit Risk Modelling in Python](https://medium.com/analytics-vidhya/credit-risk-modelling-in-python-3ab4b00f6505) \
# [GINI, CUMULATIVE ACCURACY PROFILE, AUC](https://www.listendata.com/2019/09/gini-cumulative-accuracy-profile-auc.html) \
# [Evaluating classification models with Kolmogorov-Smirnov (KS) test](https://towardsdatascience.com/evaluating-classification-models-with-kolmogorov-smirnov-ks-test-e211025f5573) \
# [SAS : CALCULATING KS STATISTICS](https://www.listendata.com/2016/01/sas-calculating-ks-test.html)

# In[92]:


from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# In[93]:


from scipy.stats import ks_2samp

#Evaluation Before Hypertunning
lgbm.fit(X_train_encoding, y_train)
y_pred_train_lgbm = lgbm.predict(X_train_encoding)
y_train_lgbm_proba = lgbm.predict_proba(X_train_encoding)[:,1]
fpr_, tpr_, thresholds_ = roc_curve(y_train, y_train_lgbm_proba)

roc_auc_value_ = roc_auc_score(y_train, y_train_lgbm_proba).round(4)
gini_value_ = ((2*roc_auc_value_)-1).round(4)
stat_KS_, p_value_ = ks_2samp(y_train, y_train_lgbm_proba)
df_fpr_tpr_before = pd.DataFrame({'FPR':fpr_, 'TPR':tpr_, 'Threshold':thresholds_})


#Evaluation After Hypertunning
lgbm_rs.fit(X_train_encoding, y_train)
y_pred_train_lgbm_rs = lgbm_rs.predict(X_train_encoding)
y_train_lgbm_rs_proba = lgbm_rs.predict_proba(X_train_encoding)[:,1]
fpr, tpr, thresholds = roc_curve(y_train, y_train_lgbm_rs_proba)
df_fpr_tpr_after = pd.DataFrame({'FPR':fpr, 'TPR':tpr, 'Threshold':thresholds})
                                                                 
roc_auc_value = roc_auc_score(y_train, y_train_lgbm_rs_proba).round(4)
gini_value = ((2*roc_auc_value)-1).round(4)
stat_KS, p_value = ks_2samp(y_train, y_train_lgbm_rs_proba)                                 
                              

#Table
tabel_perform_model = pd.DataFrame({'Kategori (Data Test)':['Before Hyperparamater Tunning', 'After Hyperparamater Tunning'],
                             'AUC_ROC' : [roc_auc_value_, roc_auc_value],
                             'Gini' : [gini_value_, gini_value],
                            'KS' : [round(stat_KS_,4), round(stat_KS,4)]})


# In[94]:


tabel_perform_model


# In[95]:


fig, ax = plt.subplots(1,1,figsize=(10,8))
ax.plot()
plt.rcParams['figure.figsize'] = (10,7)
sns.set(font_scale=1.2)
plt.plot(fpr_, tpr_, 'Olive', label='%s AUC = %0.4f, Gini = %0.4f, KS = %0.4f' % ('Before Hypertunning: ', tabel_perform_model['AUC_ROC'][0], tabel_perform_model['Gini'][0], tabel_perform_model['KS'][0]),linewidth=3)
plt.plot(fpr, tpr, 'Steelblue', label='%s AUC = %0.4f, Gini = %0.4f, KS = %0.4f' % ('After Hypertunning: ', tabel_perform_model['AUC_ROC'][1], tabel_perform_model['Gini'][1], tabel_perform_model['KS'][1]),linewidth=3)
plt.plot([0,1], [0,1], 'r--', label='Random Classifier', linewidth=3)
plt.text(x=-0.05,y=1.07,s="ROC Curve Light Gradient Boosting Machine Data Training",fontsize=16,weight='bold')
plt.xlabel('FPR',weight='bold',fontsize=14)
plt.ylabel('TPR',weight='bold',fontsize=14)
legend_properties = {'weight':'bold'}
plt.legend(prop=legend_properties)
plt.show()


# - Nilai AUC pada model Light GBM setelah dilakukan hypertunning parameter nilainya lebih besar dibandingkan sebelum hypertunning parameter. Meski perbedaannya tidak terlalu signifikan yaitu hanya 0.5%, Gini setelah dilakukan hypertunning parameter nilainya lebih tinggi dibandingkan sebelum hypertunning. Sedangkan untuk performa menggunakan KS tidak ada perbedaan. Oleh karena itu, parameter yang ada pada hypertunning yang akan digunakan untuk proses selanjutnya

# ## Tuning Threshold

# - Akan dilakukan tuning threshold untuk mengetahui threshold optimal pada case ini. Salah satu teknik yang digunakan adalah dengan menggunakan Gmean
# 
# [Tuning Treshold for Imbalanced Data](https://towardsdatascience.com/optimal-threshold-for-imbalanced-classification-5884e870c293)

# In[96]:


# Calculate the G-mean
gmean = np.sqrt(tpr * (1 - fpr))

# Find the optimal threshold
index = np.argmax(gmean)
thresholdOpt = round(thresholds[index], ndigits=4)
gmeanOpt = round(gmean[index], ndigits = 4)
fprOpt = round(fpr[index], ndigits = 4)
tprOpt = round(tpr[index], ndigits = 4)
print('Best Threshold: {} with G-Mean: {}'.format(thresholdOpt, gmeanOpt))
print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))


# In[97]:


best_th = df_fpr_tpr_after.round(4)
best_th.drop_duplicates(subset=None, keep='first', inplace=True)
best_th = best_th[(best_th['Threshold']==thresholdOpt) & (best_th['FPR']==fprOpt) & (best_th['TPR']==tprOpt)]
best_th


# In[98]:


threshold = thresholdOpt 
plt.rcParams['figure.figsize'] = (30,10)
sns.set(font_scale=2)
fig, ax = plt.subplots(1,2)
ax[0].set_title("Data Train use LGBM with default Threshold")
ax[1].set_title("Data Train use LGBM after Tuning Threshold")

lgbm.fit(X_train_encod, y_train)
y_pred_train_lgbm_rs = lgbm_rs.predict(X_train_encod)
y_pred_train_thr = np.where(lgbm_rs.predict_proba(X_train_encod)[:,1]>=threshold,1,0)

print("Score LGBM Hyperparameter with default Threshold")
print(metrics.classification_report(y_train,y_pred_train_lgbm_rs))
metrics.ConfusionMatrixDisplay(
confusion_matrix = metrics.confusion_matrix(y_train, y_pred_train_lgbm_rs), 
display_labels = [False, True]).plot(ax=ax[0], cmap='Reds')

print("Score LGBM after Tuning Threshold")
print(metrics.classification_report(y_train,y_pred_train_thr))
confusion_matrix = metrics.confusion_matrix(y_train, y_pred_train_thr)
metrics.ConfusionMatrixDisplay(
confusion_matrix = metrics.confusion_matrix(y_train, y_pred_train_thr), 
display_labels = [False, True]).plot(ax=ax[1], cmap='Reds');


# In[99]:


fig, ax = plt.subplots(1,1,figsize=(10,8))
ax.plot()
plt.rcParams['figure.figsize'] = (10,7)
sns.set(font_scale=1.2)
sns.scatterplot(x=best_th['FPR'],y=best_th['TPR'],s=300,color='r',alpha=1)
plt.plot(fpr, tpr, 'Steelblue', label='%s AUC = %0.4f, Gini = %0.4f, KS = %0.4f' % ('Light GBM: ', tabel_perform_model['AUC_ROC'][1], tabel_perform_model['Gini'][1], tabel_perform_model['KS'][1]),linewidth=3)
plt.plot([0,1], [0,1], 'r--', label='Random Classifier', linewidth=3)
plt.text(x=-0.05,y=1.07,s="ROC Curve Light Gradient Boosting Machine Data Training",fontsize=16,weight='bold')
plt.text(x=best_th['FPR']+0.03,y=best_th['TPR']-0.01,s=f'Optimal Threshold : {thresholdOpt}',fontsize=15,weight='bold')
plt.xlabel('FPR',weight='bold',fontsize=14)
plt.ylabel('TPR',weight='bold',fontsize=14)
legend_properties = {'weight':'bold'}
plt.legend(prop=legend_properties)
plt.show()


# - Threshold optimal yang diperoleh adalah 0.422, sehingga ketika nasabah yang memiliki peluang > 0.422 maka diprediksi nasabah tersebut mengalami kesulitan pembayaran.

# ## Implementasi Model in Data Test

# In[101]:


threshold = thresholdOpt 
plt.rcParams['figure.figsize'] = (10,6)
sns.set(font_scale=2)

lgbm_rs.fit(X_train_encod, y_train)
y_pred_test_lgbm_rs = lgbm_rs.predict(X_test_encod)
y_pred_test_thr_test = np.where(lgbm_rs.predict_proba(X_test_encod)[:,1]>=threshold,1,0)
y_pred_test_proba_test = lgbm_rs.predict_proba(X_test_encod)[:,1]

print("Score LGBM after Tuning Threshold")
print(metrics.classification_report(y_test,y_pred_test_thr_test))
confusion_matrix = metrics.confusion_matrix(y_test, y_pred_test_thr_test)
metrics.ConfusionMatrixDisplay(
confusion_matrix = metrics.confusion_matrix(y_test, y_pred_test_thr_test), 
display_labels = [False, True]).plot(cmap='Reds')
plt.title("Data Test use LGBM with default Threshold");


# - Setelah diimplementasikan ke Data Test dari total 92.918 customer, 16.354 diantaranya diprediksi mengalami kesulitan pembayaran.
# - Nilai AUC pada Data Test 0.940, dan nilai Gini serta KS masih di atas 0.8. Performa model masih cukup baik dalam melakukan prediksi pada Data Test.

# 

# In[102]:


#perform roc_auc dan gini
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, lgbm_rs.predict_proba(X_test_encod)[:,1])
df_fpr_tpr_test = pd.DataFrame({'FPR':fpr_test, 'TPR':tpr_test, 'Threshold':thresholds_test})
roc_auc_value_test = roc_auc_score(y_test, lgbm_rs.predict_proba(X_test_encod)[:,1]).round(4)
gini_value_test = ((2*roc_auc_value_test)-1).round(4)


from scipy.stats import ks_2samp

#perform Kolmogorov-Smirnov test
y_test_lgbm_rs_proba = lgbm_rs.predict_proba(X_test_encoding)[:,1]

stat_KS, p_value = ks_2samp(y_test, y_test_lgbm_rs_proba)


print(f'AUC Evaluation : {roc_auc_value_test}')
print(f'Gini Evaluation : {gini_value_test}')
print(f'KS Evaluation : {stat_KS}')


# In[103]:


best_th = df_fpr_tpr_test.round(4)
best_th.drop_duplicates(subset=None, keep='first', inplace=True)
best_th = best_th[best_th['Threshold']==0.422]
best_th


# In[104]:


fig, ax = plt.subplots(1,1,figsize=(10,8))
ax.plot()
plt.rcParams['figure.figsize'] = (10,7)
sns.set(font_scale=1.2)
plt.plot(fpr_test, tpr_test, 'Steelblue', label='%s AUC = %0.3f, Gini = %0.3f, KS = %0.3f' % ('Model: ', roc_auc_value_test, gini_value_test, stat_KS),linewidth=3)
plt.plot([0,1], [0,1], 'r--', label='Random Classifier', linewidth=3)
plt.text(x=-0.05,y=1.07,s="ROC Curve Light Gradient Boosting Machine Data Test",fontsize=16,weight='bold')
plt.xlabel('FPR',weight='bold',fontsize=14)
plt.ylabel('TPR',weight='bold',fontsize=14)
legend_properties = {'weight':'bold'}
plt.legend(prop=legend_properties)
plt.show()


# - Dari hasil model Light GBM yang diimplementasikan pada data test nilai AUC_ROC = 0.940 dimana kinerja algoritma Light GBM untuk kasus dataset ini dapat memprediksi data test dengan baik dan Gini di atas 0.8. 
# - Selain itu, nilai statistik dari KS di atas 0.8 dimana semakin tinggi nilai statistik dari KS semakin baik model membedakan antara calon customer yang kesulitan bayar dan lancar dalam pembayaran

# ## Feature Importance

# In[105]:


def show_feature_importance(model):
    feat_importances = pd.Series(model.feature_importances_, index=X_test_encoding.columns)
    sns.set(font_scale=1.2)
    ax = feat_importances.nlargest(50).plot(kind='barh', figsize=(8, 10), color = 'Steelblue')
    ax.invert_yaxis()

    plt.xlabel('score')
    plt.ylabel('feature')
    plt.title('feature importance score')


# In[106]:


lgbm_rs.fit(X_train_encod, y_train)
show_feature_importance(lgbm_rs)


# ## Shap Value

# In[107]:


import shap
explainer = shap.TreeExplainer(lgbm_rs)
shap_values = explainer.shap_values(X_test_encoding)
shap.summary_plot(shap_values[1], X_test_encoding) # Summary shap value terhadap label positive


# - Semakin tinggi Total Principal atau pokok hutang yang telah dibayarkan customer, maka semakin besar peluang customer mengalami kesulitan pembayaran
# - Semakin tinggi Outstanding Principal atau sisa pokok hutang customer, maka peluang customer mengalami kesulitan pembayaran
# - Semakin tinggi Interet Rate atau bunga pinjaman, maka semakin besar peluang customer mengalami kesulitan pembayaran

# In[ ]:





print('Hola! Este es el script de entrenamiento del modelo de prediccion de clientes')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

#Cargar Datos
print('Estamos cargando y limpiando tus datos!')
df_cust = pd.read_csv('../telecom_customer_churn.csv', encoding='utf-8')
# Limpieza de Datos
df_cust.columns = df_cust.columns.str.lower().str.replace(' ', '_')
categorical_columns = list(df_cust.dtypes[df_cust.dtypes=='object'].index)
for c in categorical_columns:
    df_cust[c] = df_cust[c].str.lower().str.replace(' ', '_')
null2no = ['multiple_lines', 'internet_type', 'online_security', 'online_backup', 'device_protection_plan',
           'premium_tech_support', 'streaming_tv', 'streaming_movies', 'streaming_music', 'unlimited_data']
df_cust[null2no] = df_cust[null2no].fillna(value='no')
df_cust['avg_monthly_gb_download'].fillna(value=0, inplace=True)
df_cust['avg_monthly_long_distance_charges'].fillna(value=0, inplace=True)
df_cust[['churn_category','churn_reason']] = df_cust[['churn_category','churn_reason']].fillna(value='no_churn')
df = df_cust[df_cust['customer_status'] != 'joined'].copy()
df['customer_status'] = (df['customer_status'] == 'stayed').astype(int)
# Entrenamiento del modelo
print('Estamos entrenando el modelo!')
df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
numeric_vars = ['age', 'number_of_dependents', 'number_of_referrals', 'tenure_in_months',
                'avg_monthly_long_distance_charges', 'avg_monthly_gb_download', 'monthly_charge',
                'total_charges', 'total_refunds', 'total_extra_data_charges', 'total_long_distance_charges',
                'total_revenue']
categoric_vars = ['gender', 'married', 'offer', 'phone_service', 'multiple_lines', 'internet_service', 'internet_type',
                'online_security', 'online_backup', 'device_protection_plan', 'premium_tech_support', 'streaming_tv', 'streaming_movies',
                'streaming_music', 'unlimited_data', 'contract', 'paperless_billing', 'payment_method']
def train_model(df_train, y_train):
    dicc_train = df_train[categoric_vars + numeric_vars].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    x_train = dv.fit_transform(dicc_train)
    model = LogisticRegression(max_iter=5000)
    model.fit(x_train, y_train)
    return dv, model
def predict(df, dv, model):
    dicts = df[categoric_vars + numeric_vars].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred
#Exportar el modelo
dv, model = train_model(df, df['customer_status'])
print(f'{dv} y {model} están siendo guardados!')
archivo_exportado = 'modelo.bin'
with open(archivo_exportado, 'wb') as exportar:
    pickle.dump((dv,model), exportar)
print(f'Archivo exportado como {archivo_exportado}! n.n/')
print('Presiona enter para cerrar el script')
input()

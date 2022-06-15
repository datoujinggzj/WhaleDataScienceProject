import pandas as pd
import numpy as np
import colorama
from colorama import Fore, Style
from tabulate import tabulate

def meta(train,test,missing_values = -1,cols_ignore_missing = []):
    
    df = pd.concat([train,test]).reset_index(drop=True).fillna('未知')
    data = []
    for col in df.columns:
        # 定义role
        if col == 'target':
            role = 'target'
        elif col == 'id':
            role = 'id'
        else:
            role = 'feature'
        
        # 定义category
        if 'ind' in col:
            category = 'individual'
        elif 'car' in col:
            category = 'car'
        elif 'calc' in col:
            category = 'calculated'
        elif 'reg' in col:
            category = 'region'
        else:
            category = 'other'
        
        
        # 定义 level of measurements
        if 'bin' in col or col == 'target':
            level = 'binary'
        elif 'cat' in col[-3:] or col == 'id':
            level = 'nominal'
        elif df[col].dtype == 'float64' and df[col].replace(missing_values,np.nan).max()-df[col].replace(missing_values,np.nan).min() > 1:
            level = 'interval'
        elif df[col].dtype == 'float64' and df[col].replace(missing_values,np.nan).max()-df[col].replace(missing_values,np.nan).min() <= 1:
            level = 'ratio'
        elif df[col].dtype == 'int64':
            level = 'ordinal'
            
        # 定义 data type
        dtype = df[col].dtype
        
        # 定义 unique
        if col == 'id' or df[col].dtype == 'float64':
            uniq = 'Ignore'
        else:
            if col in cols_ignore_missing:
                uniq = df[col].nunique()
            else:
                uniq = df[col].replace({missing_values:np.nan}).nunique()
                
        # 定义 cardinality
        if uniq == 'Ignore':
            cardinality = 'Ignore'
        elif uniq <= 10:
            cardinality = 'Low Cardinality'
        elif uniq <= 30:
            cardinality = 'Medium Cardinality'
        else:
            cardinality = 'High Cardinality'
        
        # 定义 missing
        if col in cols_ignore_missing:
            missing = 0
        else:
            missing = sum(df[col] == missing_values)
            
        # 定义 missing percent
        missing_percent = f'{missing}({round(missing*100/len(df),2)}%)'
        
        # 定义 imputation
        if missing > df.shape[0]*0.4:
            imputation = 'remove'
        elif missing > 0:
            if level == 'binary' or level == 'nominal':
                imputation = ('mode')
            if level == 'ordinal':
                imputation = ('mode','median')
            if level == 'interval' or level == 'ratio':
                imputation = ('mode','median','mean')        
        else:
            imputation = "No Missing"
            
        # 定义 keep
        keep = True
        if col  == 'id' or imputation == 'remove':
            keep = False
        col_dict = {
            'colname': col,
            'role': role,
            'category': category,
            'level': level,
            'dtype': dtype,
            'cardinality': uniq,
            'cardinality_level':cardinality,
            'missing': missing,
            'missing_percent': missing_percent,
            'imputation':imputation,
            'keep': keep,
        }
        data.append(col_dict)
    meta = pd.DataFrame(data, columns=list(col_dict.keys()))
    meta.set_index('colname', inplace=True)
    
    return meta

def data_report(train,test,metadata,verbose = False):
    
    fullset = pd.concat([train,test]).reset_index(drop=True).fillna('未知')
    
    print(f"train总行数：{Fore.RED}{train.shape[0]}{Style.RESET_ALL} | test总行数：{Fore.BLUE}{test.shape[0]}{Style.RESET_ALL}")
    print(f"train总列数：{Fore.RED}{train.shape[1]}{Style.RESET_ALL} | test总列数：{Fore.BLUE}{test.shape[1]}{Style.RESET_ALL}")
    print(f"train总元素数：{train.size}")
    print(f"test总元素数：{test.size}")
    print('-'*50+ f"{Fore.RED}INFO{Style.RESET_ALL}"  + '-'*50)
    print('【train info】')
    train.info(verbose = verbose)
    print('-'*104)
    print('【test info】')
    test.info(verbose = verbose)
    
    if verbose:
    
        print('-'*48 + f"{Fore.RED}SUMMARY{Style.RESET_ALL}" + '-'*48)


        ############ SUMMARY #############
        print('*'*48 + f"{Fore.BLUE} COUNTS {Style.RESET_ALL}" + '*'*48)
        print('【Counts groupby role & level】'.upper())
        role_level_count = pd.DataFrame(
        {
            'count':metadata.groupby(['role','level']).size()
        }
        ).reset_index().sort_values(by = 'count',ascending=False)
        print(tabulate(role_level_count,tablefmt="grid",headers = ['role','level','count']))

        print('【Counts groupby role & category】'.upper())
        role_cate_count = pd.DataFrame(
        {
            'count':metadata.groupby(['role','category']).size()
        }
        ).reset_index().sort_values(by = 'count',ascending=False)
        print(tabulate(role_cate_count,tablefmt="grid",headers = ['role','category','count']))

        print('【Counts groupby role & cardinality_level】'.upper())
        role_cardinality_count = pd.DataFrame(
        {
            'count':metadata.groupby(['role','cardinality_level']).size()
        }
        ).reset_index().sort_values(by = 'count',ascending=False)
        print(tabulate(role_cardinality_count,tablefmt="grid",headers = ['role','cardinality_level','count']))


        print('*'*48 + f"{Fore.BLUE} MISSING {Style.RESET_ALL}" + '*'*48)
        print('【Cols to drop】'.upper())
        for col in metadata[metadata['keep'] == False].index:
            print(f" • {col}")

        print('【Cols to impute using (mode)】'.upper())
        for col in metadata[metadata['imputation'] == ('mode')].index:
            print(f" • {col}")

        print('【Cols to impute using (mode|median)】'.upper())
        for col in metadata[metadata['imputation'] == ('mode','median')].index:
            print(f" • {col}")

        print('【Cols to impute using (mode|median|mean)】'.upper())
        for col in metadata[metadata['imputation'] == ('mode','median','mean')].index:
            print(f" • {col}")

        print('*'*48 + f"{Fore.BLUE} CARDINALITY {Style.RESET_ALL}" + '*'*48)
        print('【Cols with medium cardinality】 ==> '.upper()+f'{Fore.YELLOW}PLEASE TAKE CARE OF USING ONEHOT-ENCODING{Style.RESET_ALL}')
        for col in metadata[metadata['cardinality_level'] == 'Medium Cardinality'].index:
            print(f" • {col}")

        print('【Cols with High cardinality】 ==> '.upper()+f'{Fore.YELLOW}PLEASE APPLY TARGET-ENCODING{Style.RESET_ALL}')
        for col in metadata[metadata['cardinality_level'] == 'High Cardinality'].index:
            print(f" • {Fore.GREEN}{col}{Style.RESET_ALL}")


        print('-'*42 + f"{Fore.RED}DESCRIPTIVE ANALYSIS{Style.RESET_ALL}" + '-'*42)
        conti_descrip = fullset[metadata[metadata['level'].isin(['interval','ratio'])].index].describe()
        print(tabulate(conti_descrip.T,tablefmt="grid",headers = conti_descrip.T.columns))

        print('-'*50 + f"{Fore.RED}META{Style.RESET_ALL}" + '-'*50)
        cols = ['role','category', 'level', 'dtype','cardinality', 'missing_percent','keep']
        print(tabulate(metadata[cols],tablefmt="grid",headers = cols))
import pandas as pd
import numpy as np
import colorama
from colorama import Fore, Style
from tabulate import tabulate

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
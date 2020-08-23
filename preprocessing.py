import datetime, numpy as np, pandas as pd
import scipy.stats as st, sklearn
import matplotlib.pyplot as plt
import traceback
import imp
import string
from hashlib import sha256
import math
from statsmodels.stats.power import tt_ind_solve_power

def try_except_decorator(func):
    """Обертка с обработкой исключений"""
    def wrapper_func(*args,**kwargs):
        try:
            return func(*args, **kwargs)
        except:
            return
    return wrapper_func

# обработка данных
@try_except_decorator
def get_int(x):
    return int(x)

@try_except_decorator
def get_date(x):
    return datetime.datetime.strptime(x[:10], '%Y-%m-%d').date()

@try_except_decorator
def get_trunc_date(x, trunc_type='month', is_str=False):
    """Выделяем нужный фрагмент даты, x - datetime.date
    trunc_type - month, quarter, year
    is_str - строковое представление вывода
    """
    # x -> date, is_str - вернуть ли строковое представление
    if trunc_type == 'month':
        x = datetime.date(x.year, x.month, 1)
    elif trunc_type == 'quarter':
        q_month = (x.month-1)//3 + 1
        x = datetime.date(x.year, q_month, 1)
    elif trunc_type == 'year':
        x = datetime.date(x.year, 1, 1)
    if is_str:
        return str(x)[:7]
    return x

# inspect tools
def inspect_methods(object_):
    # список вызываемых публичных методов инстанса
    return [method_name for method_name in dir(object_)
            if callable(getattr(object_, method_name)) and not method_name.startswith('_')]

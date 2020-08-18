import datetime, numpy as np, pandas as pd
import scipy.stats as st, sklearn
import matplotlib.pyplot as plt
import traceback
import imp
import string
from hashlib import sha256
import math
from statsmodels.stats.power import tt_ind_solve_power

from preprocessing import get_trunc_date

# Стат-тесты
# TODO - добавить проверки типов распределений, z-test, t-test, ранговый тест,
# оптимизировать бутстрап, расчет доверительных интервалов, chi-squared
# залить это в отдельный класс?
def get_bootstrap_func(seq_1, seq_2, func, alpha=5):
    # бутстрапирование выбранной произвольно функции
    stat_list = []
    for j in range(3000):
        seq_boot_1 = np.random.choice(seq_1, len(seq_1), replace=True)
        seq_boot_2 = np.random.choice(seq_2, len(seq_2), replace=True)
        stat_list.append(func(seq_boot_2) - func(seq_boot_1))
    return np.round(np.percentile(stat_list, [alpha/2, 100-alpha/2]), 3)

# Аномалии
# TODO - подредактировать, оформить в класс
def outliers_percentile_filter(list_value, percentile_1, percentile_2):
    """Фильтруем выбросы за "усами" боксплотов"""
    list_value = np.array(list_value)
    list_value_dropna = list_value[~np.isnan(list_value)] # dropna
    q1, q3= np.percentile(list_value_dropna,[percentile_1, percentile_2]) # перцентили боксплота
    iqr = abs(q3 - q1)
    lower_bound = q1 -(1.5 * iqr)
    upper_bound = q3 +(1.5 * iqr)
    # Выбросы заменяем значением None
    return [replace_if(j, lower_bound, upper_bound) for j in list_value]
def replace_if(value, bound_1, bound_2):
    if (bound_1 < value) and (value < bound_2):
        return value

def winsorize_filter(list_value, percentile_1, percentile_2):
    """winsorization = замена экстремальных выбросов границами для экстремумов"""
    list_value = np.array(list_value)
    return st.mstats.winsorize(list_value, (percentile_1/1e2, 1 - percentile_2/1e2))


# Верхнеуровневые тулзы
# TODO -
def get_k_means(input_data, cl_number_list = [5]):
    """Кластеризация без учителя методом К-средних
    input_data - pd.DataFrame или list - по группе значений либо одной колонке
    cl_number_list - список из кол-ва кластеров. Если len(cl_number_list) > 1
    строим и выводим зависимость инерции кластеризации от кол-ва для метода локтя
    """

    cl_inertia_list = []
    for n_clusters in cl_number_list:
        kmeans = KMeans(n_clusters=n_clusters).fit(input_data)
        cl_inertia_list.append(kmeans.inertia_**(1/2))

    if len(cl_number_list) > 1:
        plt.bar(x=cl_number_list, height=cl_inertia_list)
        plt.xlabel('Кол-во кластеров')
        plt.ylabel('Корень из суммы квадратов отклонений \n точек от центров своих кластеров')
        plt.title('Определение числа кластеров')
        plt.xticks(cl_number_list)
        plt.grid()
    else:
        predict_list = kmeans.predict(input_data)
        # TODO сортировка результата в зависимости от наполненности, среднего итд

    return predict_list

# Расчет Retention-N бина
def get_cohort_retention_N_bin_dist(df_ext, cohort_type='month', bin_size=30, bin_max=10):
    """
    Расчет ретеншн для N-го бина времени относительно первой покупки участников выбранной когорты
    df_ext - входной cash_flow для выбранного сегмента -> user_id, action_time (datetime.date)
    cohort_type - month, quarter, year
    bin_size - кол-во дней в одном бине (бины нумеруются от 0 - текущий бин)
    bin_max - макс. кол-во бинов с начала когорты (порог)
    """
    # инициализируем копию
    df = df_ext[['user_id', 'action_time']].copy()

    # первые взаимодействия и когорты
    d_first_actions = df.sort_values(by='action_time').drop_duplicates(subset='user_id')
    d_first_actions.rename(columns={'action_time' : 'first_action_time'}, inplace=True)
    d_first_actions['cohort_time'] = d_first_actions.first_action_time.apply(get_trunc_date, args=(cohort_type,))
    d_first_actions['cohort_num'] = d_first_actions.cohort_time.rank(method='dense').astype(int)

    # бинаризация
    df = df.merge(d_first_actions, on='user_id')
    df['bin_num'] = (df['action_time'] - df['first_action_time']).apply(lambda x: x.days // bin_size)

    # убираем далекие бины
    if bin_max:
        df = df[df.bin_num <= bin_max]

    # ускоряем работу в цикле
    df.index = df.cohort_num.values
    cohort_map = df[['cohort_time']].drop_duplicates().to_dict()['cohort_time']

    # Контейнер результата
    res = pd.DataFrame(None)

    # выделяем когорты
    cohort_num_list = sorted(d_first_actions.cohort_num.unique())
    for cohort_num in cohort_num_list:

        # собираем d_cohort = [bin_num, cohort_time, cohort_num, N_employers, retention_percent]
        tmp = df.loc[cohort_num, :]
        cohort_users_count = tmp.user_id.unique().shape[0]
        d_cohort = tmp.groupby('bin_num').user_id.nunique().reset_index(name='N_users')
        d_cohort['cohort_time'] = cohort_map[cohort_num]
        d_cohort['cohort_num'] = cohort_num
        d_cohort['retention_percent'] = 100 * d_cohort['N_users'] / cohort_users_count
        res = pd.concat([res, d_cohort], ignore_index=True)

    return res


# Визуализации
# TODO - boxplots, heatmap, ...
def get_percentile_curve(val_list, per_start, per_end, per_step=5, plot_style='-o'):
    """Перцентильная кривая для выборки val_list"""
    per_range = list(range(per_start, per_end + per_step, per_step))
    per_list = np.percentile(val_list, per_range)
    plt.plot(per_range, per_list, plot_style)
    plt.xticks(per_range)
    plt.grid()

#

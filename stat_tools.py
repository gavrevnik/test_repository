import datetime, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import traceback
import imp
import string
from hashlib import sha256
import math

import scipy.stats as st, sklearn
from statsmodels.stats.power import tt_ind_solve_power
from scipy.stats import mannwhitneyu, sem, t, kstest, ks_2samp, chi2_contingency, chisquare
from statsmodels.stats.weightstats import ztest, ttest_ind
from statsmodels.stats.proportion import proportion_confint

# from preprocessing import get_trunc_date

class Stat_tests:

    def __init__(self, alpha=0.05, power=0.8, alternative='two-sided'):
        # Ошибка первого рода (стат-значимость, доля ложного срабатывания)
        self._alpha = alpha
        # Мощность критерия (1 - ошибка 2го рода) TODO
        self._power = power
        # Постановка альтернативной гипотезы
        # two-sided - проверяем что a != b
        # larger/smaller - проверяем что a>b или a<b
        self._alternative = alternative

    def get_init_params(self):
        return {'alpha' : self._alpha,
                'power' : self._power,
                'alternative' : self._alternative}

    def _get_decision(self, p_value, T):
        """Принятие решения на основе данных стат-теста"""

        if self._alternative == 'two-sided':
            if p_value >= self._alpha:
                decision = 'var_1 = var_2'
            else:
                decision = 'var_1 != var_2'

        if self._alternative == 'larger':
            if p_value >= self._alpha:
                decision = 'var_1 not-larger var_2'
            else:
                decision = 'var_1 > var_2'

        if self._alternative == 'smaller':
            if p_value >= self._alpha:
                decision = 'var_1 not-smaller var_2'
            else:
                decision = 'var_1 < var_2'
        return {'Score' : T, 'p_value' : round(p_value, 4), 'decision' : decision}

    # ПАРАМЕТРИЧЕСКИЕ И РАНГОВЫЕ ТЕСТЫ
    def t_test(self, val_list_1, val_list_2):
        # для независимых выборок, распределение статистики по Стьюденту,
        # подходит для малых выборок
        T, p_value, _ = ttest_ind(val_list_1, val_list_2, alternative=self._alternative)
        return self._get_decision(p_value, T)

    def z_test(self, val_list_1, val_list_2):
        # предельный случай t-test'a, распределение статистики нормальное
        # n > 30, n -> inf
        T, p_value = ztest(val_list_1, val_list_2, alternative=self._alternative)
        return self._get_decision(p_value, T)

    def mannwhitneyu_test(self, val_list_1, val_list_2):
        # ранговый критерий, он же критерий Уилкоксона для независимых выборок
        alternative = self._alternative.replace('smaller', 'less').replace('larger', 'greater')
        T, p_value = mannwhitneyu(val_list_1, val_list_2, alternative=alternative)
        return self._get_decision(p_value, T)

    def bootstrap_test(self, val_list_1, val_list_2, func, n_iter = 1e4, alpha=5):
        # бутстрап произвольной статистики
        # TODO - тоже подумать про критерии теста
        stat_list = []
        for _ in range(n_iter):
            seq_boot_1 = np.random.choice(val_list_1, len(val_list_1), replace=True)
            seq_boot_2 = np.random.choice(val_list_2, len(val_list_2), replace=True)
            stat_list.append(func(seq_boot_2) - func(seq_boot_1))
        return np.round(np.percentile(stat_list, [alpha/2, 100-alpha/2]), 3)

    # КРИТЕРИИИ СОГЛАСИЯ
    def ks_test(self, data_1, data_2, args=()):
        """Колмогорова-Смирнова тест для сравнения выборочного распределения с
        аналитическим или выборочным распределением
        data_1 - выборка #1
        data_2 - выборка #2 либо аналитическое распределение (string)
        args - параметры аналитического распределения (loc, scale, ...)
        При неизвестных параметрах распределения лучше использовать частотный
        критерий Пирсона
        """
        alternative = self._alternative.replace('smaller', 'less').replace('larger', 'greater')
        if type(data_2) == type('str'):
            T, p_value = kstest(data_1, data_2, args=args, alternative=alternative)
        else: # сравнение двух выборочных распределений
            T, p_value = kstest(data_1, data_2, alternative=alternative)
        return self._get_decision(p_value, T)

    def chi2_test(self, obs_freq, exp_freq=None):
        """Критерий Пирсона о согласии частот/долей для категориальных переменных в разных группах
        obs_freq - одномерное или n-мерное распределение частот (таблица сопряженности)
        exp_freq - ожидаемое одномерное распределение частот (по умолчанию равномерное)
        """
        alternative = self._alternative.replace('smaller', 'less').replace('larger', 'greater')
        if len(np.shape(obs_freq)) > 1:
            T, p_value, *_ = chi2_contingency(obs_freq)
        else:
            T, p_value = chisquare(obs_freq, exp_freq)
        return self._get_decision(p_value, T)

    # ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ
    def conf_interval_mean(self, val_list):
        """Доверительный интервал для среднего по выборке
        Выборочное среднее при n->inf распределено нормально"""
        confidence = 1 - self._alpha
        a = 1.0 * np.array(val_list); n = len(a)
        m, se = np.mean(a), sem(a)
        h = se * t.ppf((1 + confidence) / 2., n-1)
        return m-h, m+h

    def conf_interval_bootstrap_agg(self, val_list, agg, n_iter=1e3, show_dist=False):
        """Доверительный интервал для произвольного выборочного agg(val_list)
        show_dist позволяет отобразить распределение выборочной статистики"""
        stat_list = []; alpha = 100 * self._alpha
        for _ in range(n_iter):
            val_list_boot = np.random.choice(val_list, len(val_list), replace=True)
            stat_list.append(agg(val_list_boot))
        if show_dist:
            pd.DataFrame({'agg' : stat_list})['agg'].hist(bins=50)
            plt.title('agg - dist')
        return tuple(np.percentile(stat_list, [alpha/2, 100-alpha/2]))

    def conf_interval_proportion(self, count, nobs):
        """Доверительный интервал для вероятности prob=count/nobs
        построено на биномиальном распределении
        в пределе сходится к conf_interval_mean(count * [1] + (nobs-count) * [0])"""
        return proportion_confint(count, nobs, alpa=self.alpha)




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
# lorenz_curve


# добавить вывод value_counts
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
def get_cohort_retention_N_bin_dist(df_ext, cohort_type='month', bin_size=30, bin_max=10, cohort_min_users=500):
    """
    Расчет ретеншн для N-го бина времени относительно первой покупки участников выбранной когорты
    df_ext - входной cash_flow для выбранного сегмента -> user_id, action_time (datetime.date)
    cohort_type - month, quarter, year
    bin_size - кол-во дней в одном бине (бины нумеруются от 0 - текущий бин)
    bin_max - макс. кол-во бинов с начала когорты
    cohort_min_users - мин кол-во юзеров в одной когорте для вычислений
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
        tmp = df.loc[[cohort_num], :]
        cohort_users_count = tmp.user_id.unique().shape[0]
        if cohort_users_count >= cohort_min_users:
            d_cohort = tmp.groupby('bin_num').user_id.nunique().reset_index(name='N_users')
            d_cohort['cohort_time'] = cohort_map[cohort_num]
            d_cohort['cohort_num'] = cohort_num
            d_cohort['retention_percent'] = 100 * d_cohort['N_users'] / cohort_users_count
            res = pd.concat([res, d_cohort], ignore_index=True)

    return res


# Визуализации
def init_matplot(figsize_x=10, figsize_y=5, subplot_grid=None):
    """Преднастройка отображения графиков в plt"""
    matplotlib.rcParams['figure.figsize'] = (figsize_x, figsize_y)
    matplotlib.rcParams['figure.titlesize'] = 20
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['legend.fontsize'] = 20
    matplotlib.rcParams['axes.titlesize'] = 22
    matplotlib.rcParams['axes.labelsize'] = 20
    matplotlib.rcParams['xtick.labelsize'] = 15
    matplotlib.rcParams['ytick.labelsize'] = 15
    matplotlib.rcParams['lines.linewidth'] = 4

    # сетка графиков - на выходе fig, list_ax[n,m]
    # управление через plt.sca(ax)
    if subplot_grid:
        m, n = subplot_grid
        return plt.subplots(m, n)


# TODO - boxplots, heatmap, ...
# stack диаграмма ретеншн
def get_percentile_curve(val_list, per_start, per_end, per_step=5, plot_style='-o'):
    """Перцентильная кривая для выборки val_list"""
    per_range = list(range(per_start, per_end + per_step, per_step))
    per_list = np.percentile(val_list, per_range)
    plt.plot(per_range, per_list, plot_style)
    plt.xticks(per_range)
    plt.grid()

#

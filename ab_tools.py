import datetime, numpy as np, pandas as pd
import scipy.stats as st, sklearn
import matplotlib.pyplot as plt
import traceback
import imp
import string
from hashlib import sha256
import math
from statsmodels.stats.power import tt_ind_solve_power

# АА-тест
def ab_get_uniform_graph(p_values, graph_range = np.arange(0, 0.11, 0.01)):
    """
    АА-тест. Валидация
    Построение зависимости доли ложных прокрасов теста (ошибка 1-го рода)
    от alpha стат-теста
    p_values - список p_value, которые бутстрапом выдавал тест на двух А-выборках
    graph_range - диапазон p_value отображаемый на графике
    """
    r_list = graph_range
    val_list=[len(p_values[p_values < j]) / len(p_values) for j in r_list]
    plt.plot(r_list, val_list, color='r')
    plt.plot(r_list, r_list, color='b')

    plt.xticks(r_list); plt.yticks(r_list)
    plt.grid()
    plt.legend(['real_dist', 'ideal_curve'])
    plt.title('Зависимость реальной ошибки первого рода \n от стат-значимости теста')
    plt.xlabel('alpha, стат-значимость')
    plt.ylabel('Доля ложных прокрашиваний AA/теста')

def ab_get_uniform_hist(title, p_values, bins=50):
    """
    Проверка равномерности распределения p_values для АА-теста
    p_values - список p_value, которые бутстрапом выдавал тест на двух А-выборках
    """
    pd.DataFrame({'p_value' : p_values}).p_value.hist(bins=bins)
    plt.xlabel('p_value')
    plt.ylabel('Кол-во A/A тестов')
    q = round(100 * len(p_values[p_values < 0.05]) / len(p_values), 1)
    r=len(p_values) / bins
    plt.plot([0,1], [r-2*np.sqrt(r), r-2*np.sqrt(r)])
    plt.plot([0,1], [r+2*np.sqrt(r), r+2*np.sqrt(r)])
    plt.title("""{} \n Доля ложных AA-срабатываний при значимости 5%: \n {}% ({} AA-тестов)""".format(title, q, len(p_values)))


# Длительность теста
# TODO Написать развилку с возможностью считать с дисперсией экспериментальной выборки
# TODO - правильно научиться оценивать время эксперимента
def ab_calc_exp_duration(metric_list, users_per_day, alpha=0.05, power=0.8, ratio=1, mde_range_percent=range(5, 30, 5)):
    """Расчет длительности эксперимента по контрольной метрике в заданном диапазоне MDE = minimal detective effect
    metric_list - значения контрольной метрики эксперимента для каждого юзера (агрегированное по дням)
    users_per_day - сколько в среднем пользователей приходило в эксперимент из выборки в день
    alpha, power - показатели ошибок 1го и 2го рода
    ratio - отношение кол-ва пользователей в контроле и эксперименте
    mde_range_percent - диапазон допускаемой разрешающей способности эксперимента
    """
    mean_control = np.mean(metric_list)
    std_control = np.std(metric_list)

    duration_list = []
    for mde in mde_range_percent:
        effect_size = mean_control * (mde / 1e2) / std_control
        n_samples = tt_ind_solve_power(effect_size=effect_size, alpha=alpha,
                                        power=power, ratio=ratio, alternative='larger')
        duration_list.append(math.ceil(n_samples / users_per_day))

    plt.plot(mde_range_percent, duration_list)
    plt.title('Зависимость длительности эксперимента \n от разрешающей способности \n alpha={0}, power={1}'.format(alpha, power))
    plt.xlabel('Минимальное значимое отклонение среднего, %')
    plt.xticks(mde_range_percent)
    plt.ylabel('Кол-во дней эксперимента')
    plt.grid()
    return duration_list

def ab_get_mde(metric_list, n_control, alpha=0.05, power=0.8, ratio=1):
    """Получение актуального mde по произведенному эксперименту"""
    mean_control = np.mean(metric_list)
    std_control = np.std(metric_list)

    effect_size = tt_ind_solve_power(nobs1=n_control, alpha=alpha,
                                    power=power, ratio=ratio, alternative='larger')

    # mde
    return round(100 * effect_size * std_control / mean_control, 1)

# Подвыборки
# получаем по соли слоя принадлежность работодателя к ветке эксперимента
def get_split(identifier, salt, n_splits, AB_HASH_LENGTH = 15):
    id_hash = sha256((str(identifier) + salt).encode('utf-8')).hexdigest()
    split = int(id_hash[:AB_HASH_LENGTH], base=16) % n_splits
    return split
def get_ab_group(employer_id, control_splits, salt, n_splits):
    if get_split(employer_id, salt, n_splits) in control_splits:
        return 'control'
    else:
        return 'experiment'

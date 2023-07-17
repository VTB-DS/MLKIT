# (A) INIT
# (A1) LOAD MODULES
from flask import Flask, render_template, request, session, g, send_file
import pandas as pd
import numpy as np
import os

from vtb_scorekit.data import DataSamples
from vtb_scorekit.model import LogisticRegressionModel
from vtb_scorekit.woe import WOE

# (A2) FLASK SETTINGS + INIT
app = Flask(__name__)
# app.debug = True
app.secret_key = os.urandom(28)

# (B) DEMO - READ CSV & GENERATE HTML TABLEpip freeze | grep Jinja2

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')
 
@app.route('/data',  methods=("POST", "GET"))
def showData():
    if request.method == 'POST':
        data = pd.read_csv(request.files.get('file'))
    else:
        data = pd.read_csv('demo-data.csv')   
    #g.data = data
    pd.to_pickle(data, "data.pkl")
    rows = data.shape[0]
    cols = data.shape[1]-1
    columns = data.columns
    df = data.describe().round(2).T
    df['Missing'] = '0.0'
    #df['Type'] = 'Integer'
    #df['Quality'] = 'Medium'
    df['Role'] = 'Feature'
    df['count'] = round(df['count'],0)
    df = df.reset_index()
    df.rename(columns={'index':'Flied Name','count':'Count','mean':'Mean','min':'Min','max':'Max'}, inplace=True)
    for i in range(len(df)):
        for c in ["Mean", "Min", "Max"]:
            if abs(float(df.loc[i, c])) > 1000000000000:
                df.loc[i, c] = str( round(df.loc[i, c] / 1000000000000 ,1) ) + "T"
            elif abs(float(df.loc[i, c])) > 1000000000:
                df.loc[i, c] = str( round(df.loc[i, c] / 1000000000 ,1) ) + "B"
            elif abs(float(df.loc[i, c])) > 1000000:
                df.loc[i, c] = str( round(df.loc[i, c] / 1000000 ,1) ) + "M"
            elif abs(float(df.loc[i, c])) > 1000:
                df.loc[i, c] = str( round(df.loc[i, c] / 1000 ,1) ) + "K"
        
        df.loc[i, 'Missing'] = round(data[df.loc[i, 'Flied Name']].isna().sum() / data[df.loc[i, 'Flied Name']].count(),2)

        #if abs(float(df.loc[i, 'Missing'])) > 0.2:
        #    df.loc[i, 'Quality'] = 'Low'

        #if (df.loc[i, 'Flied Name'] == 'record_id') or (df.loc[i, 'Flied Name'] == 'PassengerId'):
        #    df.loc[i, 'Role'] = 'ID'

        #if (df.loc[i, 'Flied Name'] == 'default_12m') or (df.loc[i, 'Flied Name'] == 'Survived'):
        #    df.loc[i, 'Role'] = 'Target'

    df = df.drop(columns=['std','25%','50%','75%'])

    df_html = df.to_html()

    return render_template("1_data.html", rows=rows, cols=cols, columns=columns, df_html=df_html, df=df)

@app.route('/sampling', methods=['GET', 'POST'])
def showSampling():
    #data = g.data
    target = request.args.get('target')
    oos_size = float(request.args.get('oos_size',0.2))
    oot_size = float(request.args.get('oot_size',0.0))
    train_size = float(request.args.get('train_size',0.8))
    val_size = float(request.args.get('val_size',0.0))
    bs = request.args.get('bootstrap')
    sf = request.args.get('stratify')

    if sf == '':
        sf = 'False'

    if sf == 'True':
        sf_flag = True
    else:
        sf_flag = False

    if bs == '':
        bs = 'False'

    print(f'SF:{sf_flag}')

    ds = DataSamples(samples={'Train': pd.read_pickle("data.pkl")}, target=target, result_folder='output', samples_split={'test_size':oos_size,'validate_size':val_size,'split_type':'oos','stratify':sf_flag}, bootstrap_split={},verbose=False)
    #ds.train_name = 'Train' 
    #ds.stats()

    ds.samples['Train'].to_csv(f'train.csv')

    try:
        ds.samples['Test'].to_csv(f'test.csv')
    except:
        ds.samples['Train'].head(0).to_csv(f'test.csv')

    try:
        ds.samples['Validate'].to_csv(f'val.csv')
    except:
        ds.samples['Train'].head(0).to_csv(f'val.csv')


    if bs == 'True':
        ds.bootstrap_split(df=pd.read_pickle("data.pkl"),                           # датафрейм, основа для нарезания подвыборок
                   bootstrap_part=1.5,           # размер каждой подвыборки
                   bootstrap_number=100,         # кол-во подвыборок
                   stratify=sf_flag,                # стратификация каждой подвыборки по целевой переменной
                   replace=True,                 # разрешается ли повторять каждое наблюдение множество раз в подвыборке
                  )
        
    samples = pd.DataFrame([[name
               , sample.shape[0]
               , sample[ds.target].sum()
               , sample[ds.target].sum() / sample.shape[0]
               , f'{sample[ds.time_column].min()} - {sample[ds.time_column].max()}' 
               if ds.time_column is not None else 'NA']
              for name, sample in {**ds.samples, **({'Bootstrap base': ds.bootstrap_base} 
                                                      if ds.bootstrap_base is not None else {})}.items()]
             , columns=['', 'amount', 'target', 'target_rate', 'period']).set_index('').T
    
    samples.columns = [x.upper() for x in samples.columns]
    
    if bs == 'False':
        train_s=samples['TRAIN'][0]
        train_tr=samples['TRAIN'][1]
    else:
        train_s=np.array(ds.bootstrap).shape[1]
        train_tr=int(np.array([ds.bootstrap_base.iloc[idx][ds.target].sum() for idx in ds.bootstrap]).mean())

    test_s=samples['TEST'][0]
    test_tr=samples['TEST'][1]

    if samples.shape[1] == 4:
        val_s=samples['VALIDATE'][0]
        val_tr=samples['VALIDATE'][1]
    else:
        val_s=0
        val_tr=0

    oot_s=0
    oot_tr=0

    return render_template("2_sampling.html",stratify=sf,target=target,train_s=train_s,oot_s=oot_s,oot_tr=oot_tr,test_s=test_s,val_s=val_s,val_tr=val_tr,train_tr=train_tr,test_tr=test_tr,oos_size=oos_size,oot_size=oot_size,train_size=train_size,val_size=val_size)


@app.route('/binning',  methods=("POST", "GET"))
def showBinning():
    ds = DataSamples(samples={'train': pd.read_csv('demo-data.csv')}, target='default_12m', result_folder='output', samples_split={}, bootstrap_split={})
    
    return render_template("3_binning.html")

@app.route('/model',  methods=("POST", "GET"))
def showModel():
    ds = DataSamples(samples={'train': pd.read_csv('demo-data.csv')}, target='default_12m', result_folder='output', samples_split={}, bootstrap_split={})
    gini = ds.calc_gini() 
    gini = gini[(gini['train'] != 0)]
    f = gini.index
    corr_mat = ds.corr_mat(sample_name=None, features=f, corr_method='pearson', corr_threshold=0.75, description_df=None, styler=False)
    corr_mat_html = corr_mat.to_html()
    
    return render_template("model.html",corr_mat_html=corr_mat_html)

@app.route('/report',  methods=("POST", "GET"))
def showReport():

    data = pd.read_csv('demo-data.csv')  

    ds = DataSamples(samples={'train': data}, target='default_12m', result_folder='titanic_output', samples_split={}, bootstrap_split={})

    logreg = LogisticRegressionModel(clf=None,        # классификатор модели (должен иметь метод fit() и атрибуты coef_, intercept_). При None выбирается SGDClassifier(alpha=0.001, loss='log', max_iter=100)
                                 ds=ds,               # Привязанный к модели ДатаСэмпл. Если задан, то он по умолчанию будет использоваться во всех методах
                                 transformer=None,    # объект класса WOE для предварительной трансформации факторов
                                 round_digits=3,      # округление коэффициентов до этого кол-ва знаков после запятой   
                                 name='Titanic',      # название модели
                                )   

    gini = ds.calc_gini() 
    gini = gini[(gini['train'] != 0)]
    f = gini.index
    gini_html = gini.to_html()

    corr_mat = ds.corr_mat(sample_name=None, features=f, corr_method='pearson', corr_threshold=0.75, description_df=None, styler=False)
    corr_mat_html = corr_mat.to_html()


    logreg.mfa(ds,                                   # ДатаСэмпл. В случае, если он не содержит трансформированные переменные, то выполняется трансформация трансформером self.transformer. При None берется self.ds
           features=None,                        # исходный список переменных для МФА. При None берутся все переменные, по которым есть активный биннинг
           hold=None,                            # список переменных, которые обязательно должны войти в модель
           features_ini=None,                    # список переменных, с которых стартует процедура отбора. Они могут быть исключены в процессе отбора
           limit_to_add=100,                     # максимальное кол-во переменных, которые могут быть добавлены к модели
           gini_threshold=5,                     # граница по джини для этапа 1
           corr_method='pearson',                # метод расчета корреляций для этапа 2. Доступны варианты 'pearson', 'kendall', 'spearman'
           corr_threshold=0.70,                  # граница по коэффициенту корреляции для этапа 2
           drop_with_most_correlations=False,    # вариант исключения факторов в корреляционном анализе для этапа 2
           drop_corr_iteratively=False,          # исключение коррелирующих факторов не на отдельном этапе 2, а итеративно в процессе этапа 3
                                                 #    (список кандидатов на добавление в модель формируется динамически после каждого шага,
                                                 #    из него исключаются все коррелирующие с уже включенными факторы).
                                                 #    Применимо только для типов отбора forward и stepwise
           selection_type='stepwise',            # тип отбора для этапа 3
           pvalue_threshold=0.05,                # граница по p-value для этапа 3
           pvalue_priority=False,                # вариант определения лучшего фактора для этапа 3
           scoring='gini',                       # максимизируемая метрика для этапа 3
                                                 #     Варианты значений: 'gini', 'AIC', 'BIC' + все метрики доступные для вычисления через sklearn.model_selection.cross_val_score.
                                                 #     Все информационные метрики после вычисления умножаются на -1 для сохранения логики максимизации скора.
           score_delta=0.1,                      # минимальный прирост метрики для этапа 3
           n_stops=1,                            # количество срабатываний нарушений правил отбора по приросту метрики/p-value до завершения этапа 3
           cv=None,                              # параметр cv для вычисления скора sklearn.model_selection.cross_val_score для этапа 3. 
                                                 #     При None берется StratifiedKFold(5, shuffle=True, random_state=self.random_state)
           drop_positive_coefs=True,             # флаг для выполнения этапа 4
           
           # --- Кросс переменные ---
           crosses_simple=True,                  # True  - после трансформации кросс-переменные учавствут в отборе наравне со всеми переменными
                                                 # False - сначала выполняется отбор только на основных переменных,
                                                 #     затем в модель добавляются по тем же правилам кросс переменные, но не более, чем crosses_max_num штук
           crosses_max_num=10,                   # максимальное кол-во кросс переменных в модели. учитывается только при crosses_simple=False
           
           # --- Отчет ---
           verbose=False,                         # флаг для вывода подробных комментариев в процессе работы
           result_file='mfa.xlsx',               # файл, в который будут сохраняться результаты мфа
           metrics=None,                         # список метрик/тестов, результы расчета которых должны быть включены в отчет.
                                                 #     Элементы списка могут иметь значения (не чувствительно к регистру):
                                                 #         'ontime': расчет динамики джини по срезам,
                                                 #         'vif'   : расчет Variance Inflation Factor,
                                                 #         'psi'   : расчет Population Population Stability Index,
                                                 #         'wald'  : тест Вальда,
                                                 #         'ks'    : тест Колмогорова-Смирнова,
                                                 #         func    : пользовательская функция, которая принимает целевую и зависимую переменную,
                                                 #                   и возвращает числовое значение метрики
                                                 #                   Например,
                                                 #                   def custom_metric(y_true, y_pred):
                                                 #                       from sklearn.metrics import roc_curve, f1_score
                                                 #                       fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                                                 #                       thres = thresholds[np.argmax(tpr * (1 - fpr))]
                                                 #                       return f1_score(y_true, (y_pred > thres).astype(int))
                                                 #                   metrics = ['vif', 'ks', 'psi', custom_metric]
           metrics_cv=None                       # список метрик, рассчитываемых через sklearn.model_selection.cross_val_score.
                                                 #     Аналогично параметру metrics элементами могут быть строки, поддерживаемые cross_val_score, либо пользовательские функции
                                                 #     Например, ['roc_auc', 'neg_log_loss', 'gini', 'f1', 'accuracy', custom_metric]
          )


    binning = WOE(ds,                            # ДатаСэмпл, для которого будут рассчитываться биннинги
              features=None,                     # список переменных. При None берется ds.features
              scorecard=None,                    # путь к эксель файлу или датафрейм с готовыми биннингами для импорта
              round_digits=3,                    # округление границ бинов до этого числа знаков после запятой.
                                                 # При округлении происходит проверка на долю мигрирующих наблюдений. Если округление приедет к миграции большой доли наблюдений,
                                                 # то round_digits увеличивается до тех пор, пока доля не упадет ниже rounding_migration_coef
              rounding_migration_coef=0.005,     # максимально допустимая доля наблюдений для миграции между бинами при округлении
              # ---Параметры для расчета WOE---
              simple=True,                       # если True, то расчет WOE происходит на трэйн сэмпле, иначе берется среднее значение по фолдам
              n_folds=5,                         # кол-во фолдов для расчета WOE при simple=False
              woe_adjust=0.5,                    # корректировочный параметр для расчета EventRate_i в бине i                                                
              alpha=0,                           # коэффициент регуляризации для расчета WOE
              alpha_range=None,                  # если alpha=None, то подбирается оптимальное значение alpha из диапазона alpha_range. При None берется диапазон range(10, 100, 10)
              alpha_scoring='neg_log_loss',      # метрика, используемая для оптимизации alpha
              alpha_best_criterion='min',        # 'min' - минимизация метрики alpha_scoring, 'max' - максимизация метрики
              missing_process='max_or_separate', # способ обработки пустых значений:
                                                 #     'separate' - помещать в отдельный бин
                                                 #     'min' - объединять с бином с минимальным WOE
                                                 #     'max' - объединять с бином с максимальным WOE
                                                 #     'nearest' - объединять с ближайшим по WOE биномом
                                                 #     'min_or_separate' - если доля пустых значений меньше missing_min_part, то объединять с бином с минимальным WOE, иначе помещать в отдельный бин
                                                 #     'max_or_separate' - если доля пустых значений меньше missing_min_part, то объединять с бином с максимальным WOE, иначе помещать в отдельный бин
                                                 #     'nearest_or_separate' - если доля пустых значений меньше missing_min_part, то объединять с ближайшим по WOE бином, иначе помещать в отдельный бин
              missing_min_part=0.01,             # минимальная доля пустых значений для выделения отдельного бина при missing_process 'min_or_separate', 'max_or_separate' или 'nearest_or_separate'
              others='missing_or_max',           # Способ обработки значений, не попавших в биннинг:
                                                 #     'min': остальным значениям присваивается минимальный WOE
                                                 #     'max': остальным значениям присваивается максимальный WOE
                                                 #     'missing_or_min': если есть бакет с пустыми значениями, то остальным значениям присваивается его WOE, иначе минимальный WOE
                                                 #     'missing_or_max': если есть бакет с пустыми значениями, то остальным значениям присваивается его WOE, иначе максимальный WOE
                                                 #     float: отсутствующим значениям присваивается заданный фиксированный WOE
              opposite_sign_to_others=True,      # В случае, когда непрерывная переменная на выборке для разработки имеет только один знак,
                                                 # то все значения с противоположным знаком относить к others
             )

    binning.auto_fit(features=None,                  # список переменных для обработки. По умолчанию берутся из self.ds.features
                 autofit_folder='auto_fit',      # название папки, в которую будут сохранены результаты автобиннинга
                 plot_flag=-1,                 # флаг для вывода графиков с биннингом:
                                                 #   -1 - графики не строить
                                                 #    0, False - графики сохранить в папку autofit_folder/Figs_binning, но не выводить в аутпут
                                                 #    1, True - графики сохранить в папку autofit_folder/Figs_binning и вывести в аутпут
                 verbose=False,                   # флаг для вывода подробны комментариев в процессе работы
                 
                 # --- Метод биннинга ---
                 method='opt',                   # 'tree' - биннинг деревом, 'opt' - биннинг деревом с последующей оптимизацией границ бинов библиотекой optbinning
                 max_n_bins=10,                  # максимальное кол-во бинов
                 min_bin_size=0.05,              # минимальное число (доля) наблюдений в каждом листе дерева.
                                                 #    Если min_bin_size < 1, то трактуется как доля наблюдений от обучающей выборки
                 
                 # --- Параметры биннинга для метода 'tree' ---
                 criterion='entropy',            # критерий расщепления. Варианты значений: 'entropy', 'gini'
                 scoring='neg_log_loss',         # метрика для оптимизации
                 max_depth=5,                    # максимальная глубина дерева
                 
                 #--- Параметры биннинга для метода 'opt' ---
                 solver='cp',                    # солвер для оптимизации биннинга:
                                                 #   'cp' - constrained programming
                                                 #   'mip' - mixed-integer programming
                                                 #   'ls' - LocalSorver (www.localsorver.com)
                 divergence='iv',                # метрика для максимизации:
                                                 #   'iv' - Information Value,
                                                 #   'js' - Jensen-Shannon,
                                                 #   'hellinger' - Hellinger divergence,
                                                 #   'triangular' - triangular discrimination

                 # --- Параметры проверок ---
                 WOEM_on=True,                   # флаг проверки на разницу WOE между соседними группами
                 WOEM_woe_threshold=0.1,         # если дельта WOE между соседними группами меньше этого значения, то группы объединяются
                 WOEM_with_missing=False,        # должна ли выполняться проверка для бина с пустыми значениями
                 SM_on=False,                    # проверка на размер бина
                 SM_target_threshold=10,         # минимальное кол-во (доля) наблюдений с целевым событием в бине
                 SM_size_threshold=100,          # минимальное кол-во (доля) наблюдений в бине
                 BL_on=True,                     # флаг проверки на бизнес-логику
                 BL_allow_Vlogic_to_increase_gini=10, # разрешить V-образную бизнес-логику, если она приводит к увеличению джини переменной на эту величину относительного монотонного тренда.
                                                 #        При значении 100 V-образная бизнес-логика запрещена
                 G_on=True,                      # флаг проверки на джини
                 G_gini_threshold=5,             # минимальное допустимое джини переменной.
                                                 #    Проверяется на трэйн сэмпле, + если заданы бутстрэп сэмплы, то проверяется условие mean-1.96*std > G_gini_threshold
                 G_with_test=True,               # так же проверять джини на остальных доступных сэмплах
                 G_gini_decrease_threshold=0.5,  # допустимое уменьшение джини на всех сэмплах относительно трэйна.
                                                 # В случае, если значение >= 1, то проверяется условие gini(train) - gini(sample) <= G_gini_decrease_threshold для основных сэмплов
                                                 #                                                    и 1.96*std <= G_gini_decrease_threshold для бутсрэп сэмплов
                                                 # если значение < 1, то проверяется условие 1 - gini(sample)/gini(train) <= G_gini_decrease_threshold для основных сэмплов
                                                 #                                         и 1.96*std/mean <= G_gini_decrease_threshold для бутсрэп сэмплов
                 G_gini_increase_restrict=False, # так же ограничение действует и на увеличение джини
                 WOEO_on=True,                   # флаг проверки на сохранения порядка WOE на бутстрэп-сэмплах
                 WOEO_dr_threshold=0.01,         # допустимая дельта между TargetRate бинов для прохождения проверки, в случае нарушения порядка
                 WOEO_correct_threshold=0.90,    # доля бутстрэп-сэмплов, на которых должна проходить проверка
                 WOEO_miss_is_incorrect=True,    # считать ли отсутствие данных в бине сэмпла ошибкой или нет
                 WOEO_with_test=False,           # так же проверять тренд на остальных доступных сэмплах

                 # ---Пространство параметров---
                 params_space=None,              # пространство параметров, с которыми будут выполнены автобиннинги. Задается в виде словаря {параметр: список значений}
                 woe_best_samples=None,          # список сэмплов, джини которых будет учитываться при выборе лучшего биннинга. По умолчанию берется джини на трэйне
                 
                 #--- Кросс переменные ---
                 cross_features_first_level=None,# список переменных первого уровня для которых будут искаться лучшие кросс пары. При None берется features
                 cross_num_second_level=0        # кол-во кросс пар, рассматриваемых для каждой переменной первого уровня
                                                 #   0 - поиск не производится
                                                 #  -1 - рассматриваются все возможные кросс пары
                                                 #   n - для каждой переменной первого уровня отбираются n лучших переменных с максимальной метрикой criterion
                )

    ds = logreg.scoring(ds, score_field='score')
    gini = ds.calc_gini(features=['score'])
    gini_train = abs(gini['train'][0] )
    gini_test = abs(gini['Test'][0] )
    f_dict=logreg.coefs
    #feature_html=''
    features_list=list(f_dict.keys())
    features_len=len(features_list)
    features_coefs = {}
    for key in f_dict:
        feature_numbers = binning.feature_woes[key].calc_groups_stat()
        n = feature_numbers['n'].tolist()
        woe = feature_numbers['woe'].tolist()
        g = list(map(str, binning.feature_woes[key].groups.values()))
        f = lambda x: x.replace("[", "Bin ").replace("]", "").replace(", ", " to ")
        g=list(map(f, g))
        features_coefs.update({key:{1:n,2:woe,3:g}})
    
    return render_template("report.html", gini_html=gini_html,corr_mat_html=corr_mat_html,gini_train=gini_train,gini_test=gini_test,features_list=features_list,features_len=features_len,features_coefs=features_coefs)

@app.route('/underconstruction')
def showUC():
    return render_template("underconstruction.html")

@app.route('/download_test', methods=['GET', 'POST'])
def downloadTest ():
    path = "../test.csv"    
    return send_file(path, as_attachment=True)
@app.route('/download_val', methods=['GET', 'POST'])
def downloadVal ():
    path = "../val.csv"    
    return send_file(path, as_attachment=True)
@app.route('/download_train', methods=['GET', 'POST'])
def downloadTrain ():
    path = "../train.csv"    
    return send_file(path, as_attachment=True)
# (C) START
if __name__ == "__main__":
    from waitress import serve
    serve(app)
  #app.run()
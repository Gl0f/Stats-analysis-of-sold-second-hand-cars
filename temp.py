import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import tkinter as tk
from tkinter import ttk, messagebox
import ast
from tabulate import tabulate
import textwrap

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", context="paper", font_scale=0.9)
plt.rcParams['figure.dpi'] = 100

# ==========================================================
# КЛАС ДЛЯ КРАСИВОГО ВИВЕДЕННЯ В КОНСОЛЬ
# ==========================================================
class UI:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

    @staticmethod
    def title(text):
        print(f"\n{UI.HEADER}{UI.BOLD}{'=' * 85}\n{text.center(85)}\n{'=' * 85}{UI.END}\n")

    @staticmethod
    def success(text):
        print(f"{UI.GREEN}✅ {text}{UI.END}")

    @staticmethod
    def warn(text):
        print(f"{UI.WARNING}⚠️ {text}{UI.END}")


# ==========================================================
# МАТЕМАТИЧНЕ ЯДРО: ВЛАСНИЙ КЛАС РЕГРЕСІЇ (+ ДІАГНОСТИКА)
# ==========================================================
class ManualOLS:
    def __init__(self, X, Y):
        X_mat = np.array(X)
        Y_vec = np.array(Y)
        n, k = X_mat.shape

        # 1. Основні коефіцієнти (МНК)
        XtX_inv = np.linalg.pinv(X_mat.T @ X_mat)
        self.beta = XtX_inv @ X_mat.T @ Y_vec
        self.params = pd.Series(self.beta, index=X.columns)

        # 2. Залишки та дисперсія
        self.fittedvalues = X_mat @ self.beta
        self.resid = Y_vec - self.fittedvalues
        self.df_resid = n - k
        self.df_model = k - 1
        self.nobs = n

        RSS = np.sum(self.resid ** 2)
        TSS = np.sum((Y_vec - np.mean(Y_vec)) ** 2)
        sigma_squared = RSS / self.df_resid

        cov_matrix = sigma_squared * XtX_inv
        diag_elements = np.maximum(0, np.diagonal(cov_matrix))
        self.bse = pd.Series(np.sqrt(diag_elements), index=X.columns)

        # 3. t-статистики та p-значення
        with np.errstate(divide='ignore', invalid='ignore'):
            self.tvalues = np.where(self.bse == 0, 0, self.params / self.bse)
            self.tvalues = pd.Series(self.tvalues, index=X.columns)

        p_vals = 2 * (1 - stats.t.cdf(np.abs(self.tvalues), self.df_resid))
        self.pvalues = pd.Series(p_vals, index=X.columns)

        # 4. Базові метрики якості (R^2, F-stat, RMSE)
        self.rsquared = 1 - (RSS / TSS)
        # СКОРИГОВАНИЙ R^2 (штраф за кількість факторів)
        self.rsquared_adj = 1 - (1 - self.rsquared) * (n - 1) / self.df_resid
        self.fvalue = ((TSS - RSS) / self.df_model) / (RSS / self.df_resid)
        self.rmse = np.sqrt(sigma_squared)

        # ==========================================================
        # РОЗШИРЕНА СТАТИСТИЧНА ДІАГНОСТИКА
        # ==========================================================

        # 1. Log-Likelihood (Логарифмічна функція вірогідності)
        self.llf = -n / 2 * (np.log(2 * np.pi) + np.log(sigma_squared) + 1)

        # 2. Інформаційні критерії (AIC та BIC)
        self.aic = -2 * self.llf + 2 * k
        self.bic = -2 * self.llf + k * np.log(n)

        # 3. Дарбін-Уотсон (Автокореляція залишків)
        # Розраховується мануально: сума квадратів різниць залишків / суму квадратів залишків
        diff_resid = np.diff(self.resid)
        self.dw_stat = np.sum(diff_resid ** 2) / np.sum(self.resid ** 2)

        # 4. Тест Харке-Бера (Нормальність залишків)
        skewness = stats.skew(self.resid)
        kurtosis = stats.kurtosis(self.resid)  # Excess kurtosis
        self.jb_stat = (n / 6) * (skewness ** 2 + (kurtosis ** 2 / 4))
        self.jb_pval = 1 - stats.chi2.cdf(self.jb_stat, df=2)

        # 5. Тест Бреуша-Пагана (Гетероскедастичність)
        try:
            bp_test = het_breuschpagan(self.resid, X_mat)
            self.bp_lm_pval = bp_test[1]
        except:
            self.bp_lm_pval = np.nan

        # 6. VIF (Мультиколінеарність)
        try:
            vif_data = [variance_inflation_factor(X_mat, i) for i in range(k)]
            self.vif = pd.Series(vif_data, index=X.columns)
        except:
            self.vif = pd.Series(np.nan, index=X.columns)


UI.title("РОЗДІЛ 2: ПІДГОТОВКА ДАНИХ ТА СЕГМЕНТАЦІЯ")
print(f"{UI.BLUE}1. Завантаження та очищення даних...{UI.END}")

file_name = 'autoscout24_dataset_20251108.csv'
df = pd.read_csv(file_name)

print("\n[АЛЬФА-ПАРСИНГ] Збір усіх унікальних опцій комплектації...")
equipment_cols = ['equipment_comfort', 'equipment_entertainment', 'equipment_extra', 'equipment_safety']
unique_features = {col: set() for col in equipment_cols}

for col in equipment_cols:
    if col in df.columns:
        for val in df[col].dropna():
            try:
                items = ast.literal_eval(val)
                if isinstance(items, list):
                    unique_features[col].update(items)
            except:
                continue

df['mileage_km_raw'] = df['mileage_km_raw'].astype(str).str.replace(r'\D', '', regex=True)
numeric_cols = ['price', 'mileage_km_raw', 'power_hp', 'production_year', 'nr_prev_owners']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

if 'registration_date' in df.columns and 'nr_prev_owners' in df.columns:
    mask = df['production_year'].isna() & (df['nr_prev_owners'] == 1)
    replacement_years = pd.to_datetime(df.loc[mask, 'registration_date'], errors='coerce').dt.year
    df.loc[mask, 'production_year'] = replacement_years

print(f"{UI.BLUE}2. Розрахунок факторів впливу (Відділення пробігу від віку)...{UI.END}")
df = df.dropna(subset=['price', 'mileage_km_raw', 'power_hp', 'production_year', 'make'])

df['car_age'] = 2026 - df['production_year']
df['service_flag'] = df['has_full_service_history'].apply(lambda x: 1 if str(x).lower() in ['true', '1', 'yes'] else 0)
df['dealer_flag'] = df['seller_is_dealer'].apply(lambda x: 1 if str(x).lower() in ['true', '1', 'yes'] else 0)

for col in ['equipment_comfort', 'equipment_entertainment', 'equipment_extra', 'equipment_safety']:
    if col not in df.columns:
        df[col] = ''

df['all_equip'] = (df['equipment_comfort'].astype(str) + ' ' +
                   df['equipment_entertainment'].astype(str) + ' ' +
                   df['equipment_extra'].astype(str) + ' ' +
                   df['equipment_safety'].astype(str)).str.lower()

df['опція_Преміум_Комфорт'] = df['all_equip'].apply(lambda x: 1 if any(opt in x for opt in ['air suspension', 'massage seats', 'seat ventilation', 'panorama roof', 'heads-up display']) else 0)
df['опція_Цифрова_Мультимедіа'] = df['all_equip'].apply(lambda x: 1 if any(opt in x for opt in ['carplay', 'android auto', 'digital cockpit', 'induction charging']) else 0)
df['опція_Спорт_Пакет'] = df['all_equip'].apply(lambda x: 1 if any(opt in x for opt in ['sport package', 'sport seats', 'sport suspension']) else 0)
df['опція_Матричне_Світло'] = df['all_equip'].apply(lambda x: 1 if any(opt in x for opt in ['laser', 'full-led']) else 0)
df['опція_ADAS_Асистенти'] = df['all_equip'].apply(lambda x: 1 if any(opt in x for opt in ['adaptive cruise', 'blind spot', 'lane departure', 'night view', '360']) else 0)

df['Рівень_Комплектації'] = (df['опція_Преміум_Комфорт'] + df['опція_Цифрова_Мультимедіа'] + df['опція_Спорт_Пакет'] + df['опція_Матричне_Світло'] + df['опція_ADAS_Асистенти'])

df_clean = df[
    (df['price'] >= 1000) & (df['price'] <= 1500000) &
    (df['mileage_km_raw'] >= 10) & (df['mileage_km_raw'] <= 400000) &
    (df['car_age'] >= 0) & (df['car_age'] <= 30) &
    (df['power_hp'] > 40) & (df['power_hp'] < 1500)
    ].copy()

df_clean['log_price'] = np.log(df_clean['price'])
df_clean['Пробіг_тис_км'] = df_clean['mileage_km_raw'] / 1000.0

df_clean['Норма_Пробігу'] = df_clean['car_age'] * 15.0
df_clean['Норма_Пробігу'] = np.where(df_clean['Норма_Пробігу'] == 0, 5.0, df_clean['Норма_Пробігу'])
df_clean['Надлишковий_Пробіг'] = df_clean['Пробіг_тис_км'] - df_clean['Норма_Пробігу']
df_clean['log_mileage_for_cluster'] = np.log(df_clean['mileage_km_raw'])

premium_brands = [x.lower() for x in ['BMW', 'Mercedes-Benz', 'Audi', 'Volvo', 'Lexus', 'Land Rover', 'Jaguar', 'Alfa Romeo', 'Acura', 'Infiniti', 'MINI', 'Tesla']]
luxury_sport_brands = [x.lower() for x in ['Porsche', 'Maserati', 'Ferrari', 'Bentley', 'Aston Martin', 'Lamborghini', 'Rolls-Royce', 'McLaren', 'Lotus']]

def categorize_brand(make):
    m = str(make).strip().lower()
    if m in luxury_sport_brands: return 'Люкс/Спорт'
    if m in premium_brands: return 'Преміум'
    return 'Масовий'

df_clean['Клас_Бренду'] = df_clean['make'].apply(categorize_brand)

def categorize_fuel(f):
    f = str(f).strip().lower()
    if 'hybrid' in f or '/' in f or ('electric' in f and ('gas' in f or 'diesel' in f or 'petrol' in f)): return 'Гібрид'
    if 'diesel' in f: return 'Дизель'
    if 'electric' in f: return 'Електро'
    if 'lpg' in f or 'cng' in f or 'gas' in f: return 'Газ_LPG'
    return 'Бензин'

def categorize_trans(t):
    t = str(t).strip().lower()
    if 'manual' in t: return 'Механіка'
    return 'Автомат'

if 'fuel' in df_clean.columns:
    df_clean['Тип_Палива'] = df_clean['fuel'].apply(categorize_fuel)
elif 'fuel_type' in df_clean.columns:
    df_clean['Тип_Палива'] = df_clean['fuel_type'].apply(categorize_fuel)
else:
    df_clean['Тип_Палива'] = df_clean['primary_fuel'].apply(categorize_fuel)

df_clean['КПП'] = df_clean['transmission'].apply(categorize_trans)

def categorize_body(b):
    b = str(b).strip().lower()
    if 'sedan' in b or 'saloon' in b or 'limousine' in b: return 'Седан'
    if 'station' in b or 'wagon' in b or 'estate' in b or 'kombi' in b or 'touring' in b: return 'Універсал'
    if 'compact' in b or 'hatch' in b: return 'Хетчбек'
    if 'off-road' in b or 'suv' in b or 'pick' in b or 'crossover' in b: return 'Кросовер/Позашляховик'
    if 'coupe' in b: return 'Купе'
    if 'convertible' in b or 'cabrio' in b: return 'Кабріолет'
    if 'van' in b or 'bus' in b or 'transporter' in b: return 'Мінівен'
    return 'Інше'

if 'body_type' in df_clean.columns:
    df_clean['Тип_Кузова'] = df_clean['body_type'].apply(categorize_body)
else:
    df_clean['Тип_Кузова'] = 'Інше'

def categorize_drive(d):
    d = str(d).strip().lower()
    if 'front' in d or 'fwd' in d: return 'Передній'
    if 'rear' in d or 'rwd' in d: return 'Задній'
    if '4wd' in d or 'awd' in d or '4x4' in d or 'all' in d: return 'Повний (4x4)'
    return 'Інше'

if 'drive_train' in df_clean.columns:
    df_clean['Привід'] = df_clean['drive_train'].apply(categorize_drive)
else:
    df_clean['Привід'] = 'Інше'

def categorize_upholstery(u):
    u = str(u).strip().lower()
    if 'full leather' in u: return 'Повна шкіра'
    if 'part leather' in u: return 'Комбінована'
    if 'cloth' in u: return 'Тканина'
    if 'velour' in u: return 'Велюр'
    if 'alcantara' in u: return 'Алькантара'
    return 'Інше'

if 'upholstery' in df_clean.columns:
    df_clean['Оббивка'] = df_clean['upholstery'].apply(categorize_upholstery)
else:
    df_clean['Оббивка'] = 'Інше'

print(f"{UI.BLUE}3. Формування природних ринкових кластерів...{UI.END}")
df_clean['Ринковий_Кластер'] = df_clean['Клас_Бренду'] + " (" + df_clean['Тип_Палива'] + ")"

UI.title("РОЗДІЛ 3: АНСАМБЛЬ (Бренд + Паливо + 6 Кластерів)")

macro_models = {}
segmentation_tools = {}

brand_classes = df_clean['Клас_Бренду'].unique()
fuel_types = df_clean['Тип_Палива'].unique()

MIN_SAMPLES_BASE = 20
MIN_SAMPLES_MICRO = 10

for b_class in brand_classes:
    for f_type in fuel_types:
        df_sub = df_clean[(df_clean['Клас_Бренду'] == b_class) & (df_clean['Тип_Палива'] == f_type)].copy()
        if len(df_sub) < MIN_SAMPLES_BASE: continue

        cluster_features = ['car_age', 'Рівень_Комплектації', 'power_hp']
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_sub[cluster_features])

        # =======================================================
        # МЕТОД ЛІКТЯ (АВТОМАТИЧНЕ ВИЗНАЧЕННЯ ОПТИМАЛЬНОГО k)
        # =======================================================
        wcss = []
        # Захист: максимальна кількість кластерів не може бути більшою за 10
        # і має бути адекватною розміру вибірки
        max_k = min(10, max(3, len(df_sub) // MIN_SAMPLES_BASE))

        K_range = range(2, max_k + 1)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(scaled_data)
            wcss.append(km.inertia_)

        # Математичний пошук точки ліктя (найбільша відстань від кривої до прямої)
        x1, y1 = K_range[0], wcss[0]
        x2, y2 = K_range[-1], wcss[-1]

        distances = []
        for i, k in enumerate(K_range):
            x0, y0 = k, wcss[i]
            # Формула перпендикулярної відстані від точки до прямої
            numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            distances.append(numerator / denominator)

        # Оптимальне k — це точка з найбільшим відхиленням (згином)
        optimal_k = K_range[np.argmax(distances)]

        # Генерація графіку для курсової (для вставки в текст)
        plt.figure(figsize=(8, 5))
        plt.plot(K_range, wcss, marker='o', linestyle='-', color='#3498db', linewidth=2, markersize=8)
        plt.axvline(x=optimal_k, color='#e74c3c', linestyle='--', linewidth=2, label=f'Оптимальне k = {optimal_k}')
        plt.title(f'Метод ліктя: {b_class} ({f_type})', fontweight='bold', fontsize=14)
        plt.xlabel('Кількість кластерів (k)', fontweight='bold')
        plt.ylabel('WCSS (Внутрішньокластерна дисперсія)', fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        safe_name = f"{b_class}_{f_type}".replace('/', '_')
        plt.savefig(f'00_elbow_{safe_name}.png', dpi=300)
        plt.close()

        print(f"[{b_class} - {f_type}] Знайдено оптимальне k: {optimal_k}")

        # =======================================================
        # НАВЧАННЯ K-MEANS З ОПТИМАЛЬНИМ k
        # =======================================================
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        initial_clusters = kmeans.fit_predict(scaled_data).astype(str)
        df_sub['Micro_Cluster'] = initial_clusters

        # =======================================================
        # РОЗУМНЕ ЗЛИТТЯ МАЛОНАПОВНЕНИХ КЛАСТЕРІВ (Адаптовано)
        # =======================================================
        centers = kmeans.cluster_centers_
        cluster_counts = df_sub['Micro_Cluster'].value_counts()

        small_clusters = cluster_counts[cluster_counts < MIN_SAMPLES_MICRO].index.tolist()
        valid_clusters = cluster_counts[cluster_counts >= MIN_SAMPLES_MICRO].index.tolist()

        mapping = {str(i): str(i) for i in range(optimal_k)}

        if small_clusters and valid_clusters:
            # Рахуємо матрицю Евклідових відстаней між центрами всіх кластерів
            dist_matrix = np.zeros((optimal_k, optimal_k))
            for i in range(optimal_k):
                for j in range(optimal_k):
                    dist_matrix[i, j] = np.linalg.norm(centers[i] - centers[j])

            for sc in small_clusters:
                sc_idx = int(sc)
                nearest_vc = None
                min_dist = float('inf')

                # Шукаємо просторово найближчий кластер
                for vc in valid_clusters:
                    vc_idx = int(vc)
                    if dist_matrix[sc_idx, vc_idx] < min_dist:
                        min_dist = dist_matrix[sc_idx, vc_idx]
                        nearest_vc = vc

                if nearest_vc is not None:
                    mapping[sc] = nearest_vc

        elif small_clusters and not valid_clusters:
            mapping = {str(i): '0' for i in range(optimal_k)}

        df_sub['Micro_Cluster'] = df_sub['Micro_Cluster'].map(mapping)

        # Зберігаємо mapping у tools, щоб UI-калькулятор знав, куди перенаправляти нові авто
        segmentation_tools[(b_class, f_type)] = {'scaler': scaler, 'kmeans': kmeans, 'mapping': mapping}

        # Тепер ітеруємось ТІЛЬКИ по тих кластерах, що вижили після злиття
        unique_clusters = df_sub['Micro_Cluster'].unique()
        for c_id in unique_clusters:
            df_micro = df_sub[df_sub['Micro_Cluster'] == c_id].copy()
            # Перевірка (про всяк випадок, хоча після злиття там має бути >= MIN_SAMPLES_MICRO)
            if len(df_micro) < 5: continue

            df_ua = df_micro[['log_price', 'car_age', 'Надлишковий_Пробіг', 'power_hp',
                              'service_flag', 'dealer_flag', 'nr_prev_owners',
                              'опція_Преміум_Комфорт', 'опція_Цифрова_Мультимедіа',
                              'опція_Спорт_Пакет', 'опція_Матричне_Світло', 'опція_ADAS_Асистенти',
                              'КПП', 'Тип_Кузова', 'Привід', 'Оббивка']].dropna()

            df_dum = pd.get_dummies(df_ua, columns=['КПП', 'Тип_Кузова', 'Привід', 'Оббивка'], drop_first=True)

            # =======================================================
            # АНТИ-АНОМАЛІЯ: ЗАХИСТ ВІД РОЗРІДЖЕНИХ ОПЦІЙ ТА НУЛЬОВОЇ ДИСПЕРСІЇ
            # =======================================================
            sparse_cols = ['опція_Преміум_Комфорт', 'опція_Цифрова_Мультимедіа',
                           'опція_Спорт_Пакет', 'опція_Матричне_Світло', 'опція_ADAS_Асистенти',
                           'dealer_flag', 'service_flag']  # Додали прапорці стану для захисту графіків

            for col in sparse_cols:
                if col in df_dum.columns:
                    count_ones = df_dum[col].sum()
                    option_ratio = count_ones / len(df_dum)

                    if option_ratio < 0.05 or option_ratio > 0.95 or count_ones < 3:
                        df_dum = df_dum.drop(columns=[col])

            # Глобальна зачистка: видаляємо будь-які інші колонки, де всі значення однакові
            for col in df_dum.columns:
                if df_dum[col].nunique() <= 1 and col != 'const':
                    df_dum = df_dum.drop(columns=[col])
            # =======================================================



            Y = df_dum['log_price']
            X = df_dum.drop('log_price', axis=1).astype(float)
            X.insert(0, 'const', 1.0)

            test_fraction = 0.2 if len(X) >= 25 else 0.1
            try:
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=42)
            except ValueError:
                X_train, X_test, Y_train, Y_test = X, X, Y, Y

            model = ManualOLS(X_train, Y_train)
            model.X_test = X_test
            model.Y_test = Y_test

            # --- ДОДАНО ДЛЯ ПОРІВНЯННЯ МОДЕЛЕЙ ---
            # 1. Ridge Регресія (штрафує за великі коефіцієнти, рятує масовий сегмент)
            ridge_model = Ridge(alpha=10.0, random_state=42)
            ridge_model.fit(X_train, Y_train)
            model.ridge_preds = ridge_model.predict(X_test)

            # 2. Дерево рішень (вловлює нелінійні зв'язки)
            tree_model = DecisionTreeRegressor(max_depth=10, random_state=42)
            tree_model.fit(X_train, Y_train)
            model.tree_preds = tree_model.predict(X_test)

            # --- ДОДАНО: Зберігаємо саму модель та назви факторів для графіка 7 ---
            model.tree_model = tree_model
            model.feature_names = X.columns
            # -------------------------------------

            macro_models[(b_class, f_type, c_id)] = model

UI.success("Ансамбль моделей успішно навчено!")

UI.title("РОЗДІЛ 4: АНАЛІЗ ВПЛИВУ ФАКТОРІВ ТА ТОЧНІСТЬ АНСАМБЛЮ")

plt.close('all')
sns.set_style("ticks")

# ==========================================================
# 1. ДЕТАЛІЗОВАНІ ТАБЛИЦІ ДЛЯ 3-Х КЛЮЧОВИХ СЕГМЕНТІВ
# ==========================================================
if not macro_models:
    UI.warn("Жодної моделі не було навчено! Перевірте дані або зменшіть MIN_SAMPLES.")
else:
    print(f"\n{UI.HEADER}{UI.BOLD}--- ДЕТАЛЬНИЙ ВПЛИВ ФАКТОРІВ ПО СЕГМЕНТАХ ---{UI.END}")

    target_classes = ['Масовий', 'Преміум', 'Люкс/Спорт']

    for t_class in target_classes:
        # Шукаємо всі моделі для поточного класу
        class_keys = [k for k in macro_models.keys() if k[0] == t_class]

        if not class_keys:
            continue

        # Вибираємо найбільш репрезентативну (найбільшу) модель у цьому класі
        target_key = max(class_keys, key=lambda k: macro_models[k].nobs)
        model_rep = macro_models[target_key]

        print(f"\n{UI.BLUE}{UI.BOLD}=== ПРЕДСТАВНИК СЕГМЕНТУ: {t_class.upper()} {target_key} ==={UI.END}")

        param_data = []
        # 1. Витягуємо константу
        if 'const' in model_rep.params:
            coef = model_rep.params['const']
            t_stat = model_rep.tvalues['const']
            p_val = model_rep.pvalues['const']
            significance = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            param_data.append(
                ["Константа (Базова ціна)", f"{coef:.4f}", f"{t_stat:.2f}", f"{p_val:.4f} {significance}", "-"])

        # 2. Сортуємо інші параметри
        sorted_params = model_rep.params.drop('const', errors='ignore').abs().sort_values(ascending=False).index

        for param in sorted_params:
            coef = model_rep.params[param]
            t_stat = model_rep.tvalues[param]
            p_val = model_rep.pvalues[param]
            vif_val = model_rep.vif.get(param, np.nan)

            significance = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            vif_str = f"{vif_val:.2f}" if pd.notna(vif_val) else "-"

            param_data.append([param, f"{coef:.4f}", f"{t_stat:.2f}", f"{p_val:.4f} {significance}", vif_str])

        print(tabulate(param_data, headers=["Параметр", "Коефіцієнт", "t-стат", "P-значення", "VIF"],
                       tablefmt="fancy_grid"))

        # ==========================================================
        # СТАТИСТИЧНА ДІАГНОСТИКА ДЛЯ ПОТОЧНОГО СЕГМЕНТУ
        # ==========================================================
        diag_table = [
            ["Спостережень (n):", int(model_rep.nobs), "AIC (Акаіке):", f"{model_rep.aic:.1f}"],
            ["R-квадрат (R²):", f"{model_rep.rsquared:.4f}", "BIC (Шварц):", f"{model_rep.bic:.1f}"],
            ["Скориг. R²:", f"{model_rep.rsquared_adj:.4f}", "Дарбін-Уотсон:", f"{model_rep.dw_stat:.2f}"],
            ["F-статистика:", f"{model_rep.fvalue:.2f}", "Харке-Бера (p-val):", f"{model_rep.jb_pval:.4f}"],
            ["RMSE (log-од.):", f"{model_rep.rmse:.4f}", "Бреуш-Паган (p-val):",
             f"{model_rep.bp_lm_pval:.4f}" if pd.notna(model_rep.bp_lm_pval) else "N/A"]
        ]

        print(tabulate(diag_table, tablefmt="simple"))
        print("-" * 85)
        # ==========================================================
        # МАТЕМАТИЧНИЙ ВИГЛЯД МОДЕЛІ (РІВНЯННЯ РЕГРЕСІЇ)
        # ==========================================================
        import textwrap

        print(f"\n{UI.BLUE}Математичний вигляд моделі (Рівняння регресії):{UI.END}")

        eq_terms = []
        # Спочатку додаємо константу (базовий логарифм ціни)
        if 'const' in model_rep.params:
            eq_terms.append(f"{model_rep.params['const']:.4f}")

        # Далі додаємо всі інші фактори
        for param in model_rep.params.drop('const', errors='ignore').index:
            coef = model_rep.params[param]
            sign = "+" if coef >= 0 else "-"

            # Беремо назви зі спецсимволами у квадратні дужки для красивого математичного запису
            safe_param = f"[{param}]" if any(c in param for c in [' ', '/', '(', ')']) else param
            eq_terms.append(f"{sign} {abs(coef):.4f} * {safe_param}")

        # Збираємо все в одну формулу
        equation_str = "log_price = " + " ".join(eq_terms)

        # Автоматично розбиваємо довгу формулу на кілька рядків, щоб вона не вилазила за екран
        wrapped_eq = textwrap.fill(equation_str, width=85, subsequent_indent="            ")
        print(f"{UI.BOLD}{wrapped_eq}{UI.END}\n")
        print("=" * 85)

# ==========================================================
# 2. ОЦІНКА ВСЬОГО АНСАМБЛЮ ТА ГРАФІКИ 2-5 (ОНОВЛЕНО З АНАЛІТИКОЮ)
# ==========================================================
actual_prices = []
predicted_prices = []
log_errors = []

# Словники для зберігання даних по сегментах
# Словники для зберігання даних по сегментах
segment_actuals = {'Масовий': [], 'Преміум': [], 'Люкс/Спорт': []}
segment_preds = {'Масовий': [], 'Преміум': [], 'Люкс/Спорт': []}
segment_ridge_preds = {'Масовий': [], 'Преміум': [], 'Люкс/Спорт': []}
segment_tree_preds = {'Масовий': [], 'Преміум': [], 'Люкс/Спорт': []}

for (b_class, f_type, c_id), model_sub in macro_models.items():
    if hasattr(model_sub, 'X_test') and len(model_sub.X_test) > 0:
        log_preds = np.array(model_sub.X_test) @ model_sub.beta
        actuals = np.exp(model_sub.Y_test.values)
        preds = np.exp(log_preds)

        # Додаємо до загального списку
        actual_prices.extend(actuals)
        predicted_prices.extend(preds)
        log_errors.extend(model_sub.Y_test.values - log_preds)

        # Додаємо до списку відповідного сегменту
        if b_class in segment_actuals:
            segment_actuals[b_class].extend(actuals)
            segment_preds[b_class].extend(preds)
            segment_ridge_preds[b_class].extend(np.exp(model_sub.ridge_preds))
            segment_tree_preds[b_class].extend(np.exp(model_sub.tree_preds))

if actual_prices:
    actual_prices = np.array(actual_prices)
    predicted_prices = np.array(predicted_prices)

    # Загальні метрики
    mae_total = mean_absolute_error(actual_prices, predicted_prices)
    r2_ensemble = r2_score(np.log(actual_prices), np.log(predicted_prices))

    print(f"\n{UI.HEADER}{UI.BOLD}--- ТОЧНІСТЬ АНСАМБЛЮ (MAE ТА R²) ---{UI.END}")
    print(f"Загальна похибка (MAE) всього ансамблю: {UI.GREEN}€{mae_total:,.2f}{UI.END}")
    print(f"Загальний R² ансамблю (на тестових даних): {UI.BOLD}{r2_ensemble:.4f}{UI.END}\n")

    # Метрики по сегментах
    print(f"{UI.BLUE}Точність по сегментах:{UI.END}")
    segment_metrics = []

    for seg in ['Масовий', 'Преміум', 'Люкс/Спорт']:
        if segment_actuals[seg]:
            s_actuals = np.array(segment_actuals[seg])
            s_preds = np.array(segment_preds[seg])
            s_mae = mean_absolute_error(s_actuals, s_preds)
            s_r2 = r2_score(np.log(s_actuals), np.log(s_preds))
            segment_metrics.append([seg, len(s_actuals), f"€ {s_mae:,.2f}", f"{s_r2:.4f}"])
        else:
            segment_metrics.append([seg, 0, "N/A", "N/A"])

    print(tabulate(segment_metrics, headers=["Сегмент", "К-сть тестових авто", "MAE (Похибка)", "R² (на логарифмах)"],
                   tablefmt="fancy_grid", colalign=("left", "center", "right", "center")))

    # ==========================================================
    # --- НОВИЙ БЛОК: ПОРІВНЯННЯ АЛГОРИТМІВ МАШИННОГО НАВЧАННЯ ---
    # ==========================================================
    print(f"\n{UI.HEADER}{UI.BOLD}--- ПОРІВНЯННЯ АЛГОРИТМІВ МАШИННОГО НАВЧАННЯ ---{UI.END}")
    comp_metrics = []

    for seg in ['Масовий', 'Преміум', 'Люкс/Спорт']:
        if segment_actuals[seg]:
            s_actuals = np.array(segment_actuals[seg])
            s_ols = np.array(segment_preds[seg])
            s_ridge = np.array(segment_ridge_preds[seg])
            s_tree = np.array(segment_tree_preds[seg])

            # Розрахунок R2 (на логарифмах, як і вимагає економетрика)
            log_actuals = np.log(s_actuals)
            r2_ols = r2_score(log_actuals, np.log(s_ols))
            r2_ridge = r2_score(log_actuals, np.log(s_ridge))
            r2_tree = r2_score(log_actuals, np.log(s_tree))

            # Розрахунок MAE (в реальних євро)
            mae_ols = mean_absolute_error(s_actuals, s_ols)
            mae_ridge = mean_absolute_error(s_actuals, s_ridge)
            mae_tree = mean_absolute_error(s_actuals, s_tree)

            comp_metrics.append([seg, "OLS (Ваша Базова)", f"€ {mae_ols:,.0f}", f"{r2_ols:.4f}"])
            comp_metrics.append(["", "Ridge Regression", f"€ {mae_ridge:,.0f}", f"{r2_ridge:.4f}"])
            comp_metrics.append(["", "Decision Tree", f"€ {mae_tree:,.0f}", f"{r2_tree:.4f}"])
            comp_metrics.append(["-" * 15, "-" * 20, "-" * 12, "-" * 10])

    if comp_metrics:
        comp_metrics.pop()  # Видаляємо останній розділювач для краси

    print(tabulate(comp_metrics, headers=["Сегмент", "Модель", "MAE (Похибка)", "R² (на логарифмах)"],
                   tablefmt="fancy_grid", colalign=("left", "left", "right", "center")))
    # ==========================================================
    # ==========================================================
    # 1. ГРАФІК 1: ВПЛИВ ФАКТОРІВ ПО СЕГМЕНТАХ (МАСОВИЙ, ПРЕМІУМ, ЛЮКС)
    # ==========================================================
    if not macro_models:
        UI.warn("Жодної моделі не було навчено! Перевірте дані або зменшіть MIN_SAMPLES.")
    else:
        print(f"\n{UI.BLUE}📊 АНАЛІТИКА ДО ГРАФІКА 1 (Вага факторів найрепрезентативніших кластерів):{UI.END}")
        print("Ми вибрали по одному найбільшому (за кількістю авто) кластеру для кожного макросегмента.")
        print("Це дозволяє побачити чистий вплив опцій на ціну без змішування різних підгруп.")

        # Шукаємо найбільший кластер (найбільше nobs) для кожного макросегмента
        segment_best_model = {}

        for (b_class, f_type, c_id), model_sub in macro_models.items():
            if b_class not in segment_best_model:
                segment_best_model[b_class] = {'key': (b_class, f_type, c_id), 'model': model_sub}
            else:
                if model_sub.nobs > segment_best_model[b_class]['model'].nobs:
                    segment_best_model[b_class] = {'key': (b_class, f_type, c_id), 'model': model_sub}

        # Генеруємо графік, таблицю та діагностику для кожного обраного кластера
        for b_class, data in segment_best_model.items():
            best_key = data['key']
            model_rep = data['model']
            f_type, c_id = best_key[1], best_key[2]

            all_params_segment = []

            # Збираємо ВСІ параметри (без фільтрації за p-value)
            for param in model_rep.params.index:
                all_params_segment.append({
                    'Параметр': param,
                    'Коефіцієнт': model_rep.params[param],
                    't-стат': model_rep.tvalues[param],
                    'P-значення': model_rep.pvalues[param],
                    'VIF': model_rep.vif.get(param, np.nan) if hasattr(model_rep, 'vif') and isinstance(model_rep.vif,
                                                                                                        dict) else np.nan
                })

            df_params_seg = pd.DataFrame(all_params_segment)

            if not df_params_seg.empty:
                df_params_seg.set_index('Параметр', inplace=True)

                # Сортуємо таблицю: спочатку константа, потім все інше за модулем впливу
                if 'const' in df_params_seg.index:
                    const_row = df_params_seg.loc[['const']]
                    other_rows = df_params_seg.drop('const').reindex(
                        df_params_seg.drop('const')['Коефіцієнт'].abs().sort_values(ascending=False).index)
                    df_params_seg_sorted = pd.concat([const_row, other_rows])
                else:
                    df_params_seg_sorted = df_params_seg.reindex(
                        df_params_seg['Коефіцієнт'].abs().sort_values(ascending=False).index)

                # ==========================================================
                # 1. ТАБЛИЦЯ КОЕФІЦІЄНТІВ
                # ==========================================================
                print(
                    f"\n{UI.BLUE}{UI.BOLD}=== ПРЕДСТАВНИК СЕГМЕНТУ: {b_class.upper()} ('{b_class}', '{f_type}', '{c_id}') ==={UI.END}")
                table_data = []
                for param, row in df_params_seg_sorted.iterrows():
                    coef = row['Коефіцієнт']
                    t_stat = row['t-стат']
                    p_val = row['P-значення']
                    vif_val = row['VIF']

                    param_name = "Константа (Базова ціна)" if param == 'const' else param
                    significance = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
                    vif_str = f"{vif_val:.2f}" if pd.notna(vif_val) else "-"

                    table_data.append(
                        [param_name, f"{coef:.4f}", f"{t_stat:.2f}", f"{p_val:.4f} {significance}", vif_str])

                print(tabulate(table_data, headers=["Параметр", "Коефіцієнт", "t-стат", "P-значення", "VIF"],
                               tablefmt="fancy_grid"))

                # ==========================================================
                # 2. ДІАГНОСТИЧНИЙ БЛОК (Метрики конкретної моделі)
                # ==========================================================
                diag_table = [
                    ["Спостережень (n):", int(model_rep.nobs), "AIC:", f"{model_rep.aic:.1f}"],
                    ["R-квадрат (R²):", f"{model_rep.rsquared:.4f}", "BIC:", f"{model_rep.bic:.1f}"],
                    ["Скориг. R²:", f"{model_rep.rsquared_adj:.4f}", "Дарбін-Уотсон:", f"{model_rep.dw_stat:.2f}"],
                    ["F-статистика:", f"{model_rep.fvalue:.2f}", "Харке-Бера (p):", f"{model_rep.jb_pval:.4f}"],
                    ["RMSE (log-од.):", f"{model_rep.rmse:.4f}", "Бреуш-Паган (p):",
                     f"{model_rep.bp_lm_pval:.4f}" if pd.notna(model_rep.bp_lm_pval) else "N/A"]
                ]
                print("-" * 85)
                print(tabulate(diag_table, tablefmt="simple"))
                print("-" * 85)

                # ==========================================================
                # 3. МАТЕМАТИЧНЕ РІВНЯННЯ
                # ==========================================================
                print(f"\n{UI.BLUE}Математичний вигляд моделі (Рівняння регресії):{UI.END}")
                eq_terms = []
                if 'const' in df_params_seg_sorted.index:
                    eq_terms.append(f"{df_params_seg_sorted.loc['const', 'Коефіцієнт']:.4f}")

                for param in df_params_seg_sorted.drop('const', errors='ignore').index:
                    coef = df_params_seg_sorted.loc[param, 'Коефіцієнт']
                    sign = "+" if coef >= 0 else "-"
                    safe_param = f"[{param}]" if any(c in param for c in [' ', '/', '(', ')']) else param
                    eq_terms.append(f"{sign} {abs(coef):.4f} * {safe_param}")

                equation_str = "log_price = " + " ".join(eq_terms)
                wrapped_eq = textwrap.fill(equation_str, width=85, subsequent_indent="            ")
                print(f"{UI.BOLD}{wrapped_eq}{UI.END}\n")
                print("=" * 85)

                # ==========================================================
                # 4. ВІДНОВЛЕНИЙ ГРАФІК (Всі фактори, крім константи)
                # ==========================================================
                # Відкидаємо константу, всі інші параметри залишаємо
                params_seg_for_plot = df_params_seg_sorted['Коефіцієнт'].drop('const', errors='ignore').sort_values()

                # Збільшуємо графік до (14, 12), щоб всі 20+ факторів помістилися
                plt.figure(figsize=(14, 12))
                colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in params_seg_for_plot.values]
                bars = plt.barh(params_seg_for_plot.index, params_seg_for_plot.values,
                                color=colors, edgecolor='black',
                                linewidth=1.2, alpha=0.85)

                for bar in bars:
                    width = bar.get_width()
                    ha = 'left' if width > 0 else 'right'
                    # Відступ для тексту, щоб не накладався на лінію графіка
                    offset = max(0.005, abs(width) * 0.02) if width > 0 else min(-0.005, -abs(width) * 0.02)
                    plt.text(width + offset, bar.get_y() + bar.get_height() / 2, f'{width:.4f}',
                             va='center', ha=ha, fontsize=10, fontweight='bold', color='#2c3e50')

                plt.title(f'Вплив факторів на ціну\nСегмент: {b_class} (Найбільший кластер: {f_type} №{c_id})',
                          fontweight='bold',
                          fontsize=16, pad=15)
                plt.xlabel('Коефіцієнт регресії (вплив на log_price)', fontweight='bold', fontsize=12)

                if not params_seg_for_plot.empty:
                    # Розширюємо межі X на 20%, щоб текст збоку від стовпця точно не обрізався
                    max_abs_val = abs(params_seg_for_plot).max() * 1.2
                    plt.xlim(-max_abs_val, max_abs_val)

                plt.axvline(0, color='#34495e', linewidth=2, linestyle='--')
                plt.grid(axis='x', linestyle=':', alpha=0.7, color='gray')
                sns.despine()
                plt.tight_layout()

                safe_name = b_class.replace('/', '_')
                plt.savefig(f'01_вага_факторів_{safe_name}.png', dpi=300)
                plt.close()  # Закриваємо фігуру, щоб не перевантажувати пам'ять
                print(f"{UI.GREEN}✅ Графік усіх факторів для сегмента '{b_class}' збережено.{UI.END}")

                # ==========================================================
                # 2. ГРАФІКИ 2: ТОЧНІСТЬ АНСАМБЛЮ (ДИНАМІЧНІ МЕЖІ + ВСІ АЛГОРИТМИ)
                # ==========================================================
                print(f"\n{UI.BLUE}📊 АНАЛІТИКА ДО ГРАФІКІВ 2 (Точність моделей):{UI.END}")
                print(
                    f"Система помиляється в середньому на {UI.GREEN}€{mae_total:.0f}{UI.END} при прогнозуванні ціни авто.")
                print(f"Ансамбль моделей (OLS) пояснює {r2_ensemble * 100:.1f}% варіації цін.")


                def plot_predictions(actuals_dict, preds_dict, title, filename):
                    plt.figure(figsize=(10, 8))
                    segment_colors = {
                        'Масовий': '#27ae60',
                        'Преміум': '#2980b9',
                        'Люкс/Спорт': '#e74c3c'
                    }
                    plot_order = ['Преміум', 'Масовий', 'Люкс/Спорт']

                    all_actuals = []
                    all_preds = []

                    for seg in plot_order:
                        if actuals_dict[seg]:
                            plt.scatter(actuals_dict[seg], preds_dict[seg],
                                        alpha=0.75, s=35, color=segment_colors[seg],
                                        edgecolors='white', linewidth=0.5, label=f'{seg}')
                            all_actuals.extend(actuals_dict[seg])
                            all_preds.extend(preds_dict[seg])

                    # Використовуємо 98-й перцентиль замість жорсткого ліміту 100 000
                    if len(all_actuals) > 0:
                        max_p = np.percentile(all_actuals, 98)
                    else:
                        max_p = 100000

                    plt.plot([0, max_p], [0, max_p], color='#2c3e50', lw=2.5, linestyle='--',
                             label='Ідеальний прогноз (Y=X)')
                    plt.xlim(0, max_p)
                    plt.ylim(0, max_p)

                    plt.title(title, fontweight='bold', fontsize=15, pad=15)
                    plt.xlabel('Фактична ціна автомобіля (€)', fontweight='bold', fontsize=12)
                    plt.ylabel('Прогнозована ціна (€)', fontweight='bold', fontsize=12)

                    if len(all_actuals) > 0 and len(all_preds) > 0:
                        r2_val = r2_score(np.log(all_actuals), np.log(all_preds))
                        plt.text(max_p * 0.05, max_p * 0.9, f'$R^2$ = {r2_val:.3f}',
                                 fontsize=14, fontweight='bold',
                                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='#bdc3c7'))

                    plt.legend(loc='lower right', frameon=True, shadow=True, markerscale=1.5)
                    plt.grid(True, linestyle=':', alpha=0.6)
                    sns.despine()
                    plt.tight_layout()
                    plt.savefig(filename, dpi=300)
                    plt.close()


                # Генеруємо 3 графіки для різних моделей
                plot_predictions(segment_actuals, segment_preds,
                                 'Якість прогнозу: OLS',
                                 '02a_predicted_vs_actual_ols.png')

                plot_predictions(segment_actuals, segment_ridge_preds,
                                 'Якість прогнозу: Ridge Regression',
                                 '02b_predicted_vs_actual_ridge.png')

                plot_predictions(segment_actuals, segment_tree_preds,
                                 'Якість прогнозу: Decision Tree',
                                 '02c_predicted_vs_actual_tree.png')

                print(
                    f"{UI.GREEN}✅ 3 графіки порівняння фактичної і прогнозованої ціни збережено (02a, 02b, 02c).{UI.END}")
#preadicted_vs_actual
    # --- Графік 3: Розподіл похибок із зоною довіри ---
    mean_err = np.mean(log_errors)
    std_err = np.std(log_errors)

    print(f"\n{UI.BLUE}📊 АНАЛІТИКА ДО ГРАФІКА 3 (Розподіл залишків):{UI.END}")
    print("Гістограма перевіряє ключове математичне припущення — чи мають похибки моделі нормальний розподіл.")
    print(
        f"Зміщення центру (bias): {mean_err:.4f} log-одиниць (близьке до нуля означає відсутність системного переоцінювання/недооцінювання).")
    print(
        f"Близько 68% усіх прогнозів (зафарбована зона) відхиляються від реальної ціни не більше ніж на ±{std_err:.3f} log-одиниць.")

    plt.figure(figsize=(10, 6))
    sns.histplot(log_errors, kde=True, color='#9b59b6', bins=60, alpha=0.4, line_kws={'linewidth': 2.5})

    plt.axvline(mean_err, color='#2c3e50', linestyle='-', lw=2, label=f'Середнє: {mean_err:.3f}')
    plt.axvline(mean_err + std_err, color='#7f8c8d', linestyle='--', lw=1.5, label='±1 Сигма (68% похибок)')
    plt.axvline(mean_err - std_err, color='#7f8c8d', linestyle='--', lw=1.5)
    plt.axvspan(mean_err - std_err, mean_err + std_err, color='#bdc3c7', alpha=0.2)

    plt.xlim(-1,1)
    plt.title('Розподіл залишків моделі (Похибка в логарифмах)', fontweight='bold', fontsize=15, pad=15)
    plt.xlabel('Відхилення (log_price)', fontweight='bold')
    plt.ylabel('Кількість автомобілів', fontweight='bold')
    plt.legend(frameon=True)
    sns.despine()
    plt.tight_layout()
    plt.savefig('03_розподіл_похибок.png', dpi=300)

    # --- Графік 4: Криві знецінення за типами палива для кожного класу авто ---
    print(f"\n{UI.BLUE}📊 АНАЛІТИКА ДО ГРАФІКА 4 (Криві знецінення за паливом):{UI.END}")
    print("Генеруємо 3 окремі графіки (Масовий, Преміум, Люкс/Спорт). На кожному порівнюємо різні типи палива.")

    df_clean['Ціна_Євро'] = df_clean['price']

    # Відбираємо авто віком до 20 років для адекватного масштабу
    df_plot = df_clean[df_clean['car_age'] <= 20].copy()

    # Задаємо єдину палітру для типів палива, щоб кольори були однаковими на всіх 3 графіках
    fuel_colors = {
        'Бензин': '#2980b9',  # Синій
        'Дизель': '#8e44ad',  # Фіолетовий
        'Гібрид': '#f39c12',  # Помаранчевий
        'Електро': '#27ae60',  # Зелений
        'Газ_LPG': '#7f8c8d'  # Сірий
    }

    target_classes = ['Масовий', 'Преміум', 'Люкс/Спорт']

    for b_class in target_classes:
        # Вибираємо дані лише для поточного класу
        df_class = df_plot[df_plot['Клас_Бренду'] == b_class].copy()

        if df_class.empty:
            continue

        # Фільтр від аномалій: залишаємо лише ті точки (паливо + вік), де є хоча б 3 авто
        age_fuel_counts = df_class.groupby(['Тип_Палива', 'car_age']).size().reset_index(name='counts')
        valid_groups = age_fuel_counts[age_fuel_counts['counts'] >= 3]
        df_class_filtered = pd.merge(df_class, valid_groups[['Тип_Палива', 'car_age']], on=['Тип_Палива', 'car_age'])

        # Відсікаємо аномалії по ціні зверху, щоб графік не розтягувало через одиничні ексклюзиви
        max_price_limit = 100000 if b_class == 'Масовий' else 250000
        df_class_filtered = df_class_filtered[df_class_filtered['Ціна_Євро'] <= max_price_limit]

        if df_class_filtered.empty:
            continue

        plt.figure(figsize=(10, 7))

        # Малюємо лінії
        sns.lineplot(
            data=df_class_filtered,
            x='car_age',
            y='Ціна_Євро',
            hue='Тип_Палива',  # Розділяємо лінії за типом палива
            estimator=np.median,
            errorbar=('pi', 50),  # <--- ПОВЕРНУЛИ ЗОНИ РОЗМАХУ
            err_kws={'alpha': 0.15},
            linewidth=2.5,
            marker='o',
            markersize=7,
            palette=fuel_colors  # Застосовуємо нашу палітру
        )

        plt.title(f'Динаміка знецінення за типами палива\nСегмент: {b_class}', fontweight='bold', fontsize=15, pad=15)
        plt.xlabel('Вік автомобіля (років)', fontweight='bold', fontsize=12)
        plt.ylabel('Медіанна ринкова ціна (€)', fontweight='bold', fontsize=12)
        plt.xticks(np.arange(0, 21, 2))  # Крок у 2 роки

        # Налаштування легенди
        plt.legend(title='Тип палива', frameon=True, fontsize=11, title_fontsize=12)

        plt.grid(True, linestyle=':', alpha=0.6)
        sns.despine()
        plt.tight_layout()

        # Зберігаємо графік (наприклад: 04_крива_знецінення_Преміум.png)
        safe_name = b_class.replace('/', '_')
        plt.savefig(f'04_крива_знецінення_{safe_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"{UI.GREEN}✅ 3 порівняльні графіки кривих знецінення успішно збережено.{UI.END}")

    # --- Графік 5: Три повноцінні матриці кореляції для різних сегментів ---
    print(f"\n{UI.BLUE}📊 АНАЛІТИКА ДО ГРАФІКА 5 (Матриці кореляції по класах):{UI.END}")
    print("Генеруємо окремі матриці для уникнення Парадоксу Сімпсона.")

    corr_cols = [
        'log_price', 'car_age', 'Пробіг_тис_км', 'power_hp',
        'nr_prev_owners', 'service_flag', 'dealer_flag',
        'Рівень_Комплектації', 'опція_Спорт_Пакет', 'опція_Преміум_Комфорт'
    ]

    labels = [
        'Ціна (log)', 'Вік', 'Пробіг (тис)', 'Потужність',
        'К-сть власників', 'Сервісна книжка', 'Офіційний дилер',
        'К-сть опцій', 'Спорт Пакет', 'Преміум Комфорт'
    ]

    target_classes = ['Масовий', 'Преміум', 'Люкс/Спорт']

    for b_class in target_classes:
        df_corr = df_clean[df_clean['Клас_Бренду'] == b_class][corr_cols].dropna()

        if len(df_corr) < 30:  # Захист від занадто малих даних
            UI.warn(f"Недостатньо даних для матриці {b_class}")
            continue

        corr = df_corr.corr()

        plt.figure(figsize=(11, 9))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdBu_r', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": .8},
                    xticklabels=labels, yticklabels=labels, annot_kws={"size": 10, "weight": "bold"})

        plt.title(f'Матриця кореляції: {b_class} сегмент', fontweight='bold', fontsize=16, pad=20)

        ax = plt.gca()
        ax.xaxis.tick_bottom()
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        # Зберігаємо кожну матрицю з окремим іменем
        safe_name = b_class.replace('/', '_')
        plt.savefig(f'05_кореляційна_матриця_{safe_name}.png', dpi=300)
        print(f"{UI.GREEN}✅ Матрицю для '{b_class}' збережено.{UI.END}")

    # ПОКАЗУЄМО УСІ ЗГЕНЕРОВАНІ ГРАФІКИ НА ЕКРАНІ
    UI.success("Преміальні графіки збережено (роздільна здатність 300 DPI). Відкриваємо вікна...")
    plt.show()

UI.title("РОЗДІЛ 5: ПОРІВНЯЛЬНИЙ АНАЛІЗ ВПЛИВУ ПАЛИВА ПО СЕГМЕНТАХ")

fuel_impact_data = []

# Збираємо коефіцієнти з усіх навчених моделей ансамблю
for (b_class, f_type, c_id), model_sub in macro_models.items():
    p = model_sub.params

    try:
        # Беремо модулі коефіцієнтів, щоб показати "силу штрафу" (вони від'ємні в моделі)
        # Множимо на 10, щоб отримати штраф за 10 років / 10 тис км надлишкового пробігу
        age_penalty = abs(p.get('car_age', 0)) * 10
        mileage_penalty = abs(p.get('Надлишковий_Пробіг', 0)) * 10

        # Переводимо у відсотки падіння ціни
        age_drop_pct = (1 - np.exp(-age_penalty)) * 100
        mileage_drop_pct = (1 - np.exp(-mileage_penalty)) * 100

        fuel_impact_data.append({
            'Сегмент': b_class,
            'Паливо': f_type,
            'К-сть авто': model_sub.nobs,
            'Знецінення за 10 років (%)': round(age_drop_pct, 1),
            'Штраф за +10 тис. км (%)': round(mileage_drop_pct, 2)
        })
    except:
        continue

df_fuel_impact = pd.DataFrame(fuel_impact_data)

# Усереднюємо дані, якщо для однієї комбінації (Бренд+Паливо) є кілька мікрокластерів
df_fuel_impact = df_fuel_impact.groupby(['Сегмент', 'Паливо']).agg({
    'К-сть авто': 'sum',
    'Знецінення за 10 років (%)': 'mean',
    'Штраф за +10 тис. км (%)': 'mean'
}).reset_index()

# Відсікаємо статистичний шум (комбінації, де сумарно менше 30 авто)
df_fuel_impact = df_fuel_impact[df_fuel_impact['К-сть авто'] >= 30]

# Сортуємо для красивого виводу
df_fuel_impact = df_fuel_impact.sort_values(by=['Сегмент', 'Знецінення за 10 років (%)'], ascending=[True, False])

print(f"{UI.BLUE}📊 Матриця чутливості до зносу (Вік vs Пробіг):{UI.END}")
print(tabulate(df_fuel_impact, headers='keys', tablefmt='fancy_grid', showindex=False))

# ==========================================================
# ГРАФІК 6.1: ВІЗУАЛІЗАЦІЯ ВПЛИВУ ПАЛИВА (ВІК)
# ==========================================================
plt.figure(figsize=(9, 6))
sns.barplot(data=df_fuel_impact, x='Сегмент', y='Знецінення за 10 років (%)', hue='Паливо', palette='Set2', edgecolor='black')
plt.title('Падіння ціни за 10 років експлуатації', fontweight='bold', fontsize=15, pad=15)
plt.ylabel('Втрата вартості (%)', fontweight='bold', fontsize=12)
plt.xlabel('Макросегмент ринку', fontweight='bold', fontsize=12)
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.legend(title='Тип палива', loc='upper right', frameon=True)

sns.despine()
plt.tight_layout()
plt.savefig('06a_порівняння_палива_вік.png', dpi=300)
plt.close() # Закриваємо графік, щоб він не накладався на наступний

# ==========================================================
# ГРАФІК 6.2: ВІЗУАЛІЗАЦІЯ ВПЛИВУ ПАЛИВА (ПРОБІГ)
# ==========================================================
plt.figure(figsize=(9, 6))
sns.barplot(data=df_fuel_impact, x='Сегмент', y='Штраф за +10 тис. км (%)', hue='Паливо', palette='Set2', edgecolor='black')
plt.title('Штраф за кожні 10 000 км перепробігу', fontweight='bold', fontsize=15, pad=15)
plt.ylabel('Втрата вартості (%)', fontweight='bold', fontsize=12)
plt.xlabel('Макросегмент ринку', fontweight='bold', fontsize=12)
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.legend(title='Тип палива', loc='upper right', frameon=True)

sns.despine()
plt.tight_layout()
plt.savefig('06b_порівняння_палива_пробіг.png', dpi=300)
plt.close()

print(f"{UI.GREEN}✅ Окремі графіки порівняння паливних систем збережено (06a_... та 06b_...).{UI.END}\n")

UI.title("РОЗДІЛ 6: ДЕТАЛІЗОВАНИЙ ПРОФІЛЬ МІКРОРИНКІВ")

# ==========================================================
# ГРАФІК 7: ВАЖЛИВІСТЬ ОЗНАК (FEATURE IMPORTANCE) ДЛЯ ПРЕМІУМ-СЕГМЕНТА
# ==========================================================
print(f"\n{UI.BLUE}📊 АНАЛІТИКА ДО ГРАФІКА 7 (Важливість ознак - Decision Tree):{UI.END}")
print("Аналізуємо нелінійний вплив факторів для найбільшого Преміум-кластера...")

premium_models = {k: v for k, v in macro_models.items() if k[0] == 'Преміум'}
if premium_models:
    best_premium_key = max(premium_models.keys(), key=lambda k: premium_models[k].nobs)
    best_premium_model = premium_models[best_premium_key]

    if hasattr(best_premium_model, 'tree_model'):
        importances = best_premium_model.tree_model.feature_importances_
        features = best_premium_model.feature_names

        # Відкидаємо константу (в деревах вона не потрібна) і беремо Топ-10 факторів
        feat_imp = pd.Series(importances, index=features).drop('const', errors='ignore')
        feat_imp = feat_imp[feat_imp > 0].sort_values(ascending=True).tail(10)

        plt.figure(figsize=(10, 6))

        # Градієнтна заливка для краси
        colors = sns.color_palette("viridis", len(feat_imp))
        bars = plt.barh(feat_imp.index, feat_imp.values, color=colors, edgecolor='black', alpha=0.85)

        # Додаємо відсотки біля кожного стовпця
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f'{width * 100:.1f}%',
                     va='center', ha='left', fontsize=10, fontweight='bold', color='#2c3e50')

        plt.title(
            f'Важливість ознак (Decision Tree) - Преміум сегмент\nКластер: {best_premium_key[1]} №{best_premium_key[2]}',
            fontweight='bold', fontsize=15, pad=15)
        plt.xlabel('Відносна важливість (Зниження MSE)', fontweight='bold')

        # Розширюємо межі X, щоб вліз текст відсотків
        plt.xlim(0, max(feat_imp.values) * 1.15)
        plt.grid(axis='x', linestyle=':', alpha=0.7)
        sns.despine()

        plt.tight_layout()
        plt.savefig('07_важливість_ознак_преміум.png', dpi=300)
        plt.close()
        print(f"{UI.GREEN}✅ Графік '07_важливість_ознак_преміум.png' успішно збережено.{UI.END}")

results_data = []
for (b_class, f_type, c_id), model_sub in macro_models.items():
    df_sub = df_clean[(df_clean['Клас_Бренду'] == b_class) & (df_clean['Тип_Палива'] == f_type)].copy()
    if (b_class, f_type) not in segmentation_tools: continue

    tools = segmentation_tools[(b_class, f_type)]
    cluster_features = ['car_age', 'Рівень_Комплектації', 'power_hp']
    scaled_sub = tools['scaler'].transform(df_sub[cluster_features])

    # Застосовуємо predict
    df_sub['Micro_Cluster'] = tools['kmeans'].predict(scaled_sub).astype(str)

    # ДОДАНО: Застосовуємо мапінг злиття
    if 'mapping' in tools:
        df_sub['Micro_Cluster'] = df_sub['Micro_Cluster'].map(tools['mapping'])

    df_micro = df_sub[df_sub['Micro_Cluster'] == c_id]
    if len(df_micro) == 0: continue

    real_prices = df_micro['price'].values
    median_p = int(np.median(real_prices))
    q25_p = int(np.percentile(real_prices, 25))
    q75_p = int(np.percentile(real_prices, 75))

    min_age = int(df_micro['car_age'].min())
    max_age = int(df_micro['car_age'].max())
    min_year = 2026 - max_age
    max_year = 2026 - min_age
    mode_year = int(2026 - df_micro['car_age'].mode()[0])

    q25_m = int(np.percentile(df_micro['mileage_km_raw'], 25) / 1000)
    q75_m = int(np.percentile(df_micro['mileage_km_raw'], 75) / 1000)
    q25_hp = int(np.percentile(df_micro['power_hp'], 25))
    q75_hp = int(np.percentile(df_micro['power_hp'], 75))
    median_equip = round(np.median(df_micro['Рівень_Комплектації']), 1)

    results_data.append({
        'Ринок': f"{b_class} ({f_type})",
        'Кл.': f"№{c_id}",
        'К-сть': len(df_micro),
        'Медіана (€)': median_p,
        'Ціна (50%)': f"€ {q25_p} - {q75_p}",
        'Роки випуску': f"{min_year}-{max_year}",
        'Топ-рік': mode_year,
        'Пробіг (тис.км)': f"{q25_m}-{q75_m}",
        'Потужність': f"{q25_hp}-{q75_hp}",
        'Опції': f"{median_equip} / 5",
        'R^2': round(model_sub.rsquared, 3)
    })

df_res = pd.DataFrame(results_data).sort_values(by=['Ринок', 'Медіана (€)'], ascending=[True, True])
print(tabulate(df_res, headers='keys', tablefmt='fancy_grid', showindex=False, numalign="center"))
print(f"{UI.GREEN}{UI.BOLD}\n✅ Усі розрахунки завершено. Запускаємо калькулятор...{UI.END}\n")

# ==========================================================
# ЕТАП 7: ГРАФІЧНИЙ ІНТЕРФЕЙС (TKINTER APP)
# ==========================================================
def calculate_price():
    try:
        age = float(entry_age.get())
        mileage_raw = float(entry_mileage.get())
        hp = float(entry_hp.get())
        owners = float(entry_owners.get())
        service = var_service.get()
        dealer = 1 if combo_seller.get() == "Офіційний дилер" else 0

        opt_comfort = var_comfort.get()
        opt_media = var_media.get()
        opt_sport = var_sport.get()
        opt_light = var_light.get()
        opt_adas = var_adas.get()

        brand_class = combo_brand.get()
        fuel = combo_fuel.get()
        transmission = combo_trans.get()
        body = combo_body.get()
        drive = combo_drive.get()
        upholstery = combo_upholstery.get()

        if mileage_raw < 0 or hp <= 0 or age < 0:
            raise ValueError("Значення повинні бути додатними.")

        mileage_k = mileage_raw / 1000.0
        norm_m = age * 15.0 if age > 0 else 5.0
        excess_mileage = mileage_k - norm_m

        model_key_base = (brand_class, fuel)
        # =======================================================
        # СИСТЕМА FALLBACK (РЕЗЕРВНОГО РИНКУ)
        # =======================================================
        fallback_warning = ""
        if model_key_base not in segmentation_tools:
            fallback_fuel = "Бензин"  # Резервний базовий ринок
            fallback_key = (brand_class, fallback_fuel)

            if fallback_key in segmentation_tools:
                model_key_base = fallback_key
                fallback_warning = f"⚠️ Замало даних для комбінації '{brand_class} + {fuel}'. Оцінку виконано за алгоритмами ринку '{fallback_fuel}' цього ж класу.\n"
                fuel = fallback_fuel  # Підміняємо паливо для подальшого пошуку кластера
            else:
                # Якщо немає навіть бензину, тоді дійсно викидаємо помилку
                raise ValueError(f"Критично мало даних для об'єктивної оцінки бренду класу '{brand_class}'.")
        # =======================================================

        tools = segmentation_tools[model_key_base]
        scaler_tools = tools['scaler']
        kmeans_tools = tools['kmeans']

        equip_score = opt_comfort + opt_media + opt_sport + opt_light + opt_adas
        user_features = pd.DataFrame([[age, 2, hp]], columns=['car_age', 'Рівень_Комплектації', 'power_hp'])
        user_scaled = scaler_tools.transform(user_features)
        # НОВИЙ ВАРІАНТ:
        raw_c_id = str(kmeans_tools.predict(user_scaled)[0])
        mapping_tools = tools.get('mapping', {})
        # Перенаправляємо авто у великий кластер, якщо його рідний кластер був злитий
        c_id = mapping_tools.get(raw_c_id, raw_c_id)

        model_key_full = (brand_class, fuel, c_id)
        if model_key_full not in macro_models:
            raise ValueError(f"Ваше авто потрапило в рідкісний кластер (№{c_id}), для якого забракло даних.")

        selected_model = macro_models[model_key_full]
        p = selected_model.params

        # --- РОЗБИВКА ДЛЯ ДЕТАЛІЗАЦІЇ ---
        base_log = p.get('const', 0) + p.get('car_age', 0) * age
        if f'КПП_{transmission}' in p: base_log += p[f'КПП_{transmission}']
        if f'Тип_Кузова_{body}' in p: base_log += p[f'Тип_Кузова_{body}']
        if f'Привід_{drive}' in p: base_log += p[f'Привід_{drive}']
        if f'Оббивка_{upholstery}' in p: base_log += p[f'Оббивка_{upholstery}']

        state_log = (p.get('Надлишковий_Пробіг', 0) * excess_mileage +
                     p.get('service_flag', 0) * service +
                     p.get('dealer_flag', 0) * dealer +
                     p.get('nr_prev_owners', 0) * owners)

        equip_log = (p.get('power_hp', 0) * hp +
                     p.get('опція_Преміум_Комфорт', 0) * opt_comfort +
                     p.get('опція_Цифрова_Мультимедіа', 0) * opt_media +
                     p.get('опція_Спорт_Пакет', 0) * opt_sport +
                     p.get('опція_Матричне_Світло', 0) * opt_light +
                     p.get('опція_ADAS_Асистенти', 0) * opt_adas)

        log_pred = base_log + state_log + equip_log
        raw_final_price = np.exp(log_pred)

        max_cluster_price = np.exp(selected_model.fittedvalues.max())
        min_cluster_price = np.exp(selected_model.fittedvalues.min())

        final_price = raw_final_price
        warning_msg = ""

        if final_price > max_cluster_price * 1.15:
            final_price = max_cluster_price * 1.15
            warning_msg = "⚠️ Ціна обмежена історичним максимумом кластера (аномальні параметри)."
        if final_price < min_cluster_price * 0.85:
            final_price = min_cluster_price * 0.85
            warning_msg = "⚠️ Ціна піднята до історичного мінімуму кластера (занадто старе авто)."

        # ДОДАЄМО НАШЕ ПОПЕРЕДЖЕННЯ СЮДИ:
        warning_msg = fallback_warning + warning_msg

        rmse = selected_model.rmse
        lower_bound = np.exp(np.log(final_price) - rmse)
        upper_bound = np.exp(np.log(final_price) + rmse)

        min_scrap_value = 500
        is_scrap = False

        if lower_bound < min_scrap_value:
            lower_bound = min_scrap_value
            is_scrap = True
        if final_price < min_scrap_value: final_price = min_scrap_value
        if upper_bound < min_scrap_value + 200:
            upper_bound = min_scrap_value + 200

        # Оновлення головних цифр
        label_result.config(
            text=f"Діапазон: € {lower_bound:,.0f} — € {upper_bound:,.0f}\n(Середня: € {final_price:,.0f})".replace(',', ' '),
            fg="#27ae60",
            font=("Segoe UI", 18, "bold")
        )

        # Формування статистики під ціною
        status_info = f"Ринковий Кластер: {brand_class} ({fuel}) №{c_id} | Точність: ±{np.round(rmse, 3)} log\n"
        status_info += "-" * 60 + "\n"
        status_info += f"📊 Норма для {int(age)} років: {int(norm_m * 1000)} км. "

        if excess_mileage > 0:
            status_info += f"Перепробіг: +{int(excess_mileage * 1000)} км\n"
        else:
            status_info += f"Гаражне збер.: {int(abs(excess_mileage) * 1000)} км\n"

        mileage_impact_percent = (np.exp(p.get('Надлишковий_Пробіг', 0) * excess_mileage) - 1) * 100
        equip_impact_percent = (np.exp(equip_log) - 1) * 100

        status_info += f"📉 Вплив пробігу на ціну: {mileage_impact_percent:+.1f}%\n"
        status_info += f"📈 Націнка за комплектацію та двигун: {equip_impact_percent:+.1f}%\n"

        if warning_msg:
            status_info += f"\n{warning_msg}"
            label_cluster.config(fg="#e67e22")
        elif is_scrap:
            status_info += "\n*Досягнуто мінімальної вартості металобрухту (€500)"
            label_cluster.config(fg="#c0392b")
        else:
            label_cluster.config(fg="#e67e22") # Оранжево-коричневий колір як на твоєму скріншоті

        label_cluster.config(text=status_info, justify="left")

    except Exception as e:
        messagebox.showerror("Помилка", f"Перевірте введені дані.\nДеталі: {str(e)}")

# ==========================================================
# ІНІЦІАЛІЗАЦІЯ UI (ДВОКОЛОНКОВИЙ ДАШБОРД)
# ==========================================================
root = tk.Tk()
root.title("Калькулятор орієнтовної вартості авто")
# Збільшуємо базову висоту з 880 до 950
root.geometry("900x950")
root.configure(bg="#f8f9fa")
# Дозволяємо користувачу (тобі) розтягувати вікно по вертикалі (True),
# але забороняємо по горизонталі (False), щоб не зламати верстку
root.resizable(False, True)

style = ttk.Style()
style.theme_use('clam')
style.configure('TLabel', background="#f8f9fa", font=("Segoe UI", 10), foreground="#2c3e50")
style.configure('TCheckbutton', background="#f8f9fa", font=("Segoe UI", 10), foreground="#34495e")
style.configure('TLabelframe', background="#f8f9fa", bordercolor="#dee2e6")
style.configure('TLabelframe.Label', font=("Segoe UI", 11, "bold"), foreground="#2980b9", background="#f8f9fa")
style.configure('TButton', font=("Segoe UI", 12, "bold"), padding=8)

ttk.Label(root, text="Детальний калькулятор вартості", font=("Segoe UI", 18, "bold"), foreground="#2c3e50").pack(pady=(15, 10))

# ================= 1. БЛОК БАЗОВИХ ХАРАКТЕРИСТИК (ДВІ КОЛОНКИ) =================
frame_specs = ttk.LabelFrame(root, text="Основні параметри автомобіля")
frame_specs.pack(padx=30, fill="x", pady=5, ipadx=10, ipady=5)

def create_input(parent, row, col, label_text, widget):
    ttk.Label(parent, text=label_text).grid(row=row, column=col*2, sticky="w", pady=6, padx=(10, 5))
    widget.grid(row=row, column=col*2+1, sticky="ew", pady=6, padx=(0, 20), ipadx=3)

frame_specs.columnconfigure(1, weight=1)
frame_specs.columnconfigure(3, weight=1)

# Ліва колонка
entry_age = ttk.Entry(frame_specs)
entry_age.insert(0, "10")
create_input(frame_specs, 0, 0, "Вік автомобіля (років):", entry_age)

entry_mileage = ttk.Entry(frame_specs)
entry_mileage.insert(0, "150000")
create_input(frame_specs, 1, 0, "Фактичний пробіг (км):", entry_mileage)

entry_hp = ttk.Entry(frame_specs)
entry_hp.insert(0, "150")
create_input(frame_specs, 2, 0, "Потужність (к.с.):", entry_hp)

entry_owners = ttk.Entry(frame_specs)
entry_owners.insert(0, "1")
create_input(frame_specs, 3, 0, "Кількість власників:", entry_owners)

combo_seller = ttk.Combobox(frame_specs, values=["Приватна особа", "Офіційний дилер"], state="readonly")
combo_seller.current(0)
create_input(frame_specs, 4, 0, "Тип продавця:", combo_seller)

# Права колонка
combo_brand = ttk.Combobox(frame_specs, values=["Масовий", "Преміум", "Люкс/Спорт"], state="readonly")
combo_brand.current(0)
create_input(frame_specs, 0, 1, "Клас бренду:", combo_brand)

combo_fuel = ttk.Combobox(frame_specs, values=["Бензин", "Дизель", "Газ_LPG", "Електро", "Гібрид"], state="readonly")
combo_fuel.current(0)
create_input(frame_specs, 1, 1, "Тип палива:", combo_fuel)

combo_trans = ttk.Combobox(frame_specs, values=["Автомат", "Механіка"], state="readonly")
combo_trans.current(0)
create_input(frame_specs, 2, 1, "Коробка передач:", combo_trans)

combo_body = ttk.Combobox(frame_specs, values=["Хетчбек", "Седан", "Універсал", "Кросовер/Позашляховик", "Купе", "Мінівен", "Інше"], state="readonly")
combo_body.current(1)
create_input(frame_specs, 3, 1, "Тип кузова:", combo_body)

combo_drive = ttk.Combobox(frame_specs, values=["Передній", "Задній", "Повний (4x4)", "Інше"], state="readonly")
combo_drive.current(0)
create_input(frame_specs, 4, 1, "Тип приводу:", combo_drive)

combo_upholstery = ttk.Combobox(frame_specs, values=["Тканина", "Повна шкіра", "Комбінована", "Велюр", "Алькантара", "Інше"], state="readonly")
combo_upholstery.current(0)
create_input(frame_specs, 5, 1, "Матеріал салону:", combo_upholstery)

# ================= 2. БЛОК КОМПЛЕКТАЦІЇ ТА СТАНУ =================
frame_options = ttk.LabelFrame(root, text="Комплектація та стан")
frame_options.pack(padx=30, fill="x", pady=5, ipadx=10, ipady=5)

# ДОДАНО master=root до кожної змінної tk.IntVar
var_service = tk.IntVar(master=root, value=1)
ttk.Checkbutton(frame_options, text="📖 Є повна офіційна сервісна книжка", variable=var_service).grid(row=0, column=0, columnspan=2, sticky="w", pady=4, padx=10)

var_media = tk.IntVar(master=root, value=0)
ttk.Checkbutton(frame_options, text="📱 Цифрова мультимедіа (CarPlay, LCD)", variable=var_media).grid(row=1, column=0, sticky="w", pady=4, padx=10)

var_adas = tk.IntVar(master=root, value=0)
ttk.Checkbutton(frame_options, text="🛡️ ADAS (Камера 360, Мертві зони, Круїз)", variable=var_adas).grid(row=2, column=0, sticky="w", pady=4, padx=10)

var_light = tk.IntVar(master=root, value=0)
ttk.Checkbutton(frame_options, text="💡 Топове світло (Matrix / Full-LED / Laser)", variable=var_light).grid(row=1, column=1, sticky="w", pady=4, padx=10)

var_sport = tk.IntVar(master=root, value=0)
ttk.Checkbutton(frame_options, text="🏁 Спортивний пакет (Підвіска, Сидіння)", variable=var_sport).grid(row=2, column=1, sticky="w", pady=4, padx=10)

var_comfort = tk.IntVar(master=root, value=0)
ttk.Checkbutton(frame_options, text="🛋️ Преміум-комфорт (Пневмо, Панорама, Масаж)", variable=var_comfort).grid(row=3, column=0, columnspan=2, sticky="w", pady=4, padx=10)
# ================= 3. БЛОК КНОПКИ ТА РЕЗУЛЬТАТІВ =================
btn_calc = tk.Button(root, text="Розрахувати ринкову вартість", command=calculate_price,
                     bg="#27ae60", fg="white", font=("Segoe UI", 12, "bold"),
                     activebackground="#2ecc71", activeforeground="white", borderwidth=0, cursor="hand2")
btn_calc.pack(pady=(15, 10), fill="x", padx=150, ipady=6)

frame_result = tk.Frame(root, bg="#ffffff", highlightbackground="#dee2e6", highlightthickness=1)
frame_result.pack(padx=30, fill="both", expand=True, pady=(0, 20))

label_result = tk.Label(frame_result, text="Введіть дані та натисніть розрахувати",
                        font=("Segoe UI", 14), bg="#ffffff", fg="#7f8c8d")
label_result.pack(pady=(15, 5))

# justify="left" вирівняє текст аналітики зліва (як список), але він буде в центрі блоку
label_cluster = tk.Label(frame_result, text="", font=("Segoe UI", 10, "italic"), bg="#ffffff", fg="#e67e22",
                         justify="left", wraplength=700)
label_cluster.pack(pady=(0, 15), padx=10)

print(f"\n✅ Навчено моделей: {len(macro_models)}")
print(f"Кількість спостережень у датасеті: {len(df_clean)}")
root.mainloop()
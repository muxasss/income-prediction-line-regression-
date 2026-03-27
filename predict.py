import joblib
import pandas as pd

try:
    model = joblib.load('model.pkl')
    feature_names = model.feature_names_in_                                                          #реаль.назв.колон.
except Exception as e:
    print(f"Ошибка: Не удалось загрузить модель. {e}")
    exit()

print("\n=== ВВОД ДАННЫХ ДЛЯ ПРОГНОЗА ===")


try:
    rd = float(input("R&D Spend: "))  
    adm = float(input("Administration: "))                                                               #расх.
    mkt = float(input("Marketing Spend: "))  
except ValueError:
    print("Ошибка: Вводите только числа!")
    exit()

# ValueError 

if 'State_Florida' in feature_names or 'State_New York' in feature_names:
    print("\nШтат: 1 - Florida, 2 - New York, 3 - California/Other")
    st_choice = input("Выбор: ")
    if st_choice not in ['1', '2', '3']:
        print("Ошибка: выберите 1, 2 или 3")
        exit()

    
    st_fl = 1.0 if st_choice == '1' else 0.0
    st_ny = 1.0 if st_choice == '2' else 0.0
else:
    
    st_fl = st_ny = 0.0


data = []
for col in feature_names:
    if col == 'R&D Spend':
        data.append(rd)
    elif col == 'Administration':
        data.append(adm)
    elif col == 'Marketing Spend':
        data.append(mkt)
    elif col == 'State_Florida':
        data.append(st_fl)
    elif col == 'State_New York':
        data.append(st_ny)
    else:
        data.append(0) 

   #расх. на исс.и.раз.

input_df = pd.DataFrame([data], columns=feature_names)


prediction = model.predict(input_df)[0]          




print("\n" + "="*40)
print("       РЕЗУЛЬТАТЫ ПРЕДСКАЗАНИЯ")
print("="*40)

summary = input_df.copy()
summary['PREDICTED PROFIT'] = prediction
print(summary.T)

print("="*40)
print(f"ИТОГОВАЯ ПРИБЫЛЬ: {prediction:,.2f} KGS")
print("="*40)
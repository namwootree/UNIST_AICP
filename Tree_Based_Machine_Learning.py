
def my_seed_everywhere(seed):
    random.seed(seed) # random
    np.random.seed(seed) # numpy
    os.environ["PYTHONHASHSEED"] = str(seed) # os
    print(f'고정된 Seed : {seed})

def preprocessing(data):
  # 'DATE'를 칼럼 대신 인덱스로 사용
  data.index = data['DATE']
  data = data.drop(columns='DATE')
  print('DATE'를 칼럼 대신 인덱스로 사용')

  # 결측치 제거 및 모델 학습에 불필요한 칼럼 제거
  data = data.dropna()
  data = data.drop(columns=['winner', 'loser', 'form_date'])
  print('결측치 제거 및 모델 학습에 불필요한 칼럼 제거')

  return data

def normal_wml(df):
  print('\nwml'이 양수면 1 & 음수면 0\n')
  cond_wml = (df['wml']>=0)
  df.loc[cond_wml, 'pos_wml'] = 1
  df.loc[~cond_wml, 'pos_wml'] = 0

  df.drop(columns=['wml'], inplace=True)

  POS_WML = df['pos_wml'].value_counts()
  print(f'Ratio : {POS_WML[0]/POS_WML[1]}')
  print(POS_WML)

  return df

def Roling_Windows(data, window_size, method, model, model_name, plot=True, plot_feature=True):

  # 학습 및 테스트 데이터 정보 & 실제값과 예측값 정보 수집
  result_dict = {
    'TRAIN_START_DATE':[],
    'TRAIN_END_DATE':[],
    'TEST_DATE':[],
    'Actual_POS_WML':[],
    f'{model_name}_PRED_POS_WML':[]
  }

  df_feature = pd.DataFrame()

  # 모델 학습이 종료되는 지점 설정
  end = data.shape[0] - window_size - 1

  for i in tqdm(range(end)):

    # 모델 학습 중지
    if i == end:
      print('Prediction using machine learning has ended.')
      break
    
    # Rolling Fixed Window
    if method == 'Fixed':
      print('*'*50)
      print('\nRolling Fixed Window를 실행합니다\n')
      print('*'*50)
      MODEL = model
      train = data.iloc[0+i:window_size+1+i]

    # Rolling Expanding Window
    elif method == 'Expanding':
      print('*'*50)
      print('\nRolling Expanding Window를 실행합니다\n')
      print('*'*50)
      MODEL = model
      train = data.iloc[0:window_size+1+i]
    
    # 'method' 잘못입력한 경우
    else:
      print("Make sure to set the method to either 'Fixed' or 'Expanding'.")
      break
    
    # 테스트 데이터 설정
    test = data.iloc[[window_size+1+i]]

    # Features와 Target 구분
    X_train = train.drop(columns=['pos_wml'])
    y_train = train['pos_wml']

    X_test = test.drop(columns=['pos_wml'])
    y_test = test['pos_wml'] 

    # 모델 학습 및 추론
    MODEL.fit(X_train, y_train)
    pred_test = MODEL.predict(X_test)

    # 학습 및 테스트 데이터 정보 & 실제값과 예측값 정보 수집
    result_dict['TRAIN_START_DATE'].append(train.index[0])
    result_dict['TRAIN_END_DATE'].append(train.index[-1])
    result_dict['TEST_DATE'].append(test.index[0])

    result_dict['Actual_POS_WML'].append(list(y_test)[0])
    result_dict[f'{model_name}_PRED_POS_WML'].append(pred_test[0])

    # 트리 기반 모델 (사이킷런)의 Feature Importance 정보 수집
    feature_importances = MODEL.feature_importances_
    df_ft_importance = pd.DataFrame(feature_importances, index = X_train.columns).T
    df_ft_importance.index = y_test.index
    df_feature = pd.concat([df_feature, df_ft_importance])

  result = pd.DataFrame(result_dict)
  result.index = df_feature.index

  result = pd.concat([result, df_feature], axis=1)
  result = result.set_index('TEST_DATE')
  
  # 모델 성능 시각화
  if plot == True:
    plot_result(result, method, model_name)
  
  # 모델의 변수 중요도 시각화
  if plot_feature == True:
    plot_feature_importances(result)
    print()

  # 학습 및 테스트 데이터 정보, 실제값과 예측값 정보, 변수 중요도 정보
  return result

# 모델 성능 시각화
def plot_result(data, method, model_name):
  print('\nPerformance results of the model\n')

  # Confusion Matrix & etc
  print(classification_report(data['Actual_POS_WML'], data[f'{model_name}_PRED_POS_WML']))

  # Accuracy, Precision, Recall, F1 Score
  accuracy = accuracy_score(data['Actual_POS_WML'], data[f'{model_name}_PRED_POS_WML'])
  precision = precision_score(data['Actual_POS_WML'], data[f'{model_name}_PRED_POS_WML'])
  recall = recall_score(data['Actual_POS_WML'], data[f'{model_name}_PRED_POS_WML'])
  F1_score = f1_score(data['Actual_POS_WML'], data[f'{model_name}_PRED_POS_WML'])

  print(f'\naccuracy : {accuracy}')
  print(f'\nprecision : {precision}')
  print(f'\nrecall : {recall}')
  print(f'\nF1-Score : {F1_score}\n')
  
  # Visualization Confusion Matrix
  confusion = confusion_matrix(data['Actual_POS_WML'], data[f'{model_name}_PRED_POS_WML'])
  sns.heatmap(confusion, annot=True, fmt='g')
  plt.title(f'Performance results ({method}) of the {model_name} model')
  
  plt.show()

# Feature Importance 시각화
def plot_feature_importances(data):

  data = data[['mvol_cum6', 'mvol_t_1', 'mvol_t_2', 'mvol_t_3',
       'mvol_t_4', 'mvol_t_5', 'mvol_t_6', 'cum_loser', 'cum_winner',
       'cum_loser_t_2_4', 'cum_winner_t_2_4', 'cum_loser_t_5_8',
       'cum_winner_t_5_8', 'cum_loser_t_9_12', 'cum_winner_t_9_12']]

  print('\nFeature Importance of the model\n')

  # 날짜 별 Feature Importance 변화 추이
  plt.figure(figsize=(15, 5))
  for col in data.columns:

    sns.lineplot(data=data,
                x=data.index,
                y=col,
                label=col,
                alpha=0.3)
    
    plt.title('Feature importances using MDI')
    plt.xlabel('DATE')
    plt.ylabel('Mean decrease in impurity')
    plt.xticks([data.index[i] for i in range(0,len(data.index), 12)])
    plt.tick_params(axis='x',
                      direction='out',
                      labelrotation=45,
                      length=1,
                      pad=10,
                      labelsize=5,
                      width=0.1)
  plt.show()

  # 각 변수 별 Feature Importance 통계값
  MEAN = display_feature_importance(data, method='mean')
  MAX = display_feature_importance(data, method='max')
  MIN = display_feature_importance(data, method='min')

  plt.figure(figsize=(15, 5))
  sns.barplot(data=MEAN,
              y=MEAN.index,
              x='mean_Feature_Importance')

  plt.show()
  print()

  FEATURE_IMPORTANCE = pd.concat([MEAN, MAX, MIN], axis=1)
  display(FEATURE_IMPORTANCE)

# 각 변수 별 Feature Importance 통계값
def display_feature_importance(data, method):

    data = data[['mvol_cum6', 'mvol_t_1', 'mvol_t_2', 'mvol_t_3',
       'mvol_t_4', 'mvol_t_5', 'mvol_t_6', 'cum_loser', 'cum_winner',
       'cum_loser_t_2_4', 'cum_winner_t_2_4', 'cum_loser_t_5_8',
       'cum_winner_t_5_8', 'cum_loser_t_9_12', 'cum_winner_t_9_12']]
       
    feature_dict = {}
    for col in data.columns:
      if method=='mean':
        feature_dict[col]=[]
        feature_dict[col].append(data[col].mean())

      if method=='max':
        feature_dict[col]=[]
        feature_dict[col].append(data[col].max())
      
      if method=='min':
        feature_dict[col]=[]
        feature_dict[col].append(data[col].min())
    
    feature_df = pd.DataFrame(feature_dict).T
    feature_df.columns = [method+'_'+'Feature_Importance']
    feature_df=feature_df.sort_values(by=method+'_'+'Feature_Importance', ascending=False)

    return feature_df

def slice_data(data, num, method, model_name):

  division = int(data.shape[0]/num)
  
  for i in range(num):

    if i == num-1:
      sliced_data = data.iloc[i*division:]

    else:
      sliced_data = data.iloc[i*division:(i+1)*division]

    start = sliced_data.index[0]
    end = sliced_data.index[-1]

    print()
    print('*'*50)
    print(f'\nDATE : {start} ~ {end}\n')
    print('*'*50)
    print()
    
    plot_result(sliced_data, method, model_name)
    plot_feature_importances(sliced_data)

if __name__=="__main__":
  print("2023년 UNIST AICP TEAM : UNIST 동학 개미")
  print('\nTree_Based_Machine_Learning.py : 트리 기반 머신러닝 모델 비교를 위한 모듈\n')
  print('코드 작성자 : 권남우(팀장)')
  print('Source Code : https://github.com/namwootree/UNIST_AICP')
  print('Thank You')

  # Load Library
  import pandas as pd
  import numpy as np

  from scipy import stats 
  from tabulate import tabulate

  import matplotlib.pyplot as plt
  import seaborn as sns

  import random
  import os
  from tqdm.notebook import tqdm

  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import classification_report
  from sklearn.metrics import accuracy_score # 정확도
  from sklearn.metrics import precision_score # 정밀도
  from sklearn.metrics import recall_score # 재현율
  from sklearn.metrics import f1_score # F1-Score
  from sklearn.metrics import confusion_matrix

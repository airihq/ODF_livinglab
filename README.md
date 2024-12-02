# ODF_livinglab
2024년도 오픈데이터포럼 리빙랩에 사용된 소스코드 및 자료입니다.

## 환경구축

**selenium Chrome Driver**
https://developer.chrome.com/docs/chromedriver/get-started?hl=ko

**python**
- numpy
- pandas
- matplotlib
- scikit-learn
- scipy
- tqdm
- streamlit
- plotly
- dash
- pyproj
- opencv-python
- selenium


# 전문가용 분석도구 실행 방법

1. Dash 앱 실행
```
python [08]interactive_with_dash.py
```

2. 앱 접속

http://localhost:8050

3. 사용법

왼쪽 패널: 사고/공사 지도.
- 상단의 시간 범위에 따라 사고, 공사 지점을 표시합니다.
- 공사를 클릭하면 우측에 공사 분석을 진행합니다.

오른쪽 패널 1: 공사/사고 데이터 표
- 좌측 상단 시간 범위에서의 공사와 사고에 대한 정보를 제공합니다.

오른쪽 패널 2: 공사 분석
- 좌측 지도에서 선택한 공사를 기준으로 근처 지역, 근처 시간의 사고/공사의 수를 분석합니다.
- 우측 상단에서 분석에 대한 설정을 변경할 수 있습니다. 


# 일반용도 위험도 지도 실행 방법

1. Dash 앱 실행
```
python "[11]visualize_risk_of_acci.py"
```

2. 앱 접속

http://localhost:8050

3. 사용법

왼쪽 패널: 위험도 지도.
- 상단에서 설정한 시간에 따라 지정된 지점의 위험도를 시각화합니다.
- 별표로 표시된 지점은 오른쪽 패널의 위험도 분석의 대상이 됩니다. 

상단 오른쪽 패널: 예측된 위험도 / 실제 사고빈도 비교
- 해당 년월을 포함하여 이전 1년간 위험도와 사고빈도의 추세를 보여줍니다.

하단 오른쪽 패널: 위험도 예측에 사용된 특징 분석
- 해당 년월을 포함하여 이전 1년간 위험도를 예측하는데 사용된 특징 값들을 나열합니다.



# 이하 각 코드 설명




# [01]crawling_accidents_points.ipynb
- TAAS에서 사고 지점 이미지를 긁어오는 크롤링 코드입니다.
- selenium 환경 설정이 필요합니다.

# [02]visualization.ipynb
- [01]에서 긁은 정보를 지도 이미지와 겹쳐서 보는 코드

# [03]circle_detect_and_visual.ipynb
- circle detection을 이용해서 시각화하는 그림을 그렸음.
- circle detection 시험 및 그림용

# [04]to_GCS_and_make_csv.ipynb
- circle detction한 결과를 바탕으로 점의 중심을 utm-k와 경도/위도 정보로 변환하고 이를 csv 파일로 저장했음.

# [05]streamlit_map.py
# [06]streamlit_table_*.py
- streamlit으로 만들어본 시각화 코드
- streamlit을 사용하려고 고민할 때 만들어본 코드라 기능이 덜 구현되어 있음.

# [07]statistics.ipynb
- 통계 그림과 선택된 공사 주변을 그리는 노트북입니다.
- 공사 지점들을 순회하면서 그 주변에 사고가 있었나, 다른 공사는 있었나 반복해서 그리는 코드.

# [08]interactive_with_dash.py
- plotly와 dash를 이용하여 시간에 따른 사고/공사 지점 시각화
- 전문가용 시각화
- 실행방법
```
python "[08]interactive_with_dash.py"
```

# [09]find_large_construction.py
- 매우 인접한 공사들을 하나로 합치고 대규모 레이블을 부여합니다.
```
python "[09]find_large_construction.py"
```

# [A]make_risky_points.py
- 위험 관심지를 지도로 찍어서 파일로 반환함.
```
python "[A]make_risky_points.py"
```

# [10]calculate_risk.ipynb
- [A]에서 위험 관심지에 따라 위험도를 계산하고 .csv 파일로 반환합니다.
- 랜덤포레스트와 SVM와 같은 머신러닝 기법을 사용합니다.

# [11]visualize_risk_of_acci.py
- [10]에서 계산한 위험도 정보를 시각화합니다.








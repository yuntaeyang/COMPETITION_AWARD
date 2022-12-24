# 날씨빅데이터콘테스트 
# 한국기상산업기술원장상(3위, 장려상)

데이터 전처리1: 1일 뒤 예측을 위한 데이터 전처리(데이터 병합, 결측치 대체, 다른지역 구분, 타겟 설정 등)

데이터 전처리2: 2일 뒤 예측을 위한 데이터 전처리(데이터 병합, 결측치 대체, 다른지역 구분, 타겟 설정 등)

학습 날씨 데이터 전처리1: 1일 뒤 예측을 위한 학습데이터 기간의 예보데이터 전처리 (1일 뒤 예보 활용, 시간단위를 하루단위로 변경)

학습 날씨 데이터 전처리2: 2일 뒤 예측을 위한 학습데이터 기간의 예보데이터 전처리 (2일 뒤 예보 활용, 시간단위를 하루단위로 변경)

테스트 날씨 데이터 전처리1: 1일 뒤 예측 테스트데이터 기간의 예보데이터 전처리 (1일 뒤 예보 활용, 시간단위를 하루단위로 변경)

테스트 날씨 데이터 전처리2: 2일 뒤 예측 테스트데이터 기간의 예보데이터 전처리 (2일 뒤 예보 활용, 시간단위를 하루단위로 변경)

1일뒤 예측 모델: 1일 뒤 예측을 위한 모델링(Train 데이터 이상치 제거, TabNet 활용)

2일뒤 예측 모델: 2일 뒤 예측을 위한 모델링(Train 데이터 이상치 제거, TabNet 활용)

프로젝트 요약

2016년 ~ 2020년 경상도 지역의 산사태 발생여부 데이터 와 임상도*토양도 데이터, 행정동 경계 데이터, 날씨 예보 데이터를 활용하여 산사태 예측 모델을 구축한다. 

Shp 파일형태의 지리정보 데이터를 공간조인하고, csv파일과  병합하였다.  결측치 대체, 이상치 제거, 이름은 같지만 다른 지역 구분, 발생하지 않은 지역 구분, 시간 단위의 날씨예보 를 하루 단위로 볼 수 있도록 전처리하여 분석 데이터 셋을 구축하였다. 

정형 데이터에 강한 딥러닝 모델인, TabNet을 산사태 예측 모델로 사용하였다. TabNet의 고유한 특징인, 순차적인 어텐션을 통한 피처 선택을 활용하여 다르게 resampling 된 6개의 데이터를 TabNet에 학습시켰다. 이를 통해 다르게 피처 선택이 되고 다르게 학습된 6개의 TabNet 모델을 생성하였다. 다르게 학습된 6개의 TabNet 모델을 앙상블하여 최종결과값을 도출하였다. 
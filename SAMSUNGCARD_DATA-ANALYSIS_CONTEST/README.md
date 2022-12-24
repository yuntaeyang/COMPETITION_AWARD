삼성카드 데이터 분석 공모전 : 고객 피드백 분류 모델 개발

# data
* 삼성카드 고객의 서비스 리뷰 데이터

├ Index : No1, No2

├ 발화

├ 발화 후보: 발화1, 발화2, 발화3

├ 발화 우선순위: 발화1 우선순위, 발화2 우선순위, 발화3 우선순위

├ Label : 우선순위 가장 높은 것 선택
* 우선순위 : 1-44순위   
* ex) 1: 칭찬> 고객서비스 > 상담원 , 25: 불만> 삼성카드>혜택

# 전처리
* 중복 발화 제거
* 불필요한 변수 제거
* 특수문자 및 불필요한 문자제거

# Modeling
KoElectra - small - v2
* See Github for more details.(https://github.com/monologg/KoELECTRA)
* optimizer : AdamW

![image](https://user-images.githubusercontent.com/90027932/178411352-71868fdd-d7db-47a4-9046-a3de2250448c.png)

# result 
* 1위

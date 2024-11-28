# ODF_livinglab
2024년도 오픈데이터포럼 리빙랩에 사용된 소스코드 및 자료입니다.

# [01]crawling_accidents_points
- TAAS에서 사고 지점 이미지를 긁어오는 크롤링 코드입니다.
- selenium 환경 설정이 필요합니다.

# [02]visualization
- [01]에서 긁은 정보를 그냥 지도 이미지와 겹쳐서 보기만 함

# [03]circle_detect_and_visual
- circle detection을 이용해서 시각화하는 그림을 그렸음.
- 보여주기용

# [04]to_GCS_and_make_csv
- circle detction한 결과를 바탕으로 점의 중심을 utm-k와 경도/위도 정보로 변환하고 이를 csv 파일로 저장했음.

# [05]streamlit_map.py
# [06]streamlit_table_*.py
- streamlit 보여주기용 시각화임

# [07]statistics
- 통계 그림과 선택된 공사 주변을 그리는 노트북입니다.
- 공사 지점들을 순회하면서 그 주변에 사고가 있었나, 다른 공사는 있었나 반복해서 그리는 코드.
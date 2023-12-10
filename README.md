# Overview

이력서, 채용 공고 및 지원 히스토리 데이터를 활용하여 구직자에게 맞춤화된 채용 공고를  
자동으로 추천할 수 있는 추천시스템 알고리즘 개발

# Running a Model
1. **model_Cornac_recsys.ipynb** 실행
2. **Recsys_preprocessing_V3.ipynb** cease용 전처리 파일 실행
3. **model_cease_recsys.ipynb** 실행
4. Top-20결과 파일을 **ensemble** 폴더내 **preds** 내로 결과를 옮기기
5. **ensemble** 폴더의 **ensemble.ipynb** 실행후 최종 제출용 Top-5파일 생성

* * *
### References
- [Utils](https://github.com/recommenders-team/recommenders/blob/main/recommenders/datasets/python_splitters.py)
- [Cornac](https://cornac.preferred.ai/)

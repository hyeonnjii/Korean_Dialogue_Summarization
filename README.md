# Korean_Dialogue_Summarization
[NLP/LLM] 한국어 일상 대화 요약

## Summary

### 🛠️ 문제 정의

- **목표**: 학교생활, 직장, 치료, 쇼핑 등 다양한 일상 주제를 포함한 한국어 일상 대화 데이터를 효과적으로 요약하는 모델 개발

### ⚙️ 수행 역할
- 형태소 분석을 통해 추출한 토큰을 데이터셋에 추가
- T5, BART, LLM 모델을 활용한 다양한 실험 
- 4-bit QLoRA 기법 적용하여 모델의 효율성 개선


### 📈 결과 및 직무에 적용할 점
1. 일상 대화 요약 외에도 NLP 및 LLM 프로젝트에 적용할 수 있는 기초 마련
2. Hugging Face에 등록된 한국어에 특화된 여러 NLP 모델과 LLM SOTA 모델을 Fine-tuning 해본 경험
3. 양자화 기법을 통해 LLM 모델 크기를 대폭 줄여 효율성 극대화


<br><br>


$${\color{gray}(※ 아래는 프로젝트의 상세 내용 입니다.)}$$
## 상세

### 📝 데이터 수집
* **데이터셋 설명**
  *  제공되는 데이터셋은 회의, 일상 대화 등 다양한 주제를 가진 대화문과, 이에 대한 요약문을 포함하고 있음
    * Train: 총 12,457건의 2~60턴의 대화문
    * Dev: 499건
    * Test: 250건
    * Hidden-test: 249건

  * 데이터 출처: [한국어 대화 요약문](https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000302/data/data.tar.gz)

---
### 🔍 📊 데이터 전처리 & EDA
* **대화문과 요약문간의 특징 분석**
<div align=center>

  ![image](https://github.com/user-attachments/assets/4682844d-0c78-43c1-a45b-9bcfc84e1eeb)
  ![image](https://github.com/user-attachments/assets/877e4f9e-96b0-423c-9639-e42903b759cd)

</div>


  - 대화문과 요약문이 전체적으로 right-skewed 된 모습 → 길이가 짧은 데이터의 기준이 더 큰 형태
  - 대화문과 요약문 길이가 평균적으로 약 5배 차이


<br>

<div align=center>

 ![image](https://github.com/user-attachments/assets/72b88cb0-de51-43d2-ba2c-a807c074f719)

</div>

```
--------------------------------------------------
7600
Topic: 생일 파티 초대
Dialogue (126 tokens):
#Person1#: 6월 19일은 제 생일입니다. 작은 파티를 계획하고 있어요. 참석하시겠어요?
#Person2#: 생일 축하드려요! 정말 가고 싶지만 지금은 확실하지 않아요. 가능하면 참석하겠습니다. 초대해주셔서 감사합니다.

Summary (81 tokens):
#Person1#이(가) #Person2#를 #Person1#의 생일 파티에 초대합니다. #Person2#는 초대에 감사하나 확신할 수 없습니다.

--------------------------------------------------
```


  - 대화문 대비 요약문 길이는 20%
    * → 실험) 요약문의 최대길이를 정하는 기준으로 세우고 0.2 이상을 outlier로 취급?
      * 이러한 가정 아래, 0.2 보다 큰 ratio를 가지고 있는 대화문과 요약문 행을 확인
      * 결론: 0.2 이상 행들을 확인해본 결과, 기존 대화문 길이가 짧은 경우로 요약문을 outlier로 판단할 근거 충분 X






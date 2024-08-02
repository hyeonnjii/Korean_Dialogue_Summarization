# Korean_Dialogue_Summarization
[NLP/LLM] 한국어 일상 대화 요약

## Summary

### 🛠️ 문제 정의

- **목표**: 학교생활, 직장, 치료, 쇼핑 등 다양한 일상 주제를 포함한 한국어 일상 대화 데이터를 효과적으로 요약하는 모델 개발

### ⚙️ 수행 역할
- 형태소 분석을 통해 corpus에 없는 단어를 추가
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


  - 대화문과 요약문 모두 오른쪽으로 치우친(right-skewed) 분포를 보이며, 이는 길이가 짧은 데이터가 상대적으로 많은 형태임을 나타냄 
  - 대화문과 요약문 길이가 평균적으로 약 5배 차이이가 남


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
    * 이를 바탕으로 요약문의 최대 길이를 정하는 기준으로 0.2를 설정하고, 이 비율을 초과하는 경우를 이상치로 간주하여 가정 후 실험
      * 이러한 가정 하에 0.2 이상의 비율을 가진 대화문과 요약문을 분석한 결과, 기존 대화문 길이가 짧은 경우에 대한 요약문은 이상치로 판단할 충분한 근거가 없음을 확인
  


* **Topic 분석**
  - 총 6,526개의 유니크한 주제
  - 단일 갯수의 주제가 77% 이상
  - 일상 대화와 같이 topic의 단어가 대부분 dialogue에 포함되는 경향은 보이지 X
```
topic
일상 대화      236
쇼핑         188
전화 통화       98
직업 면접       92
음식 주문       85
...
빵 쇼핑         1
착각           1
기업 운영        1
밴드           1
컨퍼런스 센터      1
Name: count, Length: 6526, dtype: int64
```
  <div align=center>
   
   ![image](https://github.com/user-attachments/assets/01d1e8d2-fe1e-4449-8ba4-9d8e4c392d42)

  </div>

* **Special Tokens 추가**
- 해당 데이터셋에는 개인정보가 포함되어 있어, 이를 마스킹하여 제공되어
- 대화문 및 요약문에서 마스킹된 값들을 tokenizer 에 special token으로 포함시킴
  ```
  'special_tokens': ['#Person3#', '#Person1#', '#CardNumber#', '#Person4#',
  '#SSN#', '#CarNumber#', '#Address#', '#DateOfBirth#', '#Person5#', '#PassportNumber#',
  '#PhoneNumber#', '#Person7#', '#Email#', '#Person6#','#Person#', '#Person2#']
  ```
* **잘못 입력된 요약문 수정**
  - 요약문이 잘못 입력된 사례를 발견하여 해당 행 삭제
   ```
   --------------------------------------------------
  7528
  Topic: 정보 얻기
  Dialogue (134 tokens):
  #Person1#: 보고서 작성을 시작했나요? 
  #Person2#: 정보를 얻는 데 어려움을 겪고 있어요. 
  #Person1#: 그건 쉬워요. 원하는 정보를 얻기 위해 인터넷만 검색하면 돼요. 
  #Person2#: 아, 그 생각을 못했네요.
  
  Summary (108 tokens):
  #Person1#은 자신이 MP3 플레이어와 스테레오 헤드폰을 가지고 있기 때문에 #Person1#이 주변에서 가장 멋진 하이테크 스터드라고 생각한다. #Person2#는 그것을 보고 싶어한다.
  
  --------------------------------------------------
   ```

* **결론**
    - 대화문과 요약문의 평균 길이를 기준으로 input의 `max_length`와 output의 `max_length`를 각각 1024, 256 으로 설정
    - 주제의 불균형성, pre-trained corpus 사용한다는 점 -> 모델이 학습하는 단어의 양을 늘리기 위한 방법 계획
      * 1) 형태소 분석기(Okt(), Kiwi)를 사용하여 Tokenizer에 없는 단어를 추가하여 학습
      * 2) 적은 주제를 가진 데이터를 활용하여 Data Augementation 진행 예정(Cohere API 사용)

<br>
---

### 🧠 모델링
### 평가 지표
* ROUGE-1-F1, ROUGE-2-F1, ROUGE-L-F1 세 가지 지표 활용
  - ROUGE는 참조 요약본과의 단어 겹침 정도를 기반으로 평가
  - 3개의 정답 요약 문장과 예측된 요약 문장을 비교하여 평균 점수를 산출
  - 한국어 특성 반영을 위해 형태소 분석을 통해 문장 토큰화를 진행하여 ROUGE score를 산출
 
### 모델 구성 및 학습

-** Pre-trained된 모델을 Fine-tuning 하는 방법으로 모델 학습 진행**

- **1) Korean NLP 모델**
  - KoBART(`EbanLee/kobart-summary-v3`, `gogamza/kobart-summarization`)
  - KoT5(`paust/pko-t5-base`, `eenzeenee/t5-small-korean-summarization`, `psyche/KoT5-summarization`)



- **2) LLM 모델**
  - 4-bit QLoRA 기법 적용하여 메모리 효율화 진행
  - [호랑이 한국어 LLM 리더보드](https://wandb.ai/wandb-korea/korean-llm-leaderboard/reports/-LLM---Vmlldzo3MzIyNDE2?accessToken=95bffmg3gwblgohulknz7go3h66k11uqn1l3ytjma1uj3w0l0dwh1fywgsgpbdyy&nw=nwusercchyun#%ED%98%B8%EB%9E%91%EC%9D%B4-%EB%A6%AC%EB%8D%94%EB%B3%B4%EB%93%9C%EC%9D%98-%EA%B8%B0%EB%8A%A5%EB%93%A4-%F0%9F%90%AF)를 참조하여 아래 LLM 모델 선정
    - Solar
    - KLULLM

 <div align=center>

 
 **1) Korean NLP 모델**

```

   - T5의 경우, 학습이 제대로 이루어지지 않는 문제 발생으로 KoBART 모델만 사용하여 성능 평가 진행
```

| 모델                             | Score    | ROUGE-1  | ROUGE-2  | ROUGE-L  |
|----------------------------------|----------|----------|----------|----------|
| **[KOBART]**                     |          |          |          |          |
| gogamza/kobart-summarization     | Public   | 41.3251  | 0.5070   | 0.3117   |
|                                  | Private  | 38.9090  | 0.4921   | 0.2831   |
| EbanLee/kobart-summary-v3        | Public   | 43.3165  | 0.5210   | 0.3375   |
|                                  | Private  | 40.0959  | 0.4983   | 0.3001   |

<br>

**1-1) 형태소 분석기 사용하여 Tokenizer에 새로운 Token 추가**

```
  - 오히려 성능이 떨어지는 결과
  - 새로운 형태소 추가로 기존의 학습된 표현과 충돌, 불필요한 노이즈를 증가시켜 모델이 새로운 토큰을 제대로 이해하지 못했다고 판단
```
 
| 모델                             | Score    | ROUGE-1  | ROUGE-2  | ROUGE-L  |
|----------------------------------|----------|----------|----------|----------|
| EbanLee/kobart-summary-v3        | Public   | 43.3165  | 0.5210   | 0.3375   |
|                                  | Private  | 40.0959  | 0.4983   | 0.3001   |
| **[+ Token 추가]**            |          |          |          |          |
| okt()                            | Public   | 40.5300  | 0.5008   | 0.3041   |
|                                  | Private  | 38.1289  | 0.4862   | 0.2738   |
| Kiwi()                           | Public   | 40.8384  | 0.5048   | 0.3073   |
|                                  | Private  | 39.1610  | 0.4959   | 0.2866   |

<br>

**2) LLM 모델**

```
  - 모델 생성 결과에는 문제가 없으나 평가 점수가 낮게 나옴
  - 해당 이유에 대해서는 기존 데이터가 번역체의 요약문이었기 때문에 해당 LLM에는 반영되지 않아 오히려 평가 점수가 낮았다고 판단
```

| 모델                             | Score    | ROUGE-1  | ROUGE-2  | ROUGE-L  |
|----------------------------------|----------|----------|----------|----------|
| **[LLM]**                        |          |          |          |          |
| KULLM3                           | Public   | 38.0259  | 0.4697   | 0.2821   | 0.3890   |
|                                  | Private  | 36.8688  | 0.4637   | 0.2652   | 0.3771   |


<br>

</div>

![image](https://github.com/user-attachments/assets/90bf17df-393b-4c5d-8624-a692c80df3df)
Test20.

<br>

![image](https://github.com/user-attachments/assets/af01262c-f7d8-454e-8e2b-63d16e6552df)

Test59.


---
### 🎯 결과 및 보완점
- 해당 KULLM3 모델은 번역투를 반영하지 못하여 오히려 큰 모델임에도 불구하고 평가 점수가 낮게 나왔다는 점
- 모델을 양자화하여 가지고 오는데에도 큰 메모리 차지로 데이터 증강 기법이나, 번역투 변경하는 방법론을 같이 사용할 수 없었다는 자원적의 한계가 아쉬움
- 제한된 환경에 맞게 가장 최선의 성능을 뽑아내는 것 역시 중요한 역량이라는 것을 다시 한번 더 깨닫는 계기가 됨






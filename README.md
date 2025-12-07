터미널에서

1. 가상환경 비활성화
   deactivate

2. 기존 가상환경 삭제
   rm -rf ./env

3. 3.11. 버전으로 가상환경 생성
   python3.11 -m venv ./env

4. 가상환경 활성화
   source ./env/bin/activate
   .\env\Scripts\activate
5. 다시 pyhon 버전 확인
   python --version

.\env\Scripts\python.exe -m pip install -r requirements.txt

.\env\Scripts\Activate.ps1
streamlit run Home.py

pip install yfinance openai --upgrade

huggingface 모델 주소:

- 일반 모델 -
  https://huggingface.co/mistralai/Mistral-7B-v0.1

- instruct 모델 (instruction[지시]를 따르도록 세밀하게 조정됨)-
  https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1

Mistral AI document 주소:
https://docs.mistral.ai/category/large-language-models
https://docs.mistral.ai/models/#chat-template

#RAG

Prompt
Question : {q}
Context : {docs}

Question: 독도는 몇 개의 섬과 암초로 구성되어 있나?
Answers: 89|91(o)|92|90

Question: 서도와 동도 중 면적이 더 큰 곳은 어디인가?
Answers: 동도|서도(o)|삼봉도|일출봉

Question: 일본어권에서 독도를 부르는 현재의 이름은 무엇인가?
Answers: 마쓰시마|다케시마(o)|우산도|관음도

Question: 2005년에 도입된 입도 신고제는 어느 섬에 대해 도입되었는가?
Answers: 서도|동도(o)|일출봉|삼봉도

Question: 독도 천연보호구역의 명칭이 바뀐 해는 언제인가?
Answers: 1982|1999(o)|2006|1985

Question: 1954년 독도 대첩에서 발생한 주요 결과는 무엇인가?
Answers: 일본 순시선이 대다수 퇴각했다(o)|독도가 일본에 편입되었다|전투가 끝나고 즉시 휴전되었다|양측이 협상을 통해 합의를 봤다

Question: 동도의 최고봉 이름은 현재 어떤 고시 이름으로 불리는가?
Answers: 대한봉|우산봉(o)|일출봉|상장군봉

Question: 독도는 어떤 지질학적 기원으로 형성된 섬인가?
Answers: 화산섬(o)|모래섬|빙하섬|산호섬

Question: 한국 측의 영유권 주장 근거로 자주 인용되는 고전 문헌 중 하나는 무엇인가?
Answers: 삼국사기(o)|세종실록지리지|태정관 지령|SCAPIN 제677호

Question: 울릉도에서 독도까지의 거리는 대략 몇 리인가?
Answers: 100리|200리(o)|300리|400리

st.markdown("출제 준비가 되었습니다. <br>", unsafe_allow_html=True)
choiceLevel = st.selectbox(
"Please select the difficulty level of the quiz.",
(
"난이도를 선택하세요 !!",
"Easy",
"Hard",
),
)

    if choiceLevel != "난이도를 선택하세요 !!":
        st.write(f"Selected Difficulty: {choiceLevel}")
        response = run_quiz_chain(docs, topic if topic else file.name)

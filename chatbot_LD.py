# (변경후) 레벤슈타인 거리 기반 챗봇(Chatbot based on Levenshtein Distance)_2025.06.05 작성자: 김형석 (학번 202430896)
# -----------------------------------------------------------------------------------------------------------------
# 레벤슈타인 거리(Levenshtein Distance) 기반에서는 입력 문장을 벡터화할 필요가 없습니다. 이유는 다음과 같습니다.
# 레벤슈타인 거리는 '문자 단위의 편집 거리(삽입, 삭제, 교체)'를 직접 계산하므로, 벡터화가 불필요합니다.
# 즉, "학교에 간다"와 "학교로 간다"의 레벤슈타인 거리는 '에' → '로'의 차이이므로 거리는 1입니다.
# 이는 문자 혹은 문자열 비교에 기반하므로 수치화하는 벡터화가 요구되지 않습니다.
# -----------------------------------------------------------------------------------------------------------------

import pandas as pd  # CSV 파일을 처리하기 위한 라이브러리
# from sklearn.feature_extraction.text import TfidfVectorizer  # (삭제) TF-IDF 벡터화용 모듈
# from sklearn.metrics.pairwise import cosine_similarity       # (삭제) 문장 간 코사인 유사도 계산용

# 레벤슈타인 거리(편집 거리)를 계산하는 함수 정의
def levenshtein_distance(a, b):        # 두 문자열 a, b 사이의 편집 거리를 계산
    a_len = len(a)                     # 문자열 a의 길이
    b_len = len(b)                     # 문자열 b의 길이
    # if a == "": return b_len         # (삭제)
    # if b == "": return a_len         # (삭제)
        
    # 2차워표 (a_len + 1, b_len + 1) 준비하기 --- (※1) 
    # materix 초기화의 예 : [[0, 1, 2, 3], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]
    #        서 울 시
    #    [0, 1, 2, 3]
    # 서 [1, 0, 1, 2]
    # 울 [2, 1, 0, 1]
    # 시 [3, 2, 1, 0]

    # matrix = [[] for i in range(a_len+1)] # 리스트 컴프리헨션을 사용하여 1차원 초기화
    dp = [[0] * (b_len + 1) for _ in range(a_len + 1)]  # 동적 계획법(Dynamic Programming)용 2차원표 배열 초기화
    for i in range(a_len + 1):          # 첫 열을 0으로 초기화 (삭제 연산)
      # matrix = [i] = [0 for j in range(b_len+1)] # 리스트 컴프리헨션을 사용하여 2차원 초기화
        dp[i][0] = i
    for j in range(b_len + 1):          # 첫 행을 0으로 초기화 (삽입 연산)
        dp[0][j] = j
    # 0일때 초기값을 설정
    for i in range(1, a_len + 1):       # a문자열의 문자 순회
      # matrix [i][0] = i               # (삭제)
        for j in range(1, b_len + 1):   # b문자열의 문자 순회
          # matrix [0][j] = j           # (삭제)
            if a[i - 1] == b[j - 1]:    # 문자가 같으면 비용 없음
                cost = 0
            else:
                cost = 1                 # 문자가 다르면 교체 비용 1

            # 삽입, 삭제, 교체 중 최소 비용 선택
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # 삭제 연산
                dp[i][j - 1] + 1,        # 삽입 연산
                dp[i - 1][j - 1] + cost  # 교체 연산
            )

    return dp[a_len][b_len]            # 최종 편집 거리 반환

# 챗봇 클래스 정의
class LevenshteinChatBot:
    def __init__(self, filepath):        # 생성자, CSV 경로를 입력받음
        self.questions, self.answers = self.load_data(filepath)  # 질문과 답변 리스트 불러오기
        # self.vectorizer = TfidfVectorizer()                                    # (삭제) TF-IDF 벡터라이저 초기화
        # self.question_vectors = self.vectorizer.fit_transform(self.questions)  # (삭제) 질문들을 벡터화

    def load_data(self, filepath):       # CSV 파일에서 질문과 답변을 읽어오는 함수
        data = pd.read_csv(filepath)     # CSV 파일 로드
        questions = data['Q'].tolist()   # 'Q' 열의 질문들을 리스트로 변환
        answers = data['A'].tolist()     # 'A' 열의 답변들을 리스트로 변환
        return questions, answers        # 질문과 답변 반환

    def find_best_answer(self, input_sentence):  # 사용자 입력에 대한 최적 답변 찾기
        # input_vector = self.vectorizer.transform([input_sentence])             # (삭제) 입력 문장을 벡터화
        # similarities = cosine_similarity(input_vector, self.question_vectors)  # (삭제) 벡터화한 질문과 학습질문들과 코사인 유사도 계산
        # best_match_index = similarities.argmax()                               # (삭제) 가장 유사한 인덱스 찾기

        distances = [levenshtein_distance(input_sentence, q) for q in self.questions] # 입력 문장과 각 질문 간 거리 계산
        best_match_index = distances.index(min(distances))                            # 가장 작은 거리의 인덱스 추출 (가장 유사한 질문)
        return self.answers[best_match_index]                                         # 해당 인덱스의 답변 반환

# 챗봇 실행 시작
filepath = 'ChatbotData.csv'             # 학습 데이터 파일 경로
chatbot = LevenshteinChatBot(filepath)   # 챗봇 인스턴스 생성

while True:                              # 무한 반복으로 사용자와의 대화 유지
    input_sentence = input('You: ')      # 사용자 입력 받기
    if input_sentence.lower() == '종료': # '종료' 입력 시 반복 종료
        break
    response = chatbot.find_best_answer(input_sentence)  # 최적 응답 생성
    print('Chatbot:', response)                          # 응답 출력



"""
# (변경전) TF-IDF와 Consine Similarlity 기반 챗봇
# -----------------------------------------------------------------------------------------------------------------

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleChatBot:
    def __init__(self, filepath):
        self.questions, self.answers = self.load_data(filepath)
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.questions)  # 질문을 TF-IDF로 변환하여 벡터화함.

    def load_data(self, filepath):
        data = pd.read_csv(filepath)
        questions = data['Q'].tolist()  # 질문열만 뽑아 파이썬 리스트로 저장
        answers = data['A'].tolist()    # 답변열만 뽑아 파이썬 리스트로 저장
        return questions, answers

    def find_best_answer(self, input_sentence):
        input_vector = self.vectorizer.transform([input_sentence])
        similarities = cosine_similarity(input_vector, self.question_vectors) # 벡터화한 질문을 전체 질문 중 코사인 유사도 값들을 리스트에 저장
        
        best_match_index = similarities.argmax()   # 유사도 값이 가장 큰 값의 인덱스를 반환
        return self.answers[best_match_index]

# CSV 파일 경로를 지정하세요.
filepath = 'ChatbotData.csv'

# 간단한 챗봇 인스턴스를 생성합니다.
chatbot = SimpleChatBot(filepath)

# '종료'라는 단어가 입력될 때까지 챗봇과의 대화를 반복합니다.
while True:
    input_sentence = input('You: ')
    if input_sentence.lower() == '종료':
        break
    response = chatbot.find_best_answer(input_sentence)
    print('Chatbot:', response)
    
# -----------------------------------------------------------------------------------------------------------------    
"""

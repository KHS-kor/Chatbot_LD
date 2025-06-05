from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TfidfVectorizer 객체 생성
vectorizer = TfidfVectorizer()

# 한국어 문장들
doc1 = "나는 학교에 갔다"
doc2 = "나는 영화관에 갔다"
# sentence2 = "너는 그제 밥을 지었습니다."

# 문장들을 벡터화
tfidf_matrix = vectorizer.fit_transform([doc1, doc2]) # Scipy의 CSR(Compressed Sparse Row) 형식의 희소 행렬(sparse matrix), 이 형식은 2차원 배열의 데이터를 메모리 효율적으로 저장하기 위해 0이 아닌 값들만을 저장  

# print( tfidf_matrix.toarray())

# 문장1과 문장2의 코사인 유사도 계산
cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]) # tfidf_matrix[0:1]는 첫 번째 행을 선택하는 슬라이싱
print(cosine_sim, type(cosine_sim))

print(f"문장 1: {doc1}")
print(f"문장 2: {doc2}")
print(f"두 문장의 코사인 유사도: {cosine_sim[0][0]}")
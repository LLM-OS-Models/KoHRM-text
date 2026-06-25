---
license: other
language:
- ko
pretty_name: Korean Current-Law Bar Exam Hard SFT 1000
task_categories:
- question-answering
- text-classification
tags:
- law
- korean-law
- bar-exam
- sft
- synthetic
- public-sector
configs:
- config_name: questions
  data_files:
  - split: train
    path: data/questions.csv
---

# Korean Current-Law Bar Exam Hard SFT 1000

대한민국 현행 법령을 기준으로 만든 변호사시험 선택형 **고난도 스타일** SFT 데이터 1,000문항입니다.

초기 직접 조문확인형 생성본은 실제 제14ㆍ15회 변호사시험보다 쉬워서, 이 버전은 다음 기준으로 다시 만들었습니다.

- `ㄱ/ㄴ/ㄷ/ㄹ` 복합정오형 중심
- `甲/乙/丙`, 검사ㆍ사법경찰관ㆍ행정청ㆍ회사ㆍ소송당사자 등이 등장하는 사례형 비중 확대
- 단순 근거 조문 선택형 제거
- 정답뿐 아니라 각 지문별 O/X 이유와 참고 법령 조문 제공
- 제15회 변호사시험 `data/questions.csv`와 높은 유사도 문항 제외

## Files

- `data/questions.csv`: Hugging Face preview용 메인 CSV입니다.
- `sft/train.jsonl`: `messages` 형식 SFT용 JSONL입니다.
- `metadata/qa_report.json`: 생성 수량, 난도 관련 분포, 제15회 유사도 QA 결과입니다.
- `metadata/comparison_report.json`: 실제 제14ㆍ15회 선택형과 생성본의 길이/사례형/복합정오형 비교입니다.

## Columns

- `question_text`: 문제와 5개 선택지
- `answer`: 정답 번호, `1`부터 `5`
- `answer_text`: 정답 선택지 원문
- `explanation`: 지문별 정오 판단과 근거 조문
- `references_json`: 참고 법령, 조문, 출처 URL, 로컬 경로, git commit, 시행일자
- `prompt`, `response`, `messages_json`: SFT 변환용 텍스트

## Generation Policy

- 기준일: `2026-06-13`
- 법령 원천: `legalize-kr/legalize-kr` 로컬 스냅샷 및 git 이력
- 미래 시행본 제외: `시행일자 <= 2026-06-13`인 버전만 사용
- 기존 기출 복제 금지: `gyung/korean-bar-exam-moj-multiple-choice`의 제15회 문항과 문자열 유사도 `0.72` 이상인 생성 문항 제외
- 문항 유형: 사례형 복합정오, 일반 복합정오, 옳지 않은 설명형

## QA Summary

## Difficulty Comparison

실제 제14ㆍ15회 변호사시험 선택형 300문항과 비교한 주요 지표입니다.

| metric | real rounds 14-15 | generated hard |
|---|---:|---:|
| questions | 300 | 1,000 |
| mean question length | 741.7 | 625.4 |
| median question length | 724.5 | 603.0 |
| case-style markers | 46.0% | 64.0% |
| ㄱ/ㄴ/ㄷ/ㄹ combo markers | 50.0% | 86.0% |
| precedent/legal-rule markers | 98.3% | 72.0% |
| all-correct combo form | 29.7% | 86.2% |
| direct lookup form | 0.0% | 0.0% |

생성본은 실제 시험보다 평균 문항 길이는 짧지만, 사례형ㆍ복합정오형 밀도를 더 높이고 조문+판례 혼합 지문을 별도로 넣어 단순 조문 대조형을 제거했습니다.

```json
{
  "target_count": 1000,
  "row_count": 1000,
  "attempts": 1034,
  "subject_counts": {
    "공법": 270,
    "민사법": 460,
    "형사법": 270
  },
  "question_type_counts": {
    "mixed_statute_precedent_case": 360,
    "case_multi_statement": 280,
    "multi_statement": 220,
    "negative_statement": 140
  },
  "law_counts_top20": {
    "상법": 155,
    "형사소송법": 135,
    "채무자 회생 및 파산에 관한 법률": 111,
    "민법": 105,
    "국회법": 91,
    "형법": 74,
    "민사소송법": 53,
    "행정절차법": 45,
    "헌법재판소법": 32,
    "행정심판법": 30,
    "아동ㆍ청소년의 성보호에 관한 법률": 29,
    "대한민국헌법": 25,
    "성폭력범죄의 처벌 등에 관한 특례법": 24,
    "행정소송법": 21,
    "가족관계의 등록 등에 관한 법률": 19,
    "행정기본법": 19,
    "부동산등기법": 17,
    "국가배상법": 7,
    "형의 실효 등에 관한 법률": 4,
    "특정범죄 가중처벌 등에 관한 법률": 2
  },
  "round15_similarity_threshold": 0.72,
  "round15_max_similarity": 0.2459,
  "rejected_round15_similarity": 0,
  "duplicate_fingerprints": 0,
  "difficulty_note": "Harder v2: case-framed and ㄱㄴㄷㄹ multi-statement questions replace direct lookup questions.",
  "precedent_pool_count": 5000,
  "as_of": "2026-06-13",
  "law_doc_count": 21,
  "article_count": 5247,
  "weighted_article_count": 26648,
  "round15_reference_count": 150,
  "repo_id": "gyung/korean-bar-exam-hard-current-law-precedent-sft-1000",
  "bar_questions_path": "data/bar_exam/hf_multiple_choice/data/questions.csv"
}
```

## Sources

- Existing bar-exam reference dataset: https://huggingface.co/datasets/gyung/korean-bar-exam-moj-multiple-choice
- Ministry of Justice, 제15회 변호사시험 기출문제/정답 공지
  - https://www.moj.go.kr/bbs/moj/150/602397/artclView.do
  - https://www.moj.go.kr/bbs/moj/150/602398/artclView.do
  - https://www.moj.go.kr/bbs/moj/150/602399/artclView.do
  - https://www.moj.go.kr/bbs/moj/151/603464/artclView.do
- Korean statutes: https://github.com/legalize-kr/legalize-kr and https://www.law.go.kr

## License

법령 원문은 대한민국 정부 공공저작물입니다. 데이터셋의 패키징, 생성 코드, 메타데이터는 프로젝트 생성물입니다.

이 데이터는 학습/평가용 법률 교육 데이터이며 법률 자문이 아닙니다.

HF repo: https://huggingface.co/datasets/gyung/korean-bar-exam-hard-current-law-precedent-sft-1000

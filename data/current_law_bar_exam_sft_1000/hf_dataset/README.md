---
license: other
language:
- ko
pretty_name: Korean Current-Law Bar Exam SFT 1000
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

# Korean Current-Law Bar Exam SFT 1000

대한민국 현행 법령을 기준으로 만든 변호사시험 선택형 스타일 SFT 데이터 1,000문항입니다.

이 데이터셋은 법무부 기출문제를 복제하지 않습니다. 기존 `gyung/korean-bar-exam-moj-multiple-choice`의 `data/questions.csv`는 난도와 과목 분포 참고 및 제15회 중복 방지 기준으로만 사용했습니다.

## Files

- `data/questions.csv`: Hugging Face preview용 메인 CSV입니다.
- `sft/train.jsonl`: `messages` 형식 SFT용 JSONL입니다.
- `metadata/qa_report.json`: 생성 수량, 과목 분포, 제15회 유사도 QA 결과입니다.

## Columns

- `question_text`: 문제와 5개 선택지
- `answer`: 정답 번호, `1`부터 `5`
- `answer_text`: 정답 선택지 원문
- `explanation`: 정답 근거와 오답 판단 이유
- `references_json`: 참고 법령, 조문, 출처 URL, 로컬 경로, git commit, 시행일자
- `prompt`, `response`, `messages_json`: SFT 변환용 텍스트

## Generation Policy

- 기준일: `2026-06-13`
- 법령 원천: `legalize-kr/legalize-kr` 로컬 스냅샷 및 git 이력
- 미래 시행본 제외: `시행일자 <= 2026-06-13`인 버전만 사용
- 중복 방지: 제15회 변호사시험 `question_text`와 문자열 유사도 `0.72` 이상인 생성 문항은 제외
- 문항 유형: 조문 직접 확인형, 옳지 않은 설명형, 빈칸형, 근거 조문 선택형

## QA Summary

```json
{
  "target_count": 1000,
  "row_count": 1000,
  "attempts": 1416,
  "subject_counts": {
    "공법": 270,
    "민사법": 460,
    "형사법": 270
  },
  "question_type_counts": {
    "reference_select": 371,
    "negative_statement": 207,
    "article_match": 373,
    "blank_term": 49
  },
  "law_counts_top20": {
    "민법": 218,
    "상법": 153,
    "형사소송법": 151,
    "형법": 100,
    "대한민국헌법": 68,
    "국회법": 58,
    "민사소송법": 46,
    "채무자 회생 및 파산에 관한 법률": 37,
    "행정절차법": 36,
    "헌법재판소법": 34,
    "행정기본법": 29,
    "행정심판법": 23,
    "행정소송법": 18,
    "성폭력범죄의 처벌 등에 관한 특례법": 10,
    "아동ㆍ청소년의 성보호에 관한 법률": 7,
    "국가배상법": 4,
    "부동산등기법": 4,
    "가족관계의 등록 등에 관한 법률": 2,
    "특정범죄 가중처벌 등에 관한 법률": 2
  },
  "round15_similarity_threshold": 0.72,
  "round15_max_similarity": 0.2982,
  "rejected_round15_similarity": 0,
  "rejected_duplicate_fingerprint": 0,
  "duplicate_fingerprints": 0,
  "as_of": "2026-06-13",
  "law_doc_count": 21,
  "article_count": 5247,
  "weighted_article_count": 26648,
  "round15_reference_count": 150,
  "repo_id": "gyung/korean-current-law-bar-exam-sft-1000",
  "bar_questions_path": "data/bar_exam/hf_multiple_choice/data/questions.csv"
}
```

## Sources

- Existing bar-exam reference dataset: https://huggingface.co/datasets/gyung/korean-bar-exam-moj-multiple-choice
- Ministry of Justice, 2026 제15회 변호사시험 기출문제/정답 공지
  - https://www.moj.go.kr/bbs/moj/150/602397/artclView.do
  - https://www.moj.go.kr/bbs/moj/150/602398/artclView.do
  - https://www.moj.go.kr/bbs/moj/150/602399/artclView.do
  - https://www.moj.go.kr/bbs/moj/151/603464/artclView.do
- Korean statutes: https://github.com/legalize-kr/legalize-kr and https://www.law.go.kr
- Related local legal corpora consulted for project context: `ordinance-kr`, `/home/work/.data/huggingface/hrm_text_extra/sft/korean_legal_tasks_full_20260524.jsonl`, `/home/work/.data/huggingface/hrm_text_extra/sft/korean_admrule_precedent_raw_full_20260524.jsonl`

## License

법령 원문은 대한민국 정부 공공저작물입니다. 데이터셋의 패키징, 생성 코드, 메타데이터는 프로젝트 생성물입니다.

이 데이터는 학습/평가용 법률 교육 데이터이며 법률 자문이 아닙니다.

HF repo: https://huggingface.co/datasets/gyung/korean-current-law-bar-exam-sft-1000

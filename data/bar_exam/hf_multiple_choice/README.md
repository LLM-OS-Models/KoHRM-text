---
license: other
language:
- ko
pretty_name: Korean Bar Exam Multiple-Choice Questions and Answers (MOJ)
task_categories:
- question-answering
- text-classification
tags:
- law
- korean-law
- bar-exam
- public-sector
source_datasets:
- original
---

# Korean Bar Exam Multiple-Choice Questions and Answers (MOJ)

대한민국 법무부가 공개한 변호사시험 선택형(다지선다) 기출문제와 공식 정답을 문항 단위로 정리한 데이터셋입니다.

이 데이터셋은 해설 데이터가 아니라 **문제 -> 정답 번호** 학습/평가에 맞춰져 있습니다.

## SFT에 바로 쓸 파일

SFT에 가장 바로 쓰기 좋은 파일은 다음입니다.

```text
data/questions.csv
```

핵심 컬럼:

- `question_text`: 문제 원문과 선택지를 포함한 전체 텍스트
- `stem`: 문제 본문
- `choices_json`: 선택지 배열(JSON 문자열)
- `answer`: 공식 정답 번호
- `round`, `year`, `subject`, `question_no`: 회차, 연도, 과목, 문항 번호
- `source_article_url`, `source_file_url`, `source_license`: 출처와 라이선스 추적 정보

`data/answers.csv`는 정답표 검증용입니다. 단독 SFT 입력으로 쓰기보다는 `questions.csv`와 대조하거나 평가셋을 만들 때 사용하세요.

`data/documents.csv`는 원본 문서 단위 메타데이터입니다.

## SFT 포맷 예시

가장 단순한 SFT 목적은 “변호사시험 선택형 문제를 보고 정답 번호만 답하는 모델”입니다.

예시 메시지 포맷:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "다음 변호사시험 선택형 문제의 정답 번호만 고르시오.\n\n[문제]\n..."
    },
    {
      "role": "assistant",
      "content": "정답: 3"
    }
  ]
}
```

Python 변환 예시:

```python
from datasets import load_dataset

ds = load_dataset(
    "gyung/korean-bar-exam-moj-multiple-choice",
    data_files="data/questions.csv",
    split="train",
)

def to_sft(row):
    user = (
        "다음 변호사시험 선택형 문제의 정답 번호만 고르시오.\n\n"
        f"[회차] 제{row['round']}회\n"
        f"[과목] {row['subject']}\n"
        f"[문항]\n{row['question_text']}"
    )
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": f"정답: {row['answer']}"},
        ]
    }

sft_ds = ds.map(to_sft, remove_columns=ds.column_names)
```

정답 번호만 학습시키고 싶으면 assistant 응답을 `row["answer"]`만 남겨도 됩니다.

```python
{"role": "assistant", "content": row["answer"]}
```

## 평가용 사용

모델 평가에서는 `question_text`를 프롬프트로 넣고 모델이 낸 번호를 `answer`와 비교하면 됩니다.

권장 평가 프롬프트:

```text
다음 변호사시험 선택형 문제의 정답 번호만 출력하시오. 다른 말은 쓰지 마시오.

{question_text}
```

## 데이터 한계

- 공식 해설은 포함되어 있지 않습니다.
- `answer`는 공식 정답 번호입니다.
- 일부 문항은 원문 HWP/HWPX의 서식 때문에 `choices_json` 선택지 분리가 완벽하지 않을 수 있습니다. 그래도 모든 행에는 `question_text`와 `answer`가 들어 있습니다.
- QA 기준으로 문제 2,250행과 정답 2,250행이 일치하며, 회차/과목별 문항 수 불일치는 없습니다.

## Contents

- `data/questions.csv`: question rows with `round`, `year`, `subject`, `question_type`, `question_no`, `question_text`, `choices_json`, and `answer` where an official multiple-choice answer is available.
- `data/answers.csv`: parsed official multiple-choice answer rows.
- `data/documents.csv`: source document manifest with source URLs and SHA-256 hashes.
- `qa_report.json`: parser and count checks.

Covered rounds: 1-15.

Rows:

- Questions: 2250
- Multiple-choice answer rows: 2250
- Source documents: 66

## Source and Attribution

Source: 대한민국 법무부 법조인력과, 변호사시험 기출문제 및 선택형 정답

Primary source pages:

- 기출문제: https://www.moj.go.kr/moj/405/subview.do?enc=Zm5jdDF8QEB8JTJGYmJzJTJGbW9qJTJGMTUwJTJGYXJ0Y2xMaXN0LmRvJTNGdGFibGVfY2F0ZV9zZWxlY3QlM0QxNDElMjZiYnNDbFNlcSUzRDE0MSUyNmlzVmlld01pbmUlM0RmYWxzZSUyNmJic09wZW5XcmRTZXElM0QlMjZzcmNoQ29sdW1uJTNEc2olMjZzcmNoV3JkJTNEJTI2
- 정답가안/확정정답 공지: https://www.moj.go.kr/moj/2126/subview.do?enc=Zm5jdDF8QEB8JTJGYmJzJTJGbW9qJTJGMTUxJTJGYXJ0Y2xMaXN0LmRvJTNGYmJzQ2xTZXElM0QlMjZpc1ZpZXdNaW5lJTNEZmFsc2UlMjZiYnNPcGVuV3JkU2VxJTNEJTI2c3JjaENvbHVtbiUzRHNqJTI2c3JjaFdyZCUzRCVFQiVCMyU4MCVFRCU5OCVCOCVFQyU4MiVBQyVFQyU4QiU5QyVFRCU5NyU5OCslRUMlQTAlOTUlRUIlOEIlQjUlRUElQjAlODglRUMlOTUlODglMjY%3D

Each row includes the source article URL, source file URL, file name, and source SHA-256 where applicable.

## License

The source content is provided by the Ministry of Justice under Korea Open Government License Type 1 (KOGL Type 1).
KOGL Type 1 permits online/offline use, derivative works, and commercial use, provided source attribution is shown and users do not imply endorsement by the public institution.

License summary: https://www.kogl.or.kr/info/licenseType1.do

Dataset packaging and conversion scripts in the originating repository may be Apache-2.0, but the source exam content remains governed by KOGL Type 1 attribution conditions.

## Processing Notes

HWP/HWPX files were converted to text using `pyhwp` (`hwp5html`) for HWP and XML extraction for HWPX. Multiple-choice answer tables were parsed from the converted XHTML table cells. For rounds 13-15, the Ministry of Justice final-answer notices state that the confirmed answers are the same as the posted provisional answer files; those rows are marked `final_notice_same_as_provisional`.

QA summary:

```json
{
  "attachments": 66,
  "documents_by_status": {
    "ok": 66
  },
  "question_rows": 2250,
  "answer_rows": 2250,
  "question_quality_counts": {
    "parsed_with_answer": 2244,
    "needs_review_choices": 6
  },
  "multiple_choice_count_issues": []
}
```

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

This dataset contains question-level rows extracted from official Korean Ministry of Justice 변호사시험 source files.

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

# 에이전트 실행 기록

`claude -p "<질문>" --append-system-prompt "$(cat agents/review_assistant.md)" --allowedTools "Bash(uv run python skills/credit-score-ops/scripts/*)" "Bash(cat *)" "Read" --max-turns 10 --output-format json --no-session-persistence`

| 파일 | 질문 | 확인한 것 |
|---|---|---|
| 01_batch_review.json | 2026-02 배치를 점수 매기고 점검한 뒤, 점수가 흔들렸는지와 담당자가 할 일 | 스크립트 두 개를 실제로 돌리고(6턴) monitor.json 숫자만 인용, 원인 피처 3개와 권고 |
| 02_explain_applicant.json | 기준선 아래인 신청자 한 명의 이유와 「승인해도 될까요」 | 사유 코드 3개를 점수 차이로 설명, 승인 · 거절은 담당자가 한다고 선 그음 |
| 03_calibration.json | 집단별 보정이 어긋난 곳과 「그 숫자로 결정을 바꿔야 하나」 | 29세 이하 +2.5%p 를 점검 대상으로만, 결정 근거로 쓰지 않는다고 답함 |

기록에서 로컬 경로 · 세션 정보 · 이 PC 의 훅이 붙인 안내 줄은 뺐다. 응답 본문은 그대로다.

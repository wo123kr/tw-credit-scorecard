-- 피처 테이블: 고객 1명 = 1행. 원본은 2005년 4~9월 6개월 이력, 라벨은 10월 연체 여부.
-- 달 번호 1 = 9월(가장 최근) … 6 = 4월. 원본의 PAY_0 은 9월 상환 상태라 pay_1 로 읽는다.
-- 상환 상태: -2 사용 없음, -1 전액 납부, 0 리볼빙(최소 금액 납부), 1~9 = 그 달 수만큼 연체.
-- 성별 · 결혼 여부 · 나이는 모형 피처로 쓰지 않는다(사후 공정성 점검에만 남긴다).
CREATE OR REPLACE TABLE features AS
WITH m AS (
  SELECT
    id, is_default, sex, age, limit_bal, education,
    [pay_0, pay_2, pay_3, pay_4, pay_5, pay_6]                         AS st,
    [bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6] AS bill,
    [pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6]       AS paid
  FROM raw_clients
)
SELECT
  id,
  is_default,
  sex,
  age,
  limit_bal,
  -- 학력 코드 0 · 5 · 6 은 설명이 없는 값이라 「기타(4)」로 묶는다
  CASE WHEN education IN (1, 2, 3) THEN education ELSE 4 END              AS education_grp,
  -- 연체 이력
  greatest(st[1], 0)                                                       AS delinq_now,
  greatest(list_max(st), 0)                                                AS delinq_max_6m,
  len(list_filter(st, lambda x: x >= 1))                                        AS delinq_months_6m,
  len(list_filter(st[1:3], lambda x: x >= 1))                                   AS delinq_months_3m,
  len(list_filter(st, lambda x: x = -2))                                        AS idle_months_6m,
  -- 한도 소진율(청구액 ÷ 한도). 음수 청구(과납)는 0 으로 본다
  greatest(bill[1], 0) / limit_bal                                         AS util_now,
  list_avg(list_transform(bill, lambda b: greatest(b, 0))) / limit_bal          AS util_avg_6m,
  list_max(list_transform(bill, lambda b: greatest(b, 0))) / limit_bal          AS util_max_6m,
  (bill[1] - bill[6]) / limit_bal                                          AS bill_trend_6m,
  -- 납부 비율: k 월에 낸 돈 ÷ k+1 월 청구액(직전 청구서). 청구가 없던 달은 빼고 평균, 1 을 넘으면 1
  list_avg(list_transform(
    list_filter(list_zip(paid[1:5], bill[2:6]), lambda z: z[2] > 0),
    lambda z: least(z[1] / z[2], 1.0)))                                         AS pay_ratio_avg_5m,
  -- least() 는 NULL 을 무시하므로, 직전 청구가 없으면 CASE 로 NULL 을 그대로 둔다(1.0 이 되지 않게)
  CASE WHEN bill[2] > 0 THEN least(paid[1] / bill[2], 1.0) END             AS pay_ratio_now,
  len(list_filter(list_zip(paid[1:5], bill[2:6]), lambda z: z[2] > 0 AND z[1] = 0)) AS zero_pay_months_5m,
  list_avg(paid) / limit_bal                                               AS paid_to_limit_6m
FROM m;

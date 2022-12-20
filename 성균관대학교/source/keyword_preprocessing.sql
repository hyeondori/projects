SELECT keyword AS keyword,
		NVL(SUM(2018_count), 0) 															AS 2018_count_sum,
		NVL(SUM(2019_count), 0) 															AS 2019_count_sum,
		NVL(SUM(2020_count), 0) 															AS 2020_count_sum,
		NVL(SUM(2021_count), 0) 															AS 2021_count_sum,
		NVL(SUM(2022_count), 0) 															AS 2022_count_sum,
		NVL(SUM(2018_count), 0) + NVL(SUM(2019_count), 0) + NVL(SUM(2020_count), 0) +
		NVL(SUM(2021_count), 0) + NVL(SUM(2022_count), 0)									AS total,
		NVL(SUM(2020_count), 0) + NVL(SUM(2021_count), 0) + NVL(SUM(2022_count), 0)		AS scale,
		(NVL(SUM(2020_count), 0) + NVL(SUM(2021_count), 0) + NVL(SUM(2022_count), 0)) -
		(NVL(SUM(2018_count), 0) + NVL(SUM(2019_count), 0) + NVL(SUM(2020_count), 0))	AS tendency
  FROM (
		SELECT keyword,
				works_publication_year,
				CASE WHEN (works_publication_year = 2018) THEN count END					AS 2018_count,
				CASE WHEN (works_publication_year = 2019) THEN count END					AS 2019_count,
				CASE WHEN (works_publication_year = 2020) THEN count END					AS 2020_count,
				CASE WHEN (works_publication_year = 2021) THEN count END					AS 2021_count,
				CASE WHEN (works_publication_year = 2022) THEN count END					AS 2022_count
		  FROM (SELECT keyword 															AS keyword,
						works_publication_year												AS works_publication_year,
						COUNT(*)															AS count
				  FROM (SELECT 										
						  works_id															AS works_id,
						  works_title														AS works_title,
						  works_abstract													AS works_abstract,
						  works_publication_year											AS works_publication_year,
						  keyword															AS keyword,
						  score																AS score
						  FROM "openalex"."JAEKYOON_chemical_keybert_test_joined"												-- 원천 테이블(변경 요)
						 WHERE 1=1
						   AND score > 0.5																						-- KeyBert Score 0.5 초과 조건(변경 요)
						) BASE
				 GROUP BY keyword, works_publication_year) BASE_COUNT
		) BASE_COUNT_PIVOT
 GROUP BY keyword
HAVING tendency > 0																												-- 경향성 0 초과
   AND scale > 5																												-- 규모 5 초과
select A01.id
     , COALESCE(SUM(C01.cited_2000), 0) as cited_2000
     , COALESCE(SUM(C01.cited_2001), 0) as cited_2001
     , COALESCE(SUM(C01.cited_2002), 0) as cited_2002
     , COALESCE(SUM(C01.cited_2003), 0) as cited_2003
     , COALESCE(SUM(C01.cited_2004), 0) as cited_2004
     , COALESCE(SUM(C01.cited_2005), 0) as cited_2005
     , COALESCE(SUM(C01.cited_2006), 0) as cited_2006
     , COALESCE(SUM(C01.cited_2007), 0) as cited_2007
     , COALESCE(SUM(C01.cited_2008), 0) as cited_2008
     , COALESCE(SUM(C01.cited_2009), 0) as cited_2009
     , COALESCE(SUM(C01.cited_2010), 0) as cited_2010
     , COALESCE(SUM(C01.cited_2011), 0) as cited_2011
     , COALESCE(SUM(C01.cited_2012), 0) as cited_2012
     , COALESCE(SUM(C01.cited_2013), 0) as cited_2013
     , COALESCE(SUM(C01.cited_2014), 0) as cited_2014
     , COALESCE(SUM(C01.cited_2015), 0) as cited_2015
     , COALESCE(SUM(C01.cited_2016), 0) as cited_2016
     , COALESCE(SUM(C01.cited_2017), 0) as cited_2017
     , COALESCE(SUM(C01.cited_2018), 0) as cited_2018
     , COALESCE(SUM(C01.cited_2019), 0) as cited_2019
     , COALESCE(SUM(C01.cited_2020), 0) as cited_2020
     , COALESCE(SUM(C01.cited_2021), 0) as cited_2021     
  from openalex."JAEKYOON_works_temp" A01
  left join openalex."JAEKYOON_works_referenced_works_temp" B01
    on A01.id = B01.referenced_work_id 
  left join (select id
                  , case when publication_year = '2000' then 1 else 0 end as cited_2000 
                  , case when publication_year = '2001' then 1 else 0 end as cited_2001
                  , case when publication_year = '2002' then 1 else 0 end as cited_2002
                  , case when publication_year = '2003' then 1 else 0 end as cited_2003
                  , case when publication_year = '2004' then 1 else 0 end as cited_2004
                  , case when publication_year = '2005' then 1 else 0 end as cited_2005 
                  , case when publication_year = '2006' then 1 else 0 end as cited_2006
                  , case when publication_year = '2007' then 1 else 0 end as cited_2007
                  , case when publication_year = '2008' then 1 else 0 end as cited_2008
                  , case when publication_year = '2009' then 1 else 0 end as cited_2009
                  , case when publication_year = '2010' then 1 else 0 end as cited_2010
                  , case when publication_year = '2011' then 1 else 0 end as cited_2011 
                  , case when publication_year = '2012' then 1 else 0 end as cited_2012
                  , case when publication_year = '2013' then 1 else 0 end as cited_2013
                  , case when publication_year = '2014' then 1 else 0 end as cited_2014 
                  , case when publication_year = '2015' then 1 else 0 end as cited_2015
                  , case when publication_year = '2016' then 1 else 0 end as cited_2016 
                  , case when publication_year = '2017' then 1 else 0 end as cited_2017
                  , case when publication_year = '2018' then 1 else 0 end as cited_2018
                  , case when publication_year = '2019' then 1 else 0 end as cited_2019 
                  , case when publication_year = '2020' then 1 else 0 end as cited_2020
                  , case when publication_year = '2021' then 1 else 0 end as cited_2021
                  , case when publication_year = '2022' then 1 else 0 end as cited_2022
  			   from openalex."JAEKYOON_works_temp") C01
    on B01.work_id = C01.id
 where 1=1
   and A01.id = 'https://openalex.org/W2002900195'
 group by A01.id
  ;
   
select *
--id
--     , publication_year
  from openalex."JAEKYOON_works_temp"

;
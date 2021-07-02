# Project Title

簡易快速的資料前處理－於PostgreSQL

## Getting Started

### Dependencies

* Mac OS
* Windows 10
* python3.7

### Set Up

* psql 
    ````
    CREATE EXTENSION plpython3u;
    ````


### Installing

```
pip install -r requirment.txt
```

### Executing program

````
psql -f FillMissing.sql
psql -f OutlierDetection.sql
psql -f RFE.sql
psql -f SelectFromModel.sql
psql -f SelectKBest.sql
psql -f SelectPercentile.sql
````

### Usage

* see report.pdf
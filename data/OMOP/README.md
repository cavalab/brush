# SynPUF 1k OMOP CDM Dataset Information

This zip file contains a 1000 person sample of the CMS 2008-2010 Data Entrepreneurs’ Synthetic Public Use File (DE-SynPUF).

[Information on the SynPUF data source is available here](https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/DE_Syn_PUF.html)

The csv data files in this zip file are compatible with OMOP CDM version 5.2.2.

[CDM 5.2.2 DDL for the OHDSI supported DBMSs is available here](https://github.com/OHDSI/CommonDataModel/tree/v5.2.2)

# Data loading instructions

1. Unzip the contents of the zip file. There is one csv file for each database table.
2. The format of each data file is as follows:

	* No header record
	* UTF8 character set
	* Tab field separator
	* CSV
3. Here is an example PostgreSQL DBMS copy statement to load the CDM person csv file with the psql command line client:

```SQL
\COPY PERSON FROM 'person.csv' DELIMITER E'\t' CSV ENCODING 'UTF8';
```

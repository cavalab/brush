from datetime import date, datetime

# Tuple format:
# (Field, Required, Type)
OMOP_DDL = {
    '5.2.2': {
        "PERSON": [
            ("person_id",                    True,   int      ),
            ("gender_concept_id",            True,   int      ),
            ("year_of_birth",                True,   int      ),
            ("month_of_birth",               False,  int      ),
            ("day_of_birth",                 False,  int      ),
            ("birth_datetime",               False,  datetime ),
            ("race_concept_id",              True,   int      ),
            ("ethnicity_concept_id",         True,   int      ),
            ("location_id",                  False,  int      ),
            ("provider_id",                  False,  int      ),
            ("care_site_id",                 False,  int      ),
            ("person_source_value",          False,  str      ),
            ("gender_source_concept_id",     False,  int      ),
            ("race_source_value",            False,  str      ),
            ("race_source_concept_id",       False,  int      ),
            ("ethnicity_source_value",       False,  str      ),
            ("ethnicity_source_concept_id",  False,  int      )
        ],
        "DRUG_EXPOSURE": [
            ("drug_exposure_id",             True,  int      ),
            ("person_id",                    True,  int      ),
            ("drug_concept_id",              True,  int      ),
            ("drug_exposure_start_date",     True,  date     ),
            ("drug_exposure_start_datetime", False, datetime ),
            ("drug_exposure_end_date",       True,  date     ),
            ("drug_exposure_end_datetime",   False, datetime ),
            ("verbatim_end_date",            False, date     ),
            ("drug_type_concept_id",         True,  int      ),
            ("stop_reason",                  False, str      ),
            ("refills",                      False, int      ),
            ("quantity",                     False, float    ),
            ("days_supply",                  False, int      ),
            ("sig",                          False, str      ),
            ("route_concept_id",             False, int      ),
            ("lot_number",                   False, str      ),
            ("provider_id",                  False, int      ),
            ("visit_occurrence_id",          False, int      ),
            ("drug_source_value",            False, str      ),
            ("drug_source_concept_id",       False, int      ),
            ("route_source_value",           False, str      ),
            ("dose_unit_source_value",       False, str      )
        ],
        "MEASUREMENT": [
            ("measurement_id",                 True,   int      ),
            ("person_id",                      True,   int      ),
            ("measurement_concept_id",         True,   int      ),
            ("measurement_date",               True,   date     ),
            ("measurement_datetime",           False,  datetime ),
            ("measurement_type_concept_id",    True,   int      ),
            ("operator_concept_id",            False,  int      ),
            ("value_as_number",                False,  float    ),
            ("value_as_concept_id",            False,  int      ),
            ("unit_concept_id",                False,  int      ),
            ("range_low",                      False,  float    ),
            ("range_high",                     False,  float    ),
            ("provider_id",                    False,  int      ),
            ("visit_occurrence_id",            False,  int      ),
            ("measurement_source_value",       False,  str      ),
            ("measurement_source_concept_id",  False,  int      ),
            ("unit_source_value",              False,  str      ),
            ("value_source_value",             False,  str      )
        ],
        "PROCEDURE_OCCURRENCE_ID": [
            ("procedure_occurrence_id",      True,   int      ),
            ("person_id",                    True,   int      ),
            ("procedure_concept_id",         True,   int      ),
            ("procedure_date",               True,   date     ),
            ("procedure_datetime",           False,  datetime ),
            ("procedure_type_concept_id",    True,   int      ),
            ("modifier_concept_id",          False,  int      ),
            ("quantity",                     False,  int      ),
            ("provider_id",                  False,  int      ),
            ("visit_occurrence_id",          False,  int      ),
            ("procedure_source_value",       False,  str      ),
            ("procedure_source_concept_id",  False,  int      ),
            ("qualifier_source_value",       False,  str      )
        ]
    }
}

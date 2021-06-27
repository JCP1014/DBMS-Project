/**
 * @brief Select Features According to a Percentile of the Highest Scores.
 *
 * @param[in] The name of the table. 
 * @param[in] The name of the target variable .
 * @param[in] The array with names of categorical features, default is null.
 * @param[in] The percent of features to keep, default is 10.
 *
 * @return
 * - The SQL command to select the selected features from the table.
 */

CREATE OR REPLACE FUNCTION SelectPercentile (table_name varchar(63), 
                                            target_name varchar(128),
                                            cat_names varchar(128)[] default null,
                                            percentile int default 10)
RETURNS text
AS $$
    if not 0 <= percentile <= 100:
        raise ValueError("percentile should be >=0, <=100; got %r" % percentile)

    import pandas as pd
    import numpy as np
    import plpy
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import LabelEncoder

    sql_command = "SELECT * FROM " + table_name
    result = plpy.execute(sql_command)
    df = pd.DataFrame.from_records(result)
    df.drop('index',axis=1, inplace=True)

    global cat_names
    if not cat_names:
        cat_names = []
    cat_names += [i for i in df.columns if (df[i].dtype in ['O','bool']) and (not i in cat_names)]
    for col in cat_names:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X = df.iloc[:, df.columns != target_name]
    y = df.iloc[:, df.columns == target_name]
    
    if target_name in cat_names:
        cat_names.remove(target_name)
        cat_indices = [X.columns.get_loc(i) for i in cat_names]
        scores = mutual_info_classif(X, y, cat_indices)
    else:
        cat_indices = [X.columns.get_loc(i) for i in cat_names]
        scores = mutual_info_regression(X, y, cat_indices)

    threshold = np.percentile(scores, 100 - percentile)
    indices = np.where(scores > threshold)
    feature_names = list(X.columns[indices])

    sql_command = ""
    if len(feature_names) > 0:
        sql_command = f'SELECT {feature_names[0]}'
        for i in range(1, len(feature_names)):
            sql_command += f', {feature_names[i]}'
        sql_command += f' FROM {table_name};'

    return sql_command

$$ LANGUAGE plpython3u;
SELECT SelectPercentile('boston', 'PRICE', ARRAY['RAD']);

DROP FUNCTION SelectPercentile(character varying,character varying, character varying[], int);
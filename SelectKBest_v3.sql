/**
 * @brief Select Features According to the K Highest Scores.
 *
 * @param[in] The name of the table.
 * @param[in] The name of the created view.
 * @param[in] The name of the target variable.
 * @param[in] The function taking features and target variable, and returning scores and pvalues, default is 'f_classif'.
 * @param[in] The number of top features to select, default is 10.
 *
 * @return
 * - the list of selected features.
 */

CREATE OR REPLACE FUNCTION SelectKBest (table_name varchar(63),
                                        save_as varchar(63), 
                                        target_name varchar(128),
                                        score_func varchar(22) default 'f_classif', 
                                        k int default 10)
RETURNS text
AS $$
    import pandas as pd
    import numpy as np
    import plpy
    from sklearn.feature_selection import f_classif, mutual_info_classif, chi2, f_regression, mutual_info_regression
    from sklearn.preprocessing import LabelEncoder

    sql_command = "SELECT * FROM " + table_name
    result = plpy.execute(sql_command)
    df = pd.DataFrame.from_records(result)

    cat_names = [i for i in df.columns if (df[i].dtype in ['O'])]
    for col in cat_names:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X = df.iloc[:, df.columns != target_name]
    y = df.iloc[:, df.columns == target_name]

    if not (0 <= k <= X.shape[1]):
        raise ValueError("k should be >=0, <= n_features = %d; got %r. "
                        % (X.shape[1], k))

    if score_func == 'f_classif':
        scores = f_classif(X, y)[0]
    elif score_func == 'mutual_info_classif':
        try:
            scores = mutual_info_classif(X, y)
        except:
            raise TypeError("The target variable should be discrete, " 
                            "but cotinuous value was passed.")
    elif score_func == 'chi2':
        try:
            scores = chi2(X, y)[0]
        except:
            raise TypeError("The target variable should be a class label,"
                            "but numerical value was passed.")
    elif score_func == 'f_regression':
        scores = f_regression(X, y)[0]
    elif score_func == 'mutual_info_regression':
        scores = mutual_info_regression(X, y)
    else:
        raise TypeError("The estimator should be a callable, %s "
                        "was passed."
                        % (score_func))

    indices = np.argsort(scores, kind="mergesort")[-k:]
    feature_names = list(X.columns[indices])

    sql_command = ""
    if len(feature_names) > 0:
        sql_command = 'SELECT '
        for i in range(len(feature_names)):
            sql_command += f'{feature_names[i]}, '
        sql_command += target_name
        sql_command += f' FROM {table_name};'

    sql_command = "CREATE VIEW " + save_as + " AS " + sql_command
    plpy.execute(sql_command)

    return feature_names

$$ LANGUAGE plpython3u;
SELECT SelectKBest('boston', 'selected_view', 'price', 'mutual_info_regression', 4);

DROP FUNCTION SelectKBest(character varying, character varying, character varying, character varying, integer);
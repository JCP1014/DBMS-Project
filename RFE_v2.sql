/**
 * @brief Select Features with Recursive Feature Elimination.
 *
 * @param[in] The name of the table 
 * @param[in] The name of the target variable 
 * @param[in] The name of a supervised learning estimator with a fit method that provides information about feature importance.
 * @param[in] The number of features to select, default is null.
                If null, half of the features are selected. 
                If integer, the parameter is the absolute number of features to select. 
                If float between 0 and 1, it is the fraction of features to select.
 * @param[in] The number of features to remove at each iteration, default is 1.
                If greater than or equal to 1, then step corresponds to the (integer) number of features to remove at each iteration. 
                If within (0.0, 1.0), then step corresponds to the percentage (rounded down) of features to remove at each iteration.
 * @param[in] Controls verbosity of output, default is 0.
 * @param[in] The feature importance getter to return importance for each feature, default is 'auto'.
                If 'auto', uses the feature importance either through a coef_ or feature_importances_ attributes of estimator.
                If callable, overrides the default feature importance getter. 
 *
 * @return
 * - The SQL command to select the selected features from the table.
 */

CREATE OR REPLACE FUNCTION RFE (table_name varchar(63),
                                target_name varchar(128), 
                                estimator_name text, 
                                n_features_to_select real default null, 
                                step real default 1, 
                                verbose_ int default 0, 
                                importance_getter text default 'auto')
RETURNS text
AS $$
    import pandas as pd
    import numpy as np
    import plpy
    from sklearn.feature_selection import RFE
    from sklearn.svm import LinearSVR, LinearSVC
    from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, SGDRegressor, \
                                    LogisticRegression, RidgeClassifier, RidgeClassifierCV, SGDClassifier, \
                                    LogisticRegressionCV, PassiveAggressiveClassifier, Perceptron
    from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor, \
                            DecisionTreeClassifier, ExtraTreeClassifier
    from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor, \
                                AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    try:
        exec(f'global estimator; estimator = {estimator_name}()')
    except:
        raise TypeError("The estimator should be a callable, %s "
                        "was passed."
                        % (estimator_name))

    sql_command = "SELECT * FROM " + table_name
    result = plpy.execute(sql_command)
    df = pd.DataFrame.from_records(result)
    df.drop('index',axis=1, inplace=True)

    cat_names = [i for i in df.columns if (df[i].dtype in ['O'])]
    for col in cat_names:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X = df.iloc[:, df.columns != target_name]
    y = df.iloc[:, df.columns == target_name]

    global n_features_to_select, step
    if n_features_to_select and n_features_to_select.is_integer():
        n_features_to_select = int(n_features_to_select)
    if step.is_integer():
        step = int(step)

    selector = RFE(estimator, n_features_to_select, step, verbose_, importance_getter)
    selector = selector.fit(X, y)
    feature_names = X.columns[selector.get_support()]

    sql_command = ""
    if len(feature_names) > 0:
        sql_command = f'SELECT {feature_names[0]}'
        for i in range(1, len(feature_names)):
            sql_command += f', {feature_names[i]}'
        sql_command += f' FROM {table_name};'

    return sql_command

$$ LANGUAGE plpython3u;
SELECT RFE('boston', 'PRICE', 'GradientBoostingRegressor');

DROP FUNCTION RFE(character varying, character varying, text, real, real, integer, text);
/**
 * @brief Select Feature from Model.
 *
 * @param[in] name of the table 
 * @param[in] The base estimator from which the transformer is built, default is LASSO
 * @param[in] The threshold value to use for feature selection, default is '1e-5', If 'median' (resp. “mean”), then the threshold value is the median (resp. the mean) of the feature importances.
 * @param[in] The maximum number of features to select. To only select based on max_features, set threshold=-np.inf.
 * @param[in] If ‘auto’, uses the feature importance either through a coef_ attribute or feature_importances_ attribute of estimator.
 *
 * @return
 * - the list of selected features.
 */

CREATE OR REPLACE FUNCTION SelectFromModel (table_name varchar(63),
                                            save_as varchar(63),
                                            target_name varchar(128),
                                            estimator_name varchar(63) default 'LASSO',
                                            threshold varchar(63) default null,
                                            max_features INT default null,
                                            importance_getter text default 'auto')
RETURNS text[]
AS $$

    import pandas as pd
    import numpy as np
    import plpy
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LassoCV
    from sklearn.svm import LinearSVC
    from sklearn import preprocessing
    from sklearn.preprocessing import LabelEncoder

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

    lab_enc = preprocessing.LabelEncoder()
    training_scores_encoded = lab_enc.fit_transform(y)
    
    if estimator_name == 'LinearSVC':
        est = LinearSVC(C=0.01, penalty="l2", dual=False).fit(X, training_scores_encoded)
    elif estimator_name == 'LASSO':
        est = LassoCV().fit(X, y)
    else:
        raise TypeError("The estimator should be a callable, %s (%s) "
                        "was passed."
                        % (estimator_name, type(estimator_name)))

    global threshold
    thre = 1e-5
    if threshold == 'median':
        thre = np.mean(est.coef_)
    
    if max_features != None:
        thre = -np.inf

    model = SelectFromModel(est, prefit=True,
                             threshold=thre, 
                             max_features=max_features, 
                             importance_getter=importance_getter)
    feature_names = np.array(df.columns[:-1])
    plpy.log(feature_names)
    feature_names = feature_names[model.get_support()]


    sql_command = "CREATE VIEW " + save_as + " AS " + sql_command
    plpy.execute(sql_command)

    return feature_names

$$ LANGUAGE plpython3u;
SELECT SelectFromModel('boston', 'selected_feature', 'price', 'LASSO');

-- DROP FUNCTION SelectFromModel(character varying,character varying, character varying, integer, character varying);
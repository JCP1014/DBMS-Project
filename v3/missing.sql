CREATE OR REPLACE FUNCTION fill_missing (table_name varchar(63))
RETURNS void
AS $$

    import pandas as pd
    import numpy as np
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder
    import plpy

    def missing_mask(df):
        return df.isna().to_numpy()

    def cat_indices(df):
        cat_types = ['O','bool']
        return [i for i in range(len(df.dtypes)) if df.dtypes[i] in cat_types]

    def fill_target_val(X, target_idx, cat_idx, mis_mask, min_leaf, first_iter):
        non_target_idx = list(set(range(X.shape[1]))-{target_idx})
        cat_feats = cat_idx if len(cat_idx)>0 else None
        if cat_feats:
            cat_feats = [idx-1 if idx>target_idx else idx for idx in cat_feats]
        
        mis_rows = mis_mask[:,target_idx]
        non_mis_rows = (1-mis_rows).astype('bool')
        
        if first_iter:
            X_train = X[non_mis_rows][:,non_target_idx]
            y_train = X[non_mis_rows,target_idx]
        else:
            X_train = X[:,non_target_idx]
            y_train = X[:,target_idx]
        
        model = HistGradientBoostingRegressor(categorical_features=cat_feats, min_samples_leaf=min_leaf).fit(X_train, y_train)
        X[mis_rows, target_idx] = model.predict(X[mis_rows][:,non_target_idx])

    def fill_target_class(X, target_idx, cat_idx, mis_mask, min_leaf, first_iter):
        non_target_idx = list(set(range(X.shape[1]))-{target_idx})
        cat_feats = list(set(cat_idx)-{target_idx}) if len(cat_idx)>1 else None
        if cat_feats:
            cat_feats = [idx-1 if idx>target_idx else idx for idx in cat_feats]
        
        mis_rows = mis_mask[:,target_idx]
        non_mis_rows = (1-mis_rows).astype('bool')
        
        if first_iter:
            X_train = X[non_mis_rows][:,non_target_idx]
            y_train = X[non_mis_rows,target_idx].astype(int)
        else:
            X_train = X[:,non_target_idx]
            y_train = X[:,target_idx].astype(int)
        
        model = HistGradientBoostingClassifier(categorical_features=cat_feats, min_samples_leaf=min_leaf).fit(X_train, y_train)
        X[mis_rows, target_idx] = model.predict(X[mis_rows][:,non_target_idx])

    def fill_missing(data, n_iter=1):
        
        X = data.to_numpy()
            
        min_leaf = len(X)**0.5
        
        cat_idx = cat_indices(data)
        
        ### create missing indicator
        mis_mask = missing_mask(data)
        
        ### replace string with interger in categorical variables, NAN is kept
        if len(cat_idx)>0:
            int_encoders = [LabelEncoder() for idx in cat_idx]
            for i,idx in enumerate(cat_idx):
                mis_rows = mis_mask[:,idx]
                non_mis_rows = (1-mis_rows).astype('bool')
                X[non_mis_rows,idx] = int_encoders[i].fit_transform(X[non_mis_rows,idx])
            
        ### Compute the order. The column with least missing values is filled first.
        ### find non-missing columns, which don't need filling
        n_mis_each_col = np.sum(mis_mask, axis=0)
        n_non_mis_col = np.sum(n_mis_each_col==0)
        order = np.argsort(n_mis_each_col)[n_non_mis_col:] 
        
        ### fill missing values
        for i in order:
            if i in cat_idx:
                fill_target_class(X, i, cat_idx, mis_mask, min_leaf, first_iter=True)
            else:
                fill_target_val(X, i, cat_idx, mis_mask, min_leaf, first_iter=True)

        if n_iter>1:        
            for i in range(n_iter-1):
                for i in order:
                    if i in cat_idx:
                        fill_target_class(X, i, cat_idx, mis_mask, min_leaf, first_iter=False)
                    else:
                        fill_target_val(X, i, cat_idx, mis_mask, min_leaf, first_iter=False)
        
        ### Undo the int-encoding
        for i,idx in enumerate(cat_idx):
            X[:,idx] = int_encoders[i].inverse_transform(X[:,idx].astype(int))   
                
        return X


    sql_command = "SELECT * FROM " + table_name
    result = plpy.execute(sql_command)
    data = pd.DataFrame.from_records(result)
    data.drop('index',axis=1, inplace=True)

    X_filled = fill_missing(data, n_iter=1)

    data_filled = pd.DataFrame(data=X_filled, columns=data.columns)
    data_filled = data_filled.astype(data.dtypes)
    data_filled.to_sql(table_name+'_filled', 'postgresql://postgres:Js123021104@localhost/postgres')

$$ LANGUAGE plpython3u;
SELECT fill_missing('boston_miss');
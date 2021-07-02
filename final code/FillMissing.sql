CREATE OR REPLACE FUNCTION FillMissing (table_name varchar(63), save_as varchar(63),  target varchar(63) default 'NULL')
RETURNS void
AS $$

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from lightgbm import LGBMRegressor, LGBMClassifier
    import warnings
    warnings.filterwarnings("ignore")
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
        
        model = LGBMRegressor(categorical_feature=cat_feats, n_jobs=-1).fit(X_train, y_train)
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
        
        model = LGBMClassifier(categorical_feature=cat_feats, n_jobs=-1).fit(X_train, y_train)
        X[mis_rows, target_idx] = model.predict(X[mis_rows][:,non_target_idx])

    def fill_missing(data, n_iter=1):
        
        X = data.to_numpy()
            
        min_leaf = int(len(X)**0.5)
        
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

    if target != 'NULL':
        X_filled = fill_missing(data.drop(target,axis=1, inplace=False), n_iter=1)
        non_target_col = [c for c in data.columns.to_list() if c!=target]
        data[non_target_col] = X_filled

    else:
        X_filled = fill_missing(data, n_iter=1)
        data[data.columns] = X_filled


    sql_command = 'CREATE TABLE {out_table} (LIKE {in_table});'.format(out_table=save_as, in_table=table_name)
    plpy.execute(sql_command)

    sql_command = 'select column_name, data_type from information_schema.columns where table_name = \'{out_table}\';'.format(out_table=save_as)
    col_name_type = plpy.execute(sql_command)
    col_name_type = pd.DataFrame.from_records(col_name_type)
    col_names = col_name_type['column_name'].tolist()
    col_types = col_name_type['data_type'].tolist()

    ### reorder the columns to match what is return from schema
    data = data.reindex(columns=col_names)
    data = data.to_numpy().tolist()

    sql_command = 'INSERT INTO {out_table} ({c_names}) values ({cols});'.format(out_table=save_as, c_names=','.join(col_names), cols=','.join(['$'+str(i) for i in range(1,len(col_names)+1)]))
    plan = plpy.prepare(sql_command, col_types)
    for row in data:
        plpy.execute(plan, row)


$$ LANGUAGE plpython3u;
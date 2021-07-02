CREATE OR REPLACE FUNCTION OutlierDetection (table_name varchar(63),
                                             save_as varchar(63) default 'NULL' ,
											 method varchar(63) default 'OCSVM',
											 target varchar(63) default 'NULL',
                                             index_column varchar(63) default 'NULL')				
RETURNS SETOF text
AS $$
    import numpy as np
    import pandas as pd
    import plpy

    sql_command = "SELECT * FROM " + table_name
    result = plpy.execute(sql_command)

    new_list = []
    for r in result:
        new_list.append(r)
    
    
    df = pd.DataFrame(data=new_list)
    
    if target == 'NULL':
        X = df.select_dtypes(['number'])
    else:
        X = df.drop([target], axis = 1).select_dtypes(['number'])
    
    if method == 'OCSVM':
        from sklearn.svm import OneClassSVM
        index = OneClassSVM(gamma='auto').fit_predict(X)    
        
    elif method == 'COV':
        from sklearn.covariance import EllipticEnvelope
        index= EllipticEnvelope().fit_predict(X)
        
    elif method == 'IsolationForest':
        from sklearn.ensemble import IsolationForest
        index = IsolationForest().fit_predict(X)
        
    elif method == 'DBSCAN':
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import MinMaxScaler
                   
        minmaxscaler = MinMaxScaler()
        for i in X:
            X[i] = minmaxscaler.fit_transform(np.array(X[i]).reshape(-1,1))

        model = DBSCAN().fit(X)
        index = model.labels_
    else:
        raise TypeError("No such method!")
        
    out_index =  np.array(np.where(index == -1))[0].tolist()
    non_out_index = np.array(np.where(index != -1))[0].tolist()
    df = df.iloc[non_out_index]

    ###save table
    if save_as != 'NULL':
        sql_command = 'CREATE TABLE {out_table} (LIKE {in_table});'.format(out_table=save_as, in_table=table_name)
        plpy.execute(sql_command)

        sql_command = 'select column_name, data_type from information_schema.columns where table_name = \'{out_table}\';'.format(out_table=save_as)
        col_name_type = plpy.execute(sql_command)
        col_name_type = pd.DataFrame.from_records(col_name_type)
        col_names = col_name_type['column_name'].tolist()
        col_types = col_name_type['data_type'].tolist()

        ### reorder the columns to match what is return from schema
        df = df.reindex(columns=col_names)
        df = df.to_numpy().tolist()

        sql_command = 'INSERT INTO {out_table} ({c_names}) values ({cols});'.format(out_table=save_as, c_names=','.join(col_names), cols=','.join(['$'+str(i) for i in range(1,len(col_names)+1)]))
        plan = plpy.prepare(sql_command, col_types)
        for row in df:
            plpy.execute(plan, row)

    ### return outlier's values in index column         
    if index_column=='NULL':
        return out_index
    else:
        return df[index_column].iloc[out_index].to_list()
	
$$ LANGUAGE plpython3u;


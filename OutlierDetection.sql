drop function OutlierDetection;

CREATE OR REPLACE FUNCTION OutlierDetection (table_name varchar(63),
											 method varchar(63) default 'OCSVM',
											 target varchar(63) default 'NULL')				
RETURNS SETOF INT
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
        
    return np.array(np.where(index == -1))[0].tolist()
	
$$ LANGUAGE plpython3u;

--test function
select OutlierDetection('Boston', 'DBSCAN', 'PRICE'); 

--drop outlier with method = 'DBSCAN', target = 'PRICE'
select * from Boston where index not in((select OutlierDetection('Boston', 'OCSVM'))); 


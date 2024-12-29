import time

import numpy as np
import pandas as pd

CSV_FILE_PATH = '/home/nn/workspace/MicroRank/otel-ob11-22/-1128-1130/normal/traces.csv'
root_index = 'root'


def get_span(df, start=None, end=None):
    # startTime >= start  and startTime <= end
    if start and end:
        df = df[(df['Timestamp'] >= start) & (df['Timestamp'] <= end)]
    return df


'''
  Query all the service_operation from the input span_list
  :arg
     span_list: should be a long time span_list to get all operation
  :return
       the operation list and operation list dict
'''


def get_service_operation_list(span_df: pd.DataFrame):
    span_df['operation'] = np.where(
        span_df['serviceName'] != 'ts-ui-dashboard',
        span_df['serviceName'] + '_' + span_df['operationName'],
        span_df['serviceName'] + '_' + span_df['operationName'].str.rsplit('/', n=1).str[0],
    )
    operation_list = span_df['operation'].drop_duplicates().tolist()
    return operation_list


"""
   Calculate the mean of duration and variance for each span_list 
   :arg
       operation_list: contains all operation
       span_list: should be a long time span_list
   :return
       operation dict of the mean of and variance 
       {
           # operation: {mean, variance}
           "Currencyservice_Convert": [600, 3]}
       }   
"""
def calculate_real_duration(parsed_df: pd.DataFrame) -> pd.DataFrame:
    # Sum durations of child spans
    child_duration_sum = parsed_df.groupby('ParentSpanId')['duration'].sum().reset_index()
    child_duration_sum.rename(columns={'duration': 'ChildDurationSum'}, inplace=True)
    
    # Merge with the original DataFrame
    parsed_df = parsed_df.merge(child_duration_sum, left_on='spanID', right_on='ParentSpanId', how='left')
    
    # Fill NaN values with 0 for spans without children
    parsed_df['ChildDurationSum'] = parsed_df['ChildDurationSum'].fillna(0)
    
    # Calculate RealDuration
    parsed_df['RealDuration'] = np.maximum(parsed_df['duration'] - parsed_df['ChildDurationSum'], 0)
    parsed_df['duration'] = parsed_df['RealDuration']
    return parsed_df.drop(columns=['ChildDurationSum', 'RealDuration'])

def get_operation_slo(service_operation_list, span_df: pd.DataFrame):

    # 创建操作名称列
    span_df['operation'] = np.where(
        span_df['serviceName'] != 'ts-ui-dashboard',
        span_df['serviceName'] + '_' + span_df['operationName'],
        span_df['serviceName'] + '_' + span_df['operationName'].str.rsplit('/', n=1).str[0],
    )

    # 过滤 duration > 1e9 的记录 ? 需要吗 先注释掉了
    # span_df = span_df[span_df['duration'] <= 1e9]

    # 过滤 ParentSpanId 为空且非 frontend 服务的记录 ? 需要吗 先注释掉了
    # frontend_filter = (span_df['serviceName'] == 'frontend') & (span_df['SpanKind'].str.lower().str.contains('server'))
    # span_df = span_df[frontend_filter | span_df['ParentSpanId'].notnull()]
    
    span_df = calculate_real_duration(span_df)
    
    # 聚合每个操作的持续时间
    duration_series = span_df.groupby('operation')['duration'].apply(list)

    # 计算均值和标准差
    operation_slo = {
        operation: [
            round(np.mean(durations) / 1000.0, 4) if durations else 0,
            # round(np.percentile(duration_dict[operation], 90) / 1000.0, 4) if durations else 0,
            round(np.std(durations) / 1000.0, 4) if durations else 0,
        ]
        for operation, durations in duration_series.items()
        if operation in service_operation_list
    }
    return operation_slo


'''
   Query the operation and duration in span_list for anormaly detector 
   :arg
       operation_list: contains all operation
       operation_dict:  { "operation1": 1, "operation2":2 ... "operationn": 0, "duration": 666}
       span_list: all the span_list in one anomaly detection interval (1 min or 30s)
   :return
       { 
          traceid: {
              operation1: 1
              operation2: 2
          }
       }
'''


def get_operation_duration_data(operation_list, span_df: pd.DataFrame):
    # Create operation names

    span_df['operationName'] = np.where(
        span_df['serviceName'] != 'ts-ui-dashboard',
        span_df['serviceName'] + '_' + span_df['operationName'],
        span_df['serviceName'] + '_' + span_df['operationName'].str.rsplit('/', n=1).str[0],
    )

    # Group by traceID and operationName
    grouped = span_df.groupby(['traceID', 'operationName']).size().unstack(fill_value=0)

    # Calculate duration for frontend server spans
    duration = span_df.groupby('traceID')['duration'].max()

    # Add duration to grouped DataFrame
    grouped['duration'] = duration

    # Filter based on duration and length
    grouped = grouped.dropna(subset=['duration'])
    grouped = grouped[grouped['duration'] > 0]

    # Convert to dictionary
    operation_dict = grouped.to_dict(orient='index')

    return operation_dict


'''
   Query the pagerank graph
   :arg
       trace_list: anormaly_traceid_list or normaly_traceid_list
       span_list:  异常点前后两分钟 span_list
   
   :return
       operation_operation 存储子节点 Call graph
       operation_operation[operation_name] = [operation_name1 , operation_name1 ] 

       operation_trace 存储trace经过了哪些operation, 右上角 coverage graph
       operation_trace[traceid] = [operation_name1 , operation_name2]

       trace_operation 存储 operation被哪些trace 访问过, 左下角 coverage graph
       trace_operation[operation_name] = [traceid1, traceid2]  
       
       pr_trace: 存储trace id 经过了哪些operation，不去重
       pr_trace[traceid] = [operation_name1 , operation_name2]
'''


def get_pagerank_graph(trace_list, span_df: pd.DataFrame):
    # 过滤 trace_list 中的 traceID
    filtered_df = span_df[span_df['traceID'].isin(trace_list)].copy()

    # 创建 operation_name 列
    filtered_df['operation_name'] = np.where(
        filtered_df['serviceName'] != 'ts-ui-dashboard',
        filtered_df['serviceName'] + '_' + filtered_df['operationName'],
        filtered_df['serviceName'] + '_' + filtered_df['operationName'].str.rsplit('/', n=1).str[0],
    )
    # 构建 operation_operation
    parent_child = filtered_df[['traceID', 'spanID', 'ParentSpanId', 'operation_name']]
    merged = parent_child.merge(parent_child, left_on='ParentSpanId', right_on='spanID', suffixes=('_child', '_parent'))
    operation_operation = merged.groupby('operation_name_parent')['operation_name_child'].apply(list).to_dict()
    all_operations = filtered_df['operation_name'].unique()
    for operation in all_operations:
        if operation not in operation_operation:
            operation_operation[operation] = []
    # 构建 operation_trace
    operation_trace = filtered_df.groupby('traceID')['operation_name'].apply(list).to_dict()
    # 构建 trace_operation
    trace_operation = filtered_df.groupby('operation_name')['traceID'].apply(list).to_dict()
    # 构建 pr_trace
    pr_trace = filtered_df.groupby('traceID')['operation_name'].apply(list).to_dict()

    return operation_operation, operation_trace, trace_operation, pr_trace


if __name__ == '__main__':

    def timestamp(datetime):
        timeArray = time.strptime(datetime, "%Y-%m-%d %H:%M:%S")
        ts = int(time.mktime(timeArray)) * 1000
        # print(ts)
        return ts

    start = '2020-08-28 14:56:43'
    end = '2020-08-28 14:57:44'

    span_df = get_span(CSV_FILE_PATH, start=timestamp(start), end=timestamp(end))
    # print(span_list)
    operation_list = get_service_operation_list(span_df)
    print(operation_list)
    operation_slo = get_operation_slo(operation_list, span_df)
    print(operation_slo)
    operation_dict = get_operation_duration_data(operation_list, span_df)
    # print(operation_dict)
    trace_list = list(operation_dict.keys())
    operation_operation, operation_trace, trace_operation, pr_trace = get_pagerank_graph(trace_list, span_df)
    print(operation_operation)
